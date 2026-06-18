"""S2b: CKKS (OpenFHE) with bootstrapping + approx pairwise comparison ranking.

Server pipeline:
1. Streaming CKKS inner product → encrypted author sums (pool limited for bootstrap feasibility)
2. Mean aggregation per author (encrypted scalar mult, level-free combination with weight)
3. Bootstrap each encrypted score (level refresh: ~3.2s per author)
4. Encrypted pairwise comparison via degree-3 polynomial sign approximation
5. Client: decrypt comparison signs → vote-based ranking → top-K

Key thesis contribution: demonstrates that OpenFHE CKKS bootstrapping enables encrypted
comparison that level-limited CKKS (TenSEAL) cannot. Trade-off: each EvalBootstrap =
~3.2s; N=2620 authors → ~17h impractical; N=20 authors → ~3 min feasible.

Scoring: sign(score_i - score_j) counts how many authors author_i "beats".
Sum of wins = vote score. Client decrypts vote signs → sort → top-K.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import WEIGHT_ABS, WEIGHT_TK, scenario_output_paths
from core.data_loader import load_subprofiles_split, select_cluster_medoid_author_ids
from core.embedder import build_query_embeddings
from core.reduction import apply_reduction, resolve_reduction_config
from core.result_writer import append_csv, write_json

LOG_PATH, RESULT_PATH = scenario_output_paths("s2b")
PREFIX = "[S2b]"

# OpenFHE CKKS bootstrap parameters — verified working for 64-slot CKKS
# total_depth=23, level_budget=[4,4], ring_dim=32768, scaling_mod=50
_BOOTSTRAP_LEVEL_BUDGET = [4, 4]
_TOTAL_MULT_DEPTH = 23
_SCALING_MOD_BITS = 50
_FIRST_MOD_BITS = 60
_RING_DIM = 1 << 15  # 32768

# sign(x) ≈ 1.5x - 0.5x^3 (degree-3, accurate for |x| ≤ 1)
# Scores are combined with weight*0.5 factor so differences fit in [-1, 1]
_SIGN_POLY = [0.0, 1.5, 0.0, -0.5]
_COMBINED_SCALE = 0.5  # applied to WEIGHT_* so combined score stays in [-0.5, 0.5]

# Default pool: bootstrapping N author sums costs N×~3.2s; 20 → ~64s feasible
_MAX_AUTHORS_DEFAULT = 20


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _build_bootstrap_context(num_slots: int):
    """Build OpenFHE CKKS crypto context with bootstrapping enabled."""
    try:
        from openfhe import (
            CCParamsCKKSRNS,
            FLEXIBLEAUTO,
            HYBRID,
            HEStd_NotSet,
            GenCryptoContext,
            PKESchemeFeature,
            SecretKeyDist,
        )
    except ImportError as exc:
        raise RuntimeError("openfhe is not installed. Install: pip install openfhe") from exc

    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(HEStd_NotSet)
    params.SetRingDim(_RING_DIM)
    params.SetBatchSize(num_slots)
    params.SetMultiplicativeDepth(_TOTAL_MULT_DEPTH)
    params.SetScalingModSize(_SCALING_MOD_BITS)
    params.SetFirstModSize(_FIRST_MOD_BITS)
    params.SetNumLargeDigits(3)
    params.SetKeySwitchTechnique(HYBRID)
    params.SetScalingTechnique(FLEXIBLEAUTO)
    params.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # EvalSumKeyGen generates all rotation keys needed by EvalInnerProduct / EvalSum
    cc.EvalSumKeyGen(keys.secretKey)

    # Bootstrap setup and keygen (~0.8s)
    cc.EvalBootstrapSetup(_BOOTSTRAP_LEVEL_BUDGET, [0, 0], num_slots)
    cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)

    return cc, keys


def main() -> None:
    max_authors = _get_int_env("S2B_MAX_AUTHORS", _MAX_AUTHORS_DEFAULT)
    max_subprofiles = _get_int_env("S2B_MAX_SUBPROFILES")
    top_k = _get_int_env("S2B_TOP_K", 5) or 5
    reduce_cfg = resolve_reduction_config("S2B", default_dim=64)

    print(
        f"{PREFIX} max_authors={max_authors} top_k={top_k} "
        f"reduce_dim={reduce_cfg.target_dim} pool=cluster_medoid backend=OpenFHE+bootstrap",
        flush=True,
    )

    t_total_start = time.perf_counter()

    # --- Client: embed query ---
    q_tk, q_abs, t_embed = build_query_embeddings(reduce_dim=None, device="cpu")
    print(f"{PREFIX} embed done: {t_embed:.3f}s", flush=True)

    # --- Pool selection: cluster medoid (server-side, query-independent) ---
    t_candidate_start = time.perf_counter()
    candidate_author_ids = select_cluster_medoid_author_ids(max_authors) if max_authors is not None else None
    t_candidate = time.perf_counter() - t_candidate_start
    if candidate_author_ids is not None:
        print(f"{PREFIX} cluster medoid selected {len(candidate_author_ids)} authors in {t_candidate:.3f}s", flush=True)

    # --- Load author subprofiles ---
    try:
        subprofiles_tk, subprofiles_abs, author_ids, sub_to_author, source = load_subprofiles_split(
            reduce_dim=None,
            max_authors=None if candidate_author_ids is not None else max_authors,
            max_subprofiles_per_author=max_subprofiles,
            candidate_author_ids=candidate_author_ids,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR loading data: {exc}", flush=True)
        raise

    if subprofiles_tk.size == 0:
        raise RuntimeError("No author profiles found.")

    num_authors = len(author_ids)
    num_subs = len(subprofiles_tk)
    print(
        f"{PREFIX} source={source} authors={num_authors} subprofiles={num_subs}",
        flush=True,
    )

    if top_k > len(author_ids):
        print(f"{PREFIX} top_k={top_k} reduced to author count {len(author_ids)}", flush=True)
        top_k = len(author_ids)

    # --- Apply PCA reduction (same space for query and profiles) ---
    subprofiles_tk, subprofiles_abs, q_tk, q_abs = apply_reduction(
        subprofiles_tk=subprofiles_tk,
        subprofiles_abs=subprofiles_abs,
        query_tk=q_tk,
        query_abs=q_abs,
        config=reduce_cfg,
    )
    actual_dim = int(q_tk.shape[0])
    # OpenFHE batch size must be a power of 2; round up if PCA gave fewer components
    num_slots = 1 << math.ceil(math.log2(max(actual_dim, 2)))

    # L2-normalize to bound dot products in [-1, 1] (required for sign poly accuracy)
    # apply_reduction already normalizes; this is a safety re-normalize
    q_tk = q_tk / (np.linalg.norm(q_tk) + 1e-12)
    q_abs = q_abs / (np.linalg.norm(q_abs) + 1e-12)
    norms_tk = np.linalg.norm(subprofiles_tk, axis=1, keepdims=True) + 1e-12
    norms_abs = np.linalg.norm(subprofiles_abs, axis=1, keepdims=True) + 1e-12
    subprofiles_tk = subprofiles_tk / norms_tk
    subprofiles_abs = subprofiles_abs / norms_abs

    # Pad to num_slots with zeros if needed (zeros don't affect dot product value)
    if actual_dim < num_slots:
        pad = num_slots - actual_dim
        q_tk = np.pad(q_tk, (0, pad))
        q_abs = np.pad(q_abs, (0, pad))
        subprofiles_tk = np.pad(subprofiles_tk, ((0, 0), (0, pad)))
        subprofiles_abs = np.pad(subprofiles_abs, ((0, 0), (0, pad)))
        print(f"{PREFIX} padded dim {actual_dim}→{num_slots} (next power-of-2 for OpenFHE)", flush=True)

    # --- Build OpenFHE CKKS context with bootstrap support ---
    t_context_start = time.perf_counter()
    cc, keys = _build_bootstrap_context(num_slots)
    t_context = time.perf_counter() - t_context_start
    print(f"{PREFIX} context+bootstrap_keygen: {t_context:.3f}s", flush=True)

    # --- Client: encrypt query vectors ---
    t_enc_start = time.perf_counter()
    enc_q_tk = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(q_tk.tolist()))
    enc_q_abs = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(q_abs.tolist()))
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    # --- Server Phase 1: Streaming inner products → encrypted author sums ---
    # Each EvalInnerProduct is ciphertext × plaintext (subprofile not secret).
    # enc_q_* is never decrypted on server — only subprofile plaintexts are used.
    # Accumulate encrypted sums: no decrypt between subprofiles (preserves privacy for comparison).
    print(f"{PREFIX} server phase1: {num_subs} inner products (streaming)...", flush=True)
    t_run_start = time.perf_counter()

    enc_sums_tk: dict[int, object] = {}
    enc_sums_abs: dict[int, object] = {}
    sub_counts = np.zeros(num_authors, dtype=np.int64)

    for sub_idx in range(num_subs):
        sp_tk = subprofiles_tk[sub_idx].tolist()
        sp_abs = subprofiles_abs[sub_idx].tolist()
        a_idx = int(sub_to_author[sub_idx])

        ct_tk = cc.EvalInnerProduct(enc_q_tk, cc.MakeCKKSPackedPlaintext(sp_tk), num_slots)
        ct_abs = cc.EvalInnerProduct(enc_q_abs, cc.MakeCKKSPackedPlaintext(sp_abs), num_slots)

        if a_idx not in enc_sums_tk:
            enc_sums_tk[a_idx] = ct_tk
            enc_sums_abs[a_idx] = ct_abs
        else:
            enc_sums_tk[a_idx] = cc.EvalAdd(enc_sums_tk[a_idx], ct_tk)
            enc_sums_abs[a_idx] = cc.EvalAdd(enc_sums_abs[a_idx], ct_abs)

        sub_counts[a_idx] += 1

    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} phase1 done: {t_run:.3f}s", flush=True)

    # Mean aggregation with combined weight+scale in one EvalMult (saves one level).
    # Combined = WEIGHT_TK*0.5/count * sum_tk + WEIGHT_ABS*0.5/count * sum_abs
    # → result in [-0.5, 0.5] so diff between two authors ∈ [-1, 1] ✓ for sign poly
    enc_combined: list[object] = []
    for idx in range(num_authors):
        cnt = float(sub_counts[idx]) if sub_counts[idx] > 0 else 1.0
        scale_tk = WEIGHT_TK * _COMBINED_SCALE / cnt
        scale_abs = WEIGHT_ABS * _COMBINED_SCALE / cnt
        ct_tk_mean = cc.EvalMult(enc_sums_tk.get(idx, enc_sums_tk[0]), scale_tk)
        ct_abs_mean = cc.EvalMult(enc_sums_abs.get(idx, enc_sums_abs[0]), scale_abs)
        enc_combined.append(cc.EvalAdd(ct_tk_mean, ct_abs_mean))

    # --- Server Phase 2: Bootstrap all author scores ---
    # Restores level budget so pairwise comparison is possible.
    # Each EvalBootstrap: ~3.2s. For N=20: ~64s. For N=2620: ~2.3h (impractical).
    print(f"{PREFIX} server phase2: bootstrapping {num_authors} scores...", flush=True)
    t_boot_start = time.perf_counter()

    enc_scores: list[object] = []
    for ct in enc_combined:
        enc_scores.append(cc.EvalBootstrap(ct))

    t_boot = time.perf_counter() - t_boot_start
    print(f"{PREFIX} phase2 bootstrap done: {t_boot:.3f}s", flush=True)

    # --- Server Phase 3: Encrypted pairwise comparison ---
    # sign(score_i - score_j) ≈ 1.5*(si-sj) - 0.5*(si-sj)^3  (degree-3 poly)
    # Requires |si - sj| ≤ 1; guaranteed by _COMBINED_SCALE=0.5 above.
    # enc_scores[i] is READ-ONLY here — level stays at 1 throughout all comparisons.
    # ct_diff and ct_sign are temporary (created fresh each pair), no intermediate bootstrap needed.
    n_pairs = num_authors * (num_authors - 1) // 2
    print(f"{PREFIX} server phase3: {n_pairs} encrypted pairwise comparisons...", flush=True)
    t_approx_start = time.perf_counter()

    enc_signs: dict[tuple[int, int], object] = {}
    approx_success = True
    approx_error = ""
    n_cmp_done = 0
    last_pair = (-1, -1)

    try:
        for i in range(num_authors):
            for j in range(i + 1, num_authors):
                last_pair = (i, j)
                ct_diff = cc.EvalSub(enc_scores[i], enc_scores[j])
                enc_signs[(i, j)] = cc.EvalPoly(ct_diff, _SIGN_POLY)
                n_cmp_done += 1
    except Exception as exc:
        approx_success = False
        approx_error = str(exc)
        print(f"{PREFIX} phase3 error at pair {last_pair}: {exc}", flush=True)

    t_approx = time.perf_counter() - t_approx_start
    print(
        f"{PREFIX} phase3 done: {t_approx:.3f}s "
        f"({n_cmp_done}/{n_pairs} pairs, success={approx_success})",
        flush=True,
    )

    # --- Client: decrypt comparison signs → vote ranking ---
    # Server sends enc_signs to client. Client decrypts and determines ranking.
    # Server never sees plaintext similarity scores.
    t_dec_start = time.perf_counter()

    votes = np.zeros(num_authors, dtype=np.float64)
    for (i, j), ct_sign in enc_signs.items():
        sign_val = float(cc.Decrypt(ct_sign, keys.secretKey).GetRealPackedValue()[0])
        votes[i] += sign_val
        votes[j] -= sign_val

    # Fallback: if comparison incomplete, decrypt raw scores for client sort
    dec_scores = np.zeros(num_authors, dtype=np.float64)
    rank_method = "server_approx_comparison"
    if not approx_success or n_cmp_done < n_pairs:
        print(f"{PREFIX} fallback: decrypting {num_authors} raw scores client-side", flush=True)
        for idx in range(num_authors):
            dec_scores[idx] = float(cc.Decrypt(enc_scores[idx], keys.secretKey).GetRealPackedValue()[0])
        top_idx = np.argsort(-dec_scores)[:top_k]
        rank_method = "client_decrypt_fallback"
    else:
        top_idx = np.argsort(-votes)[:top_k]

    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt+rank done: {t_dec:.3f}s method={rank_method}", flush=True)

    rows = [
        {
            "rank": rank,
            "author_idx": int(i),
            "author_id": author_ids[int(i)],
            "score": float(votes[int(i)]) if rank_method == "server_approx_comparison"
                     else float(dec_scores[int(i)]),
        }
        for rank, i in enumerate(top_idx, start=1)
    ]

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s2b",
        "top_k": rows,
        "timing": {
            "embed_sec": t_embed,
            "candidate_sec": t_candidate,
            "context_sec": t_context,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "bootstrap_sec": t_boot,
            "approx_rank_sec": t_approx,
            "decrypt_sec": t_dec,
            "per_query_sec": t_embed + t_enc + t_run + t_boot + t_approx + t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "CKKS",
            "backend": "OpenFHE",
            "mode": "approx_pairwise_comparison_with_bootstrap",
            "dim": num_slots,
            "reduce_method": reduce_cfg.method,
            "max_authors": max_authors,
            "pool_method": "cluster_medoid",
            "candidate_author_ids": author_ids,
            "bootstrap_level_budget": _BOOTSTRAP_LEVEL_BUDGET,
            "n_comparisons_attempted": n_pairs,
            "n_comparisons_done": n_cmp_done,
            "rank_method": rank_method,
            "approx_success": approx_success,
            "approx_error": approx_error if not approx_success else "",
            "note": (
                "OpenFHE CKKS bootstrapping refreshes CKKS level budget after dot products, "
                "enabling encrypted comparison that TenSEAL (no bootstrap) cannot do. "
                f"enc_scores[i] stays at level 1 throughout {n_cmp_done} comparisons (read-only); "
                "no intermediate bootstrap needed. "
                f"N={max_authors}: ~{max_authors*3.2:.0f}s bootstrap feasible. "
                "N=2620: ~17h → impractical. "
                "sign(x) = 1.5x - 0.5x^3 accurate for |x|<=1; "
                "inputs scaled by 0.5 to ensure |diff|<=1."
            ),
        },
    }
    print(f"{PREFIX} top_k={rows[:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        [
            "embed_sec", "candidate_sec", "context_sec", "encrypt_sec", "run_sec",
            "bootstrap_sec", "approx_rank_sec", "decrypt_sec", "total_sec",
        ],
        [t_embed, t_candidate, t_context, t_enc, t_run, t_boot, t_approx, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
