"""S2a: Client MiniLM embed -> encrypt TFHE -> Server similarity + encrypted top-K ranking -> Client decrypt top-K.

Adapted from code-HI/usecases/uc08_tfhe_gpu_ranking/local_tfhe_gpu_ranking.py
(encrypted_topk_mode=True branch only).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import concrete.fhe as fhe
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import EMBED_DIM, WEIGHT_ABS, WEIGHT_TK, scenario_output_paths
from core.data_loader import load_subprofiles_split, select_query_candidate_author_ids
from core.embedder import build_query_embeddings
from core.reduction import apply_reduction, resolve_reduction_config
from core.result_writer import append_csv, write_json

LOG_PATH, RESULT_PATH = scenario_output_paths("s2a")
PREFIX = "[S2a]"


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _get_str_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip()


def _resolve_use_gpu(device: str) -> tuple[bool, str]:
    d = device.strip().lower()
    if d == "cpu":
        return False, "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import concrete.compiler as cc
                if cc.check_gpu_enabled() and cc.check_gpu_available():
                    return True, "cuda"
            except Exception:
                pass
        return False, "cpu (gpu unavailable)"
    except Exception:
        return False, "cpu (torch unavailable)"


def quantize_float(vec: np.ndarray, scale: int) -> np.ndarray:
    q = np.round(vec * scale).astype(np.int64)
    return np.clip(q, -scale, scale).astype(np.int64)


def select_author_representatives(
    subprofiles_tk: np.ndarray,
    subprofiles_abs: np.ndarray,
    sub_to_author: np.ndarray,
    num_authors: int,
) -> tuple[np.ndarray, np.ndarray]:
    reps_tk = np.zeros((num_authors, subprofiles_tk.shape[1]), dtype=np.float32)
    reps_abs = np.zeros((num_authors, subprofiles_abs.shape[1]), dtype=np.float32)
    for author_idx in range(num_authors):
        mask = sub_to_author == author_idx
        if not np.any(mask):
            continue
        reps_tk[author_idx] = subprofiles_tk[mask].mean(axis=0)
        reps_abs[author_idx] = subprofiles_abs[mask].mean(axis=0)
    return reps_tk, reps_abs


def build_encrypted_topk_circuit(
    author_profiles_concat: np.ndarray,
    top_k: int,
    q_bound: int,
    *,
    use_gpu: bool,
) -> fhe.Circuit:
    num_authors = int(author_profiles_concat.shape[0])
    profiles = author_profiles_concat.astype(np.int64)
    dim = int(profiles.shape[1]) if profiles.ndim == 2 else 1

    # Cauchy-Schwarz bound: |dot(q,p)| ≤ ||q||₂ × ||p||₂
    # Much tighter than per-element × dim for L2-normalized embeddings.
    if profiles.ndim == 2 and profiles.shape[0] > 0:
        profile_norms = np.sqrt((profiles.astype(np.float64) ** 2).sum(axis=1))
        max_profile_norm = float(profile_norms.max())
        q_l2_bound = float(q_bound) * float(dim) ** 0.5
        max_dot = int(np.ceil(max_profile_norm * q_l2_bound * 1.2))  # 20% safety margin
    else:
        max_abs = int(np.max(np.abs(profiles))) if profiles.size else 1
        max_dot = max_abs * max(1, q_bound) * max(1, dim)
    min_score = -(max_dot + 1)

    def encrypted_topk(query_vec):
        scores = [np.sum(query_vec * profiles[i]) for i in range(num_authors)]
        best_indices = [np.int64(0) for _ in range(top_k)]
        best_scores = [np.int64(min_score) for _ in range(top_k)]

        for t in range(top_k):
            # Arithmetic selection sort: avoids multi-input TLU that CMUX produces.
            # Each comparison is (encrypted_score > encrypted_best) which Concrete can
            # fuse as long as both sides trace back to a single subgraph.
            # Starting from scores[0] keeps the first comparison against a known value.
            best_score = scores[0]
            best_idx = np.int64(0)

            for i in range(1, num_authors):
                is_better = scores[i] > best_score
                best_score = best_score + is_better * (scores[i] - best_score)
                best_idx = best_idx + is_better * (np.int64(i) - best_idx)

            best_indices[t] = best_idx
            best_scores[t] = best_score

            # Mark selected author: set its score to min_score so it won't be picked again.
            # Arithmetic: scores[i] -= (best_idx == i) * (scores[i] - min_score)
            for i in range(num_authors):
                is_selected = best_idx == np.int64(i)
                scores[i] = scores[i] - is_selected * (scores[i] - np.int64(min_score))

        return fhe.array(best_indices), fhe.array(best_scores)

    rng = np.random.default_rng(42)
    inputset = [
        rng.integers(-q_bound, q_bound + 1, size=(author_profiles_concat.shape[1],), dtype=np.int16)
        for _ in range(8)
    ]
    config = fhe.Configuration(
        parameter_selection_strategy="mono",
        global_p_error=0.05,
        use_gpu=use_gpu,
    )
    compiler = fhe.Compiler(encrypted_topk, {"query_vec": "encrypted"})
    return compiler.compile(inputset, configuration=config)


def main() -> None:
    max_authors = _get_int_env("S2A_MAX_AUTHORS", 16)
    max_subprofiles = _get_int_env("S2A_MAX_SUBPROFILES")
    top_k = _get_int_env("S2A_TOP_K", 5) or 5
    reduce_cfg = resolve_reduction_config("S2A", default_dim=64)
    device = _get_str_env("S2A_SERVER_DEVICE", "cpu")
    enc_topk_scale = _get_int_env("S2A_ENC_TOPK_SCALE", 1) or 1
    candidate_mode = _get_str_env("S2A_CANDIDATE_MODE", "first").lower()

    if top_k > 15:
        raise RuntimeError(f"{PREFIX} S2a supports top_k <= 15, got {top_k}.")

    num_authors_limit = max_authors or 0
    if num_authors_limit > 32:
        raise RuntimeError(
            f"{PREFIX} Encrypted top-k mode supports up to 32 authors. "
            "Set S2A_MAX_AUTHORS <= 32."
        )

    use_gpu, resolved_device = _resolve_use_gpu(device)

    print(
        f"{PREFIX} max_authors={max_authors} top_k={top_k} "
        f"reduce_dim={reduce_cfg.target_dim} reduce_method={reduce_cfg.method} "
        f"enc_topk_scale={enc_topk_scale} candidate_mode={candidate_mode} device={resolved_device}",
        flush=True,
    )

    t_total_start = time.perf_counter()

    q_tk, q_abs, t_embed = build_query_embeddings(reduce_dim=None, device="cpu")
    print(f"{PREFIX} embed done: {t_embed:.3f}s", flush=True)

    candidate_author_ids = None
    t_candidate = 0.0
    if candidate_mode == "client_prefilter" and max_authors is not None:
        t_candidate_start = time.perf_counter()
        candidate_author_ids, _ = select_query_candidate_author_ids(
            query_tk=q_tk,
            query_abs=q_abs,
            max_authors=max_authors,
            max_subprofiles_per_author=max_subprofiles,
        )
        t_candidate = time.perf_counter() - t_candidate_start
        print(
            f"{PREFIX} client prefilter selected {len(candidate_author_ids)} authors "
            f"in {t_candidate:.3f}s",
            flush=True,
        )
    elif candidate_mode not in {"first", "client_prefilter"}:
        raise ValueError(
            f"{PREFIX} Unsupported S2A_CANDIDATE_MODE={candidate_mode!r}. "
            "Use 'client_prefilter' or 'first'."
        )

    try:
        subprofiles_tk, subprofiles_abs, author_ids, sub_to_author, source = load_subprofiles_split(
            reduce_dim=None,
            max_authors=max_authors if candidate_author_ids is None else None,
            max_subprofiles_per_author=max_subprofiles,
            candidate_author_ids=candidate_author_ids,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR loading data: {exc}", flush=True)
        raise

    if subprofiles_tk.size == 0:
        raise RuntimeError("No author profiles found.")

    if len(author_ids) > 32:
        raise RuntimeError(
            f"{PREFIX} Encrypted top-k mode supports up to 32 candidate authors, "
            f"got {len(author_ids)}. Set S2A_MAX_AUTHORS <= 32."
        )

    print(
        f"{PREFIX} source={source} authors={len(author_ids)} subprofiles={subprofiles_tk.shape[0]}",
        flush=True,
    )

    if top_k > len(author_ids):
        print(f"{PREFIX} top_k={top_k} reduced to author count {len(author_ids)}", flush=True)
        top_k = len(author_ids)

    subprofiles_tk, subprofiles_abs, q_tk, q_abs = apply_reduction(
        subprofiles_tk=subprofiles_tk,
        subprofiles_abs=subprofiles_abs,
        query_tk=q_tk,
        query_abs=q_abs,
        config=reduce_cfg,
    )

    reps_tk, reps_abs = select_author_representatives(
        subprofiles_tk=subprofiles_tk,
        subprofiles_abs=subprofiles_abs,
        sub_to_author=sub_to_author,
        num_authors=len(author_ids),
    )

    q_tk_i = quantize_float(q_tk, scale=enc_topk_scale)
    q_abs_i = quantize_float(q_abs, scale=enc_topk_scale)
    q_concat = np.concatenate([q_tk_i, q_abs_i]).astype(np.int64)

    reps_tk_i = quantize_float(reps_tk, scale=enc_topk_scale)
    reps_abs_i = quantize_float(reps_abs, scale=enc_topk_scale)
    profiles_concat = np.concatenate([reps_tk_i, reps_abs_i], axis=1).astype(np.int64)

    q_bound = max(1, int(enc_topk_scale))

    t_compile_start = time.perf_counter()
    try:
        topk_circuit = build_encrypted_topk_circuit(
            author_profiles_concat=profiles_concat,
            top_k=top_k,
            q_bound=q_bound,
            use_gpu=use_gpu,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR compiling circuit: {exc}", flush=True)
        raise
    topk_circuit.keygen()
    t_compile = time.perf_counter() - t_compile_start
    print(f"{PREFIX} compile+keygen done: {t_compile:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    enc_q = topk_circuit.encrypt(q_concat.astype(np.int16))
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    enc_top_idx, enc_top_scores = topk_circuit.run(enc_q)
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    t_dec_start = time.perf_counter()
    dec_top_idx, dec_top_scores = topk_circuit.decrypt((enc_top_idx, enc_top_scores))
    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    rows = []
    for rank, (a_idx, a_score) in enumerate(zip(dec_top_idx.tolist(), dec_top_scores.tolist()), start=1):
        idx = int(a_idx)
        if 0 <= idx < len(author_ids):
            rows.append({
                "rank": rank,
                "author_idx": idx,
                "author_id": author_ids[idx],
                "score": float(a_score),
            })

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s2a",
        "top_k": rows,
        "timing": {
            "embed_sec": t_embed,
            "candidate_sec": t_candidate,
            "compile_sec": t_compile,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "TFHE",
            "mode": "miniLM_enc_topk_ranking",
            "dim": int(q_tk.shape[0]),
            "reduce_method": reduce_cfg.method,
            "max_authors": max_authors,
            "candidate_mode": candidate_mode,
            "candidate_author_ids": author_ids,
            "candidate_privacy_note": (
                "client_prefilter does not send plaintext query text or query embedding to the server, "
                "but selected author IDs are visible to the server as access-pattern leakage."
            ) if candidate_mode == "client_prefilter" else "",
            "device": resolved_device,
            "enc_topk_scale": enc_topk_scale,
        },
    }
    print(f"{PREFIX} top_k={rows[:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        ["embed_sec", "candidate_sec", "compile_sec", "encrypt_sec", "run_sec", "decrypt_sec", "total_sec"],
        [t_embed, t_candidate, t_compile, t_enc, t_run, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
