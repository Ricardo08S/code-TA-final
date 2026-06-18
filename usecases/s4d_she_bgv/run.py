"""S4d: Client MiniLM embed -> encrypt BGV (OpenFHE) -> Server similarity -> Client decrypt + rank.

Adapted from code-HI/usecases/uc07_bgv/local_bgv_similarity.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import scenario_output_paths
from core.config import EMBED_DIM
from core.data_loader import load_subprofiles_split
from core.embedder import build_query_embeddings
from core.result_writer import append_csv, format_top_k, write_json
from core.scoring import aggregate_max_scores, combine_modal_scores

LOG_PATH, RESULT_PATH = scenario_output_paths("s4d")
PREFIX = "[S4d]"

DEFAULT_SCALE = 2 ** 10
DEFAULT_MULT_DEPTH = 2


def _is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0 or k % 3 == 0:
        return False
    i = 5
    while i * i <= k:
        if k % i == 0 or k % (i + 2) == 0:
            return False
        i += 6
    return True


# OpenFHE BGV packed encoding requires: plain_modulus ≡ 1 (mod cyclotomic_order).
# For mult_depth=2 with 128-bit security, OpenFHE selects ring dim 16384 → cyclotomic_order=32768.
# Using m=65536 (2^16) covers ring dims up to 32768 and is future-safe.
_BGV_CYCLOTOMIC_M = 65536


def _find_bgv_modulus(scale: int) -> int:
    """Return smallest prime p ≡ 1 (mod _BGV_CYCLOTOMIC_M) with p/2 > scale^2.

    Required by OpenFHE BGV PackedPlaintext NTT encoding.
    Using m=65536 ensures compatibility with ring dims up to 32768.
    """
    min_p = 2 * scale * scale + 1
    k = (min_p - 1) // _BGV_CYCLOTOMIC_M
    candidate = (k + 1) * _BGV_CYCLOTOMIC_M + 1
    while not _is_prime(candidate):
        candidate += _BGV_CYCLOTOMIC_M
    print(
        f"[S4d] BGV plain_modulus={candidate} "
        f"(scale={scale}, NTT-compatible, p ≡ 1 mod {_BGV_CYCLOTOMIC_M})",
        flush=True,
    )
    return candidate


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def build_bgv_context(plain_modulus: int, multiplicative_depth: int, batch_size: int):
    try:
        from openfhe import CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature
    except ImportError as exc:
        raise RuntimeError(
            "openfhe is not installed. Install: pip install openfhe"
        ) from exc

    params = CCParamsBGVRNS()
    params.SetPlaintextModulus(int(plain_modulus))
    params.SetMultiplicativeDepth(int(multiplicative_depth))
    if batch_size > 0 and hasattr(params, "SetBatchSize"):
        params.SetBatchSize(int(batch_size))

    crypto_context = GenCryptoContext(params)
    crypto_context.Enable(PKESchemeFeature.PKE)
    crypto_context.Enable(PKESchemeFeature.KEYSWITCH)
    crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)
    crypto_context.Enable(PKESchemeFeature.ADVANCEDSHE)

    key_pair = crypto_context.KeyGen()
    crypto_context.EvalMultKeyGen(key_pair.secretKey)
    if hasattr(crypto_context, "EvalSumKeyGen"):
        crypto_context.EvalSumKeyGen(key_pair.secretKey)

    return crypto_context, key_pair


def decrypt_scalar_mod(crypto_context, secret_key, ciphertext, plain_modulus: int) -> int:
    pt = crypto_context.Decrypt(ciphertext, secret_key)
    packed = pt.GetPackedValue()
    if not packed:
        return 0
    raw = int(packed[0])
    half = plain_modulus // 2
    if raw > half:
        raw -= plain_modulus
    return raw


def encrypted_dot_scores(
    crypto_context,
    secret_key,
    enc_query,
    subprofiles_int: np.ndarray,
    plain_modulus: int,
) -> np.ndarray:
    scores = np.zeros(subprofiles_int.shape[0], dtype=np.int64)
    dim = int(subprofiles_int.shape[1])
    inner_product_ok = True
    for i in range(subprofiles_int.shape[0]):
        pt_sub = crypto_context.MakePackedPlaintext(subprofiles_int[i].tolist())
        if inner_product_ok:
            try:
                ct_score = crypto_context.EvalInnerProduct(enc_query, pt_sub, dim)
                scores[i] = decrypt_scalar_mod(crypto_context, secret_key, ct_score, plain_modulus)
                continue
            except Exception:
                inner_product_ok = False
        ct_mul = crypto_context.EvalMult(enc_query, pt_sub)
        pt_mul = crypto_context.Decrypt(ct_mul, secret_key)
        vals = np.asarray(pt_mul.GetPackedValue()[:dim], dtype=np.int64)
        vals = np.where(vals > (plain_modulus // 2), vals - plain_modulus, vals)
        scores[i] = int(vals.sum())
    return scores


def main() -> None:
    top_k = _get_int_env("S4D_TOP_K", 5) or 5
    mult_depth = _get_int_env("S4D_MULT_DEPTH", DEFAULT_MULT_DEPTH) or DEFAULT_MULT_DEPTH
    scale = _get_int_env("S4D_SCALE", DEFAULT_SCALE) or DEFAULT_SCALE
    plain_modulus = _find_bgv_modulus(scale)

    print(
        f"{PREFIX} top_k={top_k} dim={EMBED_DIM} plain_modulus={plain_modulus} (auto) mult_depth={mult_depth}",
        flush=True,
    )

    t_total_start = time.perf_counter()

    try:
        subprofiles_tk, subprofiles_abs, author_ids, sub_to_author, source = load_subprofiles_split(
            reduce_dim=None,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR loading data: {exc}", flush=True)
        raise

    if subprofiles_tk.size == 0:
        raise RuntimeError("No author profiles found.")

    print(
        f"{PREFIX} source={source} authors={len(author_ids)} subprofiles={subprofiles_tk.shape[0]}",
        flush=True,
    )

    q_tk, q_abs, t_embed = build_query_embeddings(reduce_dim=None, device="cpu")
    print(f"{PREFIX} embed done: {t_embed:.3f}s", flush=True)

    q_tk_int = np.round(q_tk * scale).astype(np.int64)
    q_abs_int = np.round(q_abs * scale).astype(np.int64)
    subprofiles_tk_int = np.round(subprofiles_tk * scale).astype(np.int64)
    subprofiles_abs_int = np.round(subprofiles_abs * scale).astype(np.int64)

    t_context_start = time.perf_counter()
    crypto_context, key_pair = build_bgv_context(
        plain_modulus=plain_modulus,
        multiplicative_depth=mult_depth,
        batch_size=0,
    )
    t_context = time.perf_counter() - t_context_start
    print(f"{PREFIX} context+keys done: {t_context:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    pt_q_tk = crypto_context.MakePackedPlaintext(q_tk_int.tolist())
    pt_q_abs = crypto_context.MakePackedPlaintext(q_abs_int.tolist())
    enc_q_tk = crypto_context.Encrypt(key_pair.publicKey, pt_q_tk)
    enc_q_abs = crypto_context.Encrypt(key_pair.publicKey, pt_q_abs)
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    dec_scores_tk_int = encrypted_dot_scores(
        crypto_context=crypto_context,
        secret_key=key_pair.secretKey,
        enc_query=enc_q_tk,
        subprofiles_int=subprofiles_tk_int,
        plain_modulus=plain_modulus,
    )
    dec_scores_abs_int = encrypted_dot_scores(
        crypto_context=crypto_context,
        secret_key=key_pair.secretKey,
        enc_query=enc_q_abs,
        subprofiles_int=subprofiles_abs_int,
        plain_modulus=plain_modulus,
    )
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run (incl. decrypt inside) done: {t_run:.3f}s", flush=True)

    t_dec = 0.0

    dec_scores_tk = dec_scores_tk_int.astype(np.float32) / float(scale * scale)
    dec_scores_abs = dec_scores_abs_int.astype(np.float32) / float(scale * scale)
    sub_scores = combine_modal_scores(dec_scores_tk, dec_scores_abs)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s4d",
        **format_top_k(author_ids, author_scores, top_k=top_k),
        "timing": {
            "embed_sec": t_embed,
            "context_sec": t_context,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "per_query_sec": t_embed + t_enc + t_run + t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "BGV",
            "backend": "openfhe_bgvrns",
            "mode": "miniLM_encrypt_server_similarity_client_rank",
            "dim": int(q_tk.shape[0]),
            "plain_modulus": plain_modulus,
            "mult_depth": mult_depth,
        },
    }
    print(f"{PREFIX} top_k={result['top_k'][:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        ["embed_sec", "context_sec", "encrypt_sec", "run_sec", "decrypt_sec", "total_sec"],
        [t_embed, t_context, t_enc, t_run, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
