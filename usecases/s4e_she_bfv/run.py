"""S4e: Client MiniLM embed -> encrypt BFV (TenSEAL) -> Server similarity -> Client decrypt + rank.

Adapted from code-HI/usecases/uc06_she/local_she_bfv_similarity.py.
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

LOG_PATH, RESULT_PATH = scenario_output_paths("s4e")
PREFIX = "[S4e]"

DEFAULT_SCALE = 2 ** 10


def _auto_poly_modulus_degree(n_elements: int) -> int:
    """Minimum poly_modulus_degree for BFV (TenSEAL):
    - Must be power of 2
    - degree/2 >= n_elements (slot capacity must fit the vector)
    - At least 4096 for practical 128-bit security (TenSEAL auto-selects coeff modulus)
    """
    import math
    min_for_slots = 2 ** math.ceil(math.log2(2 * n_elements))
    degree = max(min_for_slots, 4096)
    print(
        f"[S4e] auto poly_modulus_degree={degree} "
        f"(n_elements={n_elements}, min_for_slots={min_for_slots}, security_min=4096)",
        flush=True,
    )
    return degree


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


def _find_bfv_modulus(scale: int, poly_modulus_degree: int) -> int:
    """Return smallest prime p ≡ 1 (mod 2×poly_modulus_degree) with p/2 > scale^2.

    TenSEAL BFV batching requires plain_modulus ≡ 1 (mod 2×poly_modulus_degree).
    """
    m = 2 * poly_modulus_degree
    min_p = 2 * scale * scale + 1
    k = (min_p - 1) // m
    candidate = (k + 1) * m + 1
    while not _is_prime(candidate):
        candidate += m
    print(
        f"[S4e] BFV plain_modulus={candidate} "
        f"(scale={scale}, poly_mod_degree={poly_modulus_degree}, p ≡ 1 mod {m})",
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


def build_bfv_context(poly_modulus_degree: int, plain_modulus: int):
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError(
            "tenseal is not installed. Install: pip install tenseal"
        ) from exc

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus=plain_modulus,
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    return ts, context


def main() -> None:
    top_k = _get_int_env("S4E_TOP_K", 5) or 5
    scale = _get_int_env("S4E_SCALE", DEFAULT_SCALE) or DEFAULT_SCALE
    poly_modulus_degree = _auto_poly_modulus_degree(EMBED_DIM)
    plain_modulus = _find_bfv_modulus(scale, poly_modulus_degree)

    print(
        f"{PREFIX} top_k={top_k} dim={EMBED_DIM} poly_modulus_degree={poly_modulus_degree} (auto) plain_modulus={plain_modulus} (auto)",
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
    ts, context = build_bfv_context(poly_modulus_degree=poly_modulus_degree, plain_modulus=plain_modulus)
    t_context = time.perf_counter() - t_context_start
    print(f"{PREFIX} context+keys done: {t_context:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    enc_q_tk = ts.bfv_vector(context, q_tk_int.tolist())
    enc_q_abs = ts.bfv_vector(context, q_abs_int.tolist())
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    enc_scores_tk = [enc_q_tk.dot(sp.tolist()) for sp in subprofiles_tk_int]
    enc_scores_abs = [enc_q_abs.dot(sp.tolist()) for sp in subprofiles_abs_int]
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    t_dec_start = time.perf_counter()
    dec_scores_tk_int = np.array([int(x.decrypt()[0]) for x in enc_scores_tk], dtype=np.int64)
    dec_scores_abs_int = np.array([int(x.decrypt()[0]) for x in enc_scores_abs], dtype=np.int64)
    t_dec = time.perf_counter() - t_dec_start

    # BFV mod correction: negative dot products appear as plain_modulus - |val|
    half_mod = plain_modulus // 2
    dec_scores_tk_int = np.where(dec_scores_tk_int > half_mod, dec_scores_tk_int - plain_modulus, dec_scores_tk_int)
    dec_scores_abs_int = np.where(dec_scores_abs_int > half_mod, dec_scores_abs_int - plain_modulus, dec_scores_abs_int)
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    dec_scores_tk = dec_scores_tk_int.astype(np.float32) / float(scale * scale)
    dec_scores_abs = dec_scores_abs_int.astype(np.float32) / float(scale * scale)
    sub_scores = combine_modal_scores(dec_scores_tk, dec_scores_abs)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s4e",
        **format_top_k(author_ids, author_scores, top_k=top_k),
        "timing": {
            "embed_sec": t_embed,
            "context_sec": t_context,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "BFV",
            "backend": "tenseal_bfv",
            "mode": "miniLM_encrypt_server_similarity_client_rank",
            "dim": int(q_tk.shape[0]),
            "poly_modulus_degree": poly_modulus_degree,
            "plain_modulus": plain_modulus,
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
