"""S4b: Client MiniLM embed -> encrypt CKKS -> Server similarity -> Client decrypt + rank.

Adapted from code-HI/usecases/uc03_ckks/local_ckks_similarity.py.
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

from core.config import EMBED_DIM, scenario_output_paths
from core.data_loader import load_subprofiles_split
from core.embedder import build_query_embeddings
from core.result_writer import append_csv, format_top_k, write_json
from core.scoring import aggregate_max_scores, combine_modal_scores

LOG_PATH, RESULT_PATH = scenario_output_paths("s4b")
PREFIX = "[S4b]"


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _build_ckks_context():
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError(
            "tenseal is not installed. Install: pip install tenseal"
        ) from exc

    poly_modulus_degree = _get_int_env("S4B_POLY_MODULUS_DEGREE", 8192) or 8192
    global_scale_bits = _get_int_env("S4B_GLOBAL_SCALE_BITS", 40) or 40
    coeff_mod_raw = os.environ.get("S4B_COEFF_MOD_BIT_SIZES", "60,40,40,60")
    coeff_mod_bit_sizes = [int(x.strip()) for x in coeff_mod_raw.split(",") if x.strip()]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = 2 ** global_scale_bits
    context.generate_galois_keys()
    return ts, context


def main() -> None:
    top_k = _get_int_env("S4B_TOP_K", 5) or 5

    print(
        f"{PREFIX} top_k={top_k} dim={EMBED_DIM}",
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

    t_context_start = time.perf_counter()
    ts, context = _build_ckks_context()
    t_context = time.perf_counter() - t_context_start
    print(f"{PREFIX} context+keys done: {t_context:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    enc_q_tk = ts.ckks_vector(context, q_tk.tolist())
    enc_q_abs = ts.ckks_vector(context, q_abs.tolist())
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    enc_scores_tk = [enc_q_tk.dot(sp.tolist()) for sp in subprofiles_tk]
    enc_scores_abs = [enc_q_abs.dot(sp.tolist()) for sp in subprofiles_abs]
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    t_dec_start = time.perf_counter()
    dec_scores_tk = np.array([x.decrypt()[0] for x in enc_scores_tk], dtype=np.float32)
    dec_scores_abs = np.array([x.decrypt()[0] for x in enc_scores_abs], dtype=np.float32)
    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    sub_scores = combine_modal_scores(dec_scores_tk, dec_scores_abs)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s4b",
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
            "scheme": "CKKS",
            "mode": "miniLM_encrypt_server_similarity_client_rank",
            "dim": int(q_tk.shape[0]),
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
