"""S4c: Client MiniLM embed -> encrypt Paillier -> Server similarity -> Client decrypt + rank.

Adapted from code-HI/usecases/uc05_phe/local_phe_paillier_similarity.py.
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

LOG_PATH, RESULT_PATH = scenario_output_paths("s4c")
PREFIX = "[S4c]"

DEFAULT_SCALE = 2 ** 11
DEFAULT_KEY_BITS = 2048


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def build_paillier_keypair(key_bits: int):
    try:
        from phe import paillier
    except ImportError as exc:
        raise RuntimeError(
            "phe is not installed. Install: pip install python-paillier"
        ) from exc
    return paillier.generate_paillier_keypair(n_length=key_bits)


def encrypt_vector(public_key, values: np.ndarray):
    return [public_key.encrypt(int(v)) for v in values]


def encrypted_dot(enc_query, plain_weights: np.ndarray):
    acc = None
    for enc_v, w in zip(enc_query, plain_weights):
        term = enc_v * int(w)
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("Cannot compute encrypted dot on empty vectors")
    return acc


def main() -> None:
    top_k = _get_int_env("S4C_TOP_K", 5) or 5
    key_bits = _get_int_env("S4C_KEY_BITS", DEFAULT_KEY_BITS) or DEFAULT_KEY_BITS
    scale = _get_int_env("S4C_SCALE", DEFAULT_SCALE) or DEFAULT_SCALE

    print(
        f"{PREFIX} top_k={top_k} dim={EMBED_DIM} key_bits={key_bits} scale={scale}",
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

    t_keygen_start = time.perf_counter()
    public_key, private_key = build_paillier_keypair(key_bits=key_bits)
    t_keygen = time.perf_counter() - t_keygen_start
    print(f"{PREFIX} keygen done: {t_keygen:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    enc_q_tk = encrypt_vector(public_key, q_tk_int)
    enc_q_abs = encrypt_vector(public_key, q_abs_int)
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    enc_scores_tk = [encrypted_dot(enc_q_tk, sp) for sp in subprofiles_tk_int]
    enc_scores_abs = [encrypted_dot(enc_q_abs, sp) for sp in subprofiles_abs_int]
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    t_dec_start = time.perf_counter()
    dec_scores_tk_int = np.array([private_key.decrypt(x) for x in enc_scores_tk], dtype=np.int64)
    dec_scores_abs_int = np.array([private_key.decrypt(x) for x in enc_scores_abs], dtype=np.int64)
    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    dec_scores_tk = dec_scores_tk_int.astype(np.float32) / float(scale * scale)
    dec_scores_abs = dec_scores_abs_int.astype(np.float32) / float(scale * scale)
    sub_scores = combine_modal_scores(dec_scores_tk, dec_scores_abs)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s4c",
        **format_top_k(author_ids, author_scores, top_k=top_k),
        "timing": {
            "embed_sec": t_embed,
            "keygen_sec": t_keygen,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "PHE_Paillier",
            "mode": "miniLM_encrypt_server_similarity_client_rank",
            "dim": int(q_tk.shape[0]),
            "key_bits": key_bits,
        },
    }
    print(f"{PREFIX} top_k={result['top_k'][:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        ["embed_sec", "keygen_sec", "encrypt_sec", "run_sec", "decrypt_sec", "total_sec"],
        [t_embed, t_keygen, t_enc, t_run, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
