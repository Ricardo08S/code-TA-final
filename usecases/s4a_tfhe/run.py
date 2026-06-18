"""S4a: Client MiniLM embed -> encrypt TFHE -> Server similarity -> Client decrypt + rank.

Adapted from code-HI/usecases/uc02_tfhe/local_fhe_dot_product.py.
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

from core.config import EMBED_DIM, scenario_output_paths
from core.data_loader import load_subprofiles_split
from core.embedder import build_query_embeddings
from core.result_writer import append_csv, format_top_k, write_json
from core.scoring import aggregate_max_scores, combine_modal_scores

LOG_PATH, RESULT_PATH = scenario_output_paths("s4a")
PREFIX = "[S4a]"

SCALE = 2 ** 10


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


def quantize(vec: np.ndarray) -> np.ndarray:
    return np.round(vec * SCALE).astype(np.int16)


def dequantize(score) -> float:
    return float(score) / float(SCALE * SCALE)


def dot_product(q, w):
    return np.sum(q * w)


def encrypt_query(circuit, q_int: np.ndarray):
    dummy_w = np.zeros_like(q_int, dtype=np.int64)
    enc_result = circuit.encrypt(q_int, dummy_w)
    return enc_result[0] if isinstance(enc_result, tuple) else enc_result


def main() -> None:
    top_k = _get_int_env("S4A_TOP_K", 5) or 5
    device = _get_str_env("S4A_SERVER_DEVICE", "cpu")

    use_gpu, resolved_device = _resolve_use_gpu(device)

    print(
        f"{PREFIX} top_k={top_k} dim={EMBED_DIM} device={resolved_device}",
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

    embed_dim = int(q_tk.shape[0])
    q_int_tk = quantize(q_tk).astype(np.int64)
    q_int_abs = quantize(q_abs).astype(np.int64)

    rng = np.random.default_rng(42)
    inputset = [
        (
            rng.integers(-SCALE, SCALE + 1, size=embed_dim, dtype=np.int16),
            rng.integers(-SCALE, SCALE + 1, size=embed_dim, dtype=np.int16),
        )
        for _ in range(10)
    ]

    t_compile_start = time.perf_counter()
    config = fhe.Configuration(
        parameter_selection_strategy="mono",
        global_p_error=0.05,
        use_gpu=use_gpu,
    )
    compiler = fhe.Compiler(dot_product, {"q": "encrypted", "w": "clear"})
    circuit = compiler.compile(inputset, configuration=config)
    t_compile = time.perf_counter() - t_compile_start
    print(f"{PREFIX} compile done: {t_compile:.3f}s", flush=True)

    t_keygen_start = time.perf_counter()
    circuit.keygen()
    t_keygen = time.perf_counter() - t_keygen_start
    print(f"{PREFIX} keygen done: {t_keygen:.3f}s", flush=True)

    t_enc_start = time.perf_counter()
    enc_q_tk = encrypt_query(circuit, q_int_tk)
    enc_q_abs = encrypt_query(circuit, q_int_abs)
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    t_run_start = time.perf_counter()
    scores_tk = []
    scores_abs = []
    for i in range(subprofiles_tk.shape[0]):
        w_int_tk = quantize(subprofiles_tk[i]).astype(np.int64)
        w_int_abs = quantize(subprofiles_abs[i]).astype(np.int64)
        scores_tk.append(circuit.run(enc_q_tk, w_int_tk))
        scores_abs.append(circuit.run(enc_q_abs, w_int_abs))
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    t_dec_start = time.perf_counter()
    dec_scores_tk = [circuit.decrypt(s) for s in scores_tk]
    dec_scores_abs = [circuit.decrypt(s) for s in scores_abs]
    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    dec_scores_tk_f = np.array([dequantize(s) for s in dec_scores_tk], dtype=np.float32)
    dec_scores_abs_f = np.array([dequantize(s) for s in dec_scores_abs], dtype=np.float32)
    sub_scores = combine_modal_scores(dec_scores_tk_f, dec_scores_abs_f)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s4a",
        **format_top_k(author_ids, author_scores, top_k=top_k),
        "timing": {
            "embed_sec": t_embed,
            "compile_sec": t_compile,
            "keygen_sec": t_keygen,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "per_query_sec": t_embed + t_enc + t_run + t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "TFHE",
            "mode": "miniLM_encrypt_server_similarity_client_rank",
            "dim": embed_dim,
            "device": resolved_device,
        },
    }
    print(f"{PREFIX} top_k={result['top_k'][:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        ["embed_sec", "compile_sec", "keygen_sec", "encrypt_sec", "run_sec", "decrypt_sec", "total_sec"],
        [t_embed, t_compile, t_keygen, t_enc, t_run, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
