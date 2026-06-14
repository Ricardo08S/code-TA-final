"""Baseline: plaintext similarity with full 384-dim MiniLM embeddings."""

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
from core.data_loader import load_subprofiles_split
from core.embedder import build_query_embeddings
from core.result_writer import append_csv, format_top_k, write_json
from core.scoring import aggregate_max_scores, combine_modal_scores

LOG_PATH, RESULT_PATH = scenario_output_paths("baseline")
PREFIX = "[baseline]"


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def main() -> None:
    max_authors = _get_int_env("BASELINE_MAX_AUTHORS")
    max_subprofiles = _get_int_env("BASELINE_MAX_SUBPROFILES")
    top_k = _get_int_env("BASELINE_TOP_K", 5) or 5

    print(
        f"{PREFIX} max_authors={max_authors} max_subprofiles={max_subprofiles} top_k={top_k} dim=384",
        flush=True,
    )

    try:
        subprofiles_tk, subprofiles_abs, author_ids, sub_to_author, source = load_subprofiles_split(
            reduce_dim=None,
            max_authors=max_authors,
            max_subprofiles_per_author=max_subprofiles,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR loading data: {exc}", flush=True)
        raise

    if subprofiles_tk.size == 0:
        raise RuntimeError("No author profiles found. Ensure AUTHOR_PROFILE_CACHE_PATH is set.")

    print(
        f"{PREFIX} source={source} authors={len(author_ids)} "
        f"subprofiles={subprofiles_tk.shape[0]} dim={subprofiles_tk.shape[1]}",
        flush=True,
    )

    t_total_start = time.perf_counter()

    q_tk, q_abs, t_embed = build_query_embeddings(reduce_dim=None, device="cpu")
    print(f"{PREFIX} embed done: {t_embed:.3f}s", flush=True)

    t_score_start = time.perf_counter()
    sub_sims_tk = subprofiles_tk @ q_tk
    sub_sims_abs = subprofiles_abs @ q_abs
    sub_scores = combine_modal_scores(sub_sims_tk, sub_sims_abs)
    author_scores = aggregate_max_scores(sub_scores, sub_to_author, len(author_ids))
    t_score = time.perf_counter() - t_score_start
    print(f"{PREFIX} score done: {t_score:.3f}s", flush=True)

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "baseline",
        **format_top_k(author_ids, author_scores, top_k=top_k),
        "timing": {
            "embed_sec": t_embed,
            "score_sec": t_score,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "plaintext",
            "dim": int(q_tk.shape[0]),
            "max_authors": max_authors,
        },
    }
    print(f"{PREFIX} top_k={result['top_k'][:3]}...", flush=True)
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    append_csv(
        LOG_PATH,
        ["embed_sec", "score_sec", "total_sec"],
        [t_embed, t_score, t_total],
    )


if __name__ == "__main__":
    main()
