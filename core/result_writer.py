"""Result formatting, output helpers, and accuracy comparison utilities."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def compute_top_k_accuracy(baseline_path: Path, scenario_path: Path) -> dict | None:
    """Compare top-K result of a scenario against baseline.

    Returns dict with:
      - overlap_k: how many author_ids appear in both top-K lists (order ignored)
      - exact_k: how many author_ids are at the exact same rank position
      - k: the top-K size used (min of both lists)
    Returns None if either file is missing or malformed.
    """
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    base_list = baseline.get("top_k", [])
    scen_list = scenario.get("top_k", [])
    if not base_list or not scen_list:
        return None

    k = min(len(base_list), len(scen_list))
    base_ids = [entry["author_id"] for entry in base_list[:k]]
    scen_ids = [entry["author_id"] for entry in scen_list[:k]]

    overlap_k = len(set(base_ids) & set(scen_ids))
    exact_k = sum(1 for b, s in zip(base_ids, scen_ids) if b == s)

    return {"overlap_k": overlap_k, "exact_k": exact_k, "k": k}


def format_top_k(author_ids: list[str], scores: np.ndarray, top_k: int) -> dict:
    """Convert score vector into top-K ranked list format."""
    top_idx = np.argsort(-scores)[:top_k]
    results = [
        {
            "rank": rank,
            "author_idx": int(i),
            "author_id": author_ids[i] if i < len(author_ids) else None,
            "score": float(scores[i]),
        }
        for rank, i in enumerate(top_idx, start=1)
    ]
    return {"top_k": results}


def write_json(path: Path, data: dict) -> None:
    """Write result dictionary to a JSON file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv(path: Path, columns: list[str], timing_values: list[float]) -> None:
    """Append a timing row to CSV log, creating file with headers if needed.

    columns: timing column names (without timestamp_iso, epoch_sec which are prepended)
    timing_values: float values corresponding to each column
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    all_headers = ["timestamp_iso", "epoch_sec"] + columns
    if not path.exists():
        path.write_text(",".join(all_headers) + "\n", encoding="utf-8")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    epoch = f"{time.time():.3f}"
    formatted = [now_iso, epoch] + [f"{v:.6f}" for v in timing_values]
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(formatted) + "\n")
