"""Shared constants for code-final benchmark scenarios."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
EMBED_DIM = 384
WEIGHT_TK = 0.6
WEIGHT_ABS = 0.4

OUTPUT_DIR = ROOT_DIR / "output"
MANUAL_OUTPUT_DIR = OUTPUT_DIR / "manual"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def is_orchestrated_run() -> bool:
    """Return True when a scenario is being invoked by orchestrate.py."""
    return os.environ.get("CODE_FINAL_RUN_CONTEXT") == "orchestrate"


def scenario_output_paths(scenario: str) -> tuple[Path, Path]:
    """Return timing/result paths for a scenario.

    Orchestrated runs write directly into output/runs/<timestamp>/results and
    output/runs/<timestamp>/timing. Direct module runs are treated as
    development/manual runs and isolated under output/manual/<timestamp>_<scenario>/.
    """
    if is_orchestrated_run():
        run_dir_raw = os.environ.get("CODE_FINAL_RUN_DIR")
        if run_dir_raw:
            run_dir = Path(run_dir_raw)
            return (
                run_dir / "timing" / f"timing_{scenario}.csv",
                run_dir / "results" / f"result_{scenario}.json",
            )
        scenario_dir = OUTPUT_DIR
    else:
        timestamp = os.environ.get("CODE_FINAL_MANUAL_RUN_TIMESTAMP")
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.environ["CODE_FINAL_MANUAL_RUN_TIMESTAMP"] = timestamp
        scenario_dir = MANUAL_OUTPUT_DIR / f"{timestamp}_{scenario}"

    return scenario_dir / f"timing_{scenario}.csv", scenario_dir / f"result_{scenario}.json"


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D numpy vector; returns float32."""
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm
