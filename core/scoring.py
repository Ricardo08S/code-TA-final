"""Shared scoring helpers for all scenarios."""

from __future__ import annotations

import numpy as np

from core.config import WEIGHT_ABS, WEIGHT_TK


def combine_modal_scores(
    scores_tk: np.ndarray,
    scores_abs: np.ndarray,
    weight_tk: float = WEIGHT_TK,
    weight_abs: float = WEIGHT_ABS,
) -> np.ndarray:
    """Combine title+keyword and abstract scores using shared weights."""
    tk = np.asarray(scores_tk, dtype=np.float32)
    abs_ = np.asarray(scores_abs, dtype=np.float32)
    return (weight_tk * tk) + (weight_abs * abs_)


def aggregate_max_scores(
    sub_scores: np.ndarray,
    sub_to_author: np.ndarray,
    num_authors: int,
    fill_value: float = -1.0,
) -> np.ndarray:
    """Aggregate subprofile scores into author scores using max-over-subprofiles."""
    author_scores = np.full(num_authors, fill_value, dtype=np.float32)
    np.maximum.at(author_scores, sub_to_author, sub_scores)
    return author_scores
