"""Shared dimension reduction helpers (PCA / TruncatedSVD) for query and profile vectors."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from core.config import l2_normalize

SUPPORTED_REDUCTION_METHODS = {"full", "pca", "svd"}


@dataclass(frozen=True)
class ReductionConfig:
    method: str
    target_dim: int


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _get_str_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def resolve_reduction_config(prefix: str, default_dim: int = 64) -> ReductionConfig:
    """Resolve dimension reduction config from env vars <PREFIX>_REDUCE_METHOD and <PREFIX>_REDUCE_DIM.

    Default method: pca — preserves semantic structure via principal components.
    """
    method = (
        _get_str_env(f"{prefix}_REDUCE_METHOD")
        or _get_str_env("BENCH_REDUCE_METHOD")
        or "pca"
    ).strip().lower()

    if method not in SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f"Unsupported reduction method {method!r} for {prefix}. "
            f"Supported: {sorted(SUPPORTED_REDUCTION_METHODS)}"
        )

    target_dim = (
        _get_int_env(f"{prefix}_REDUCE_DIM")
        or _get_int_env("BENCH_REDUCE_DIM")
        or default_dim
    )
    if target_dim <= 0:
        raise ValueError(f"Invalid target dim for {prefix}: {target_dim}")
    return ReductionConfig(method=method, target_dim=target_dim)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")
    return np.stack([l2_normalize(row) for row in x], axis=0).astype(np.float32)


def _fit_pca_transform(
    matrix: np.ndarray, query_vec: np.ndarray, target_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    n_samples, input_dim = matrix.shape
    effective_dim = min(target_dim, input_dim, n_samples)
    pca = PCA(n_components=effective_dim, random_state=42)
    reduced_matrix = pca.fit_transform(matrix)
    reduced_query = pca.transform(query_vec.reshape(1, -1))[0]
    return reduced_matrix.astype(np.float32), reduced_query.astype(np.float32)


def _fit_svd_transform(
    matrix: np.ndarray, query_vec: np.ndarray, target_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import TruncatedSVD

    n_samples, input_dim = matrix.shape
    effective_dim = min(target_dim, input_dim, n_samples - 1)
    svd = TruncatedSVD(n_components=effective_dim, random_state=42)
    reduced_matrix = svd.fit_transform(matrix)
    reduced_query = svd.transform(query_vec.reshape(1, -1))[0]
    return reduced_matrix.astype(np.float32), reduced_query.astype(np.float32)


def apply_reduction(
    subprofiles_tk: np.ndarray,
    subprofiles_abs: np.ndarray,
    query_tk: np.ndarray,
    query_abs: np.ndarray,
    config: ReductionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply configured reduction to query + profile vectors consistently."""
    method = config.method
    target_dim = config.target_dim

    if method == "full":
        return (
            subprofiles_tk.astype(np.float32),
            subprofiles_abs.astype(np.float32),
            query_tk.astype(np.float32),
            query_abs.astype(np.float32),
        )

    input_dim = int(subprofiles_tk.shape[1]) if subprofiles_tk.ndim == 2 and subprofiles_tk.size else int(query_tk.shape[0])
    if target_dim >= input_dim:
        return (
            subprofiles_tk.astype(np.float32),
            subprofiles_abs.astype(np.float32),
            query_tk.astype(np.float32),
            query_abs.astype(np.float32),
        )

    if method == "pca":
        red_sub_tk, red_q_tk = _fit_pca_transform(subprofiles_tk, query_tk, target_dim)
        red_sub_abs, red_q_abs = _fit_pca_transform(subprofiles_abs, query_abs, target_dim)
        return (
            _normalize_rows(red_sub_tk),
            _normalize_rows(red_sub_abs),
            l2_normalize(red_q_tk).astype(np.float32),
            l2_normalize(red_q_abs).astype(np.float32),
        )

    if method == "svd":
        red_sub_tk, red_q_tk = _fit_svd_transform(subprofiles_tk, query_tk, target_dim)
        red_sub_abs, red_q_abs = _fit_svd_transform(subprofiles_abs, query_abs, target_dim)
        return (
            _normalize_rows(red_sub_tk),
            _normalize_rows(red_sub_abs),
            l2_normalize(red_q_tk).astype(np.float32),
            l2_normalize(red_q_abs).astype(np.float32),
        )

    raise ValueError(f"Unsupported reduction method: {method}")
