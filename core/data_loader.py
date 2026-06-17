"""Load author subprofiles from the cache.

Cache path comes from env var AUTHOR_PROFILE_CACHE_PATH.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.config import WEIGHT_ABS, WEIGHT_TK, l2_normalize

try:
    import torch
except Exception:
    torch = None


def _to_numpy(vec) -> np.ndarray:
    if hasattr(vec, "detach"):
        return vec.detach().cpu().numpy()
    return np.asarray(vec)


def _reduce_dim(vec: np.ndarray, target_dim: int | None) -> np.ndarray:
    if target_dim is None:
        return vec
    if vec.shape[0] <= target_dim:
        return vec
    return vec[:target_dim]


def _require_cache_path() -> Path:
    raw = (os.environ.get("AUTHOR_PROFILE_CACHE_PATH") or "").strip()
    if not raw:
        raise RuntimeError(
            "Missing required env var: AUTHOR_PROFILE_CACHE_PATH. "
            "Set it in your .env file or pass it as an environment variable."
        )
    return Path(raw).expanduser()


def _load_profiles() -> tuple[Dict | None, str]:
    cache_path = _require_cache_path()

    # Try pickle first (streamlit cache format)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, dict):
                profiles = cache.get("author_profiles")
                if profiles:
                    return profiles, f"streamlit_cache:{cache_path}"
                # Cache itself is author_profiles dict
                if any(True for _ in cache.values()):
                    return cache, f"pickle_cache:{cache_path}"
        except Exception:
            pass

    # Try torch
    if torch is not None and cache_path.exists():
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                profiles = data.get("author_profiles")
                if profiles:
                    return profiles, f"torch_streamlit_cache:{cache_path}"
                return data, f"torch_cache:{cache_path}"
        except Exception as exc:
            raise RuntimeError(f"Failed to load author profile cache: {cache_path}: {exc}") from exc

    if not cache_path.exists():
        raise RuntimeError(f"AUTHOR_PROFILE_CACHE_PATH does not exist: {cache_path}")

    raise RuntimeError(f"Cannot load profiles from cache: {cache_path}")


def load_subprofiles_split(
    reduce_dim: int | None = None,
    max_authors: int | None = None,
    max_subprofiles_per_author: int | None = None,
    candidate_author_ids: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    """Return (sub_tk, sub_abs, author_ids, sub_to_author_idx, source)."""
    profiles, source = _load_profiles()
    if not profiles:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty, [], np.array([], dtype=np.int64), source

    if candidate_author_ids is not None:
        author_ids = [author_id for author_id in candidate_author_ids if author_id in profiles]
        source = f"{source}:candidate_prefilter"
    else:
        author_ids = list(profiles.keys())

    if candidate_author_ids is None and max_authors is not None:
        author_ids = author_ids[:max_authors]

    vectors_tk: List[np.ndarray] = []
    vectors_abs: List[np.ndarray] = []
    sub_to_author: List[int] = []
    used_author_ids: List[str] = []

    for author_id in author_ids:
        data = profiles.get(author_id, {})
        sub_profiles = data.get("sub_profiles", []) or []
        if max_subprofiles_per_author is not None:
            sub_profiles = sub_profiles[:max_subprofiles_per_author]

        had_sub = False
        for sp in sub_profiles:
            emb_tk = _to_numpy(sp["embed_tk"]).reshape(-1)
            emb_abs = _to_numpy(sp["embed_abs"]).reshape(-1)
            emb_tk = _reduce_dim(emb_tk, reduce_dim)
            emb_abs = _reduce_dim(emb_abs, reduce_dim)
            emb_tk = l2_normalize(emb_tk)
            emb_abs = l2_normalize(emb_abs)
            vectors_tk.append(emb_tk.astype(np.float32))
            vectors_abs.append(emb_abs.astype(np.float32))
            sub_to_author.append(len(used_author_ids))
            had_sub = True
        if had_sub:
            used_author_ids.append(author_id)

    if not vectors_tk:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty, [], np.array([], dtype=np.int64), source

    return (
        np.stack(vectors_tk, axis=0),
        np.stack(vectors_abs, axis=0),
        used_author_ids,
        np.array(sub_to_author, dtype=np.int64),
        source,
    )


def select_cluster_medoid_author_ids(n_clusters: int) -> list[str]:
    """Select n_clusters author IDs as cluster medoids (server-side, query-independent pool).

    Uses KMeans on mean TK author embeddings. Deterministic (random_state=42).
    Each cluster contributes exactly one representative — the author closest to its centroid.
    """
    from sklearn.cluster import KMeans

    profiles, _ = _load_profiles()
    if not profiles:
        return []

    author_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for author_id, data in profiles.items():
        sub_profiles = data.get("sub_profiles", []) or []
        if not sub_profiles:
            continue
        embs = np.stack([l2_normalize(_to_numpy(sp["embed_tk"]).reshape(-1)) for sp in sub_profiles])
        mean_emb = l2_normalize(embs.mean(axis=0))
        author_ids.append(author_id)
        embeddings.append(mean_emb)

    X = np.array(embeddings, dtype=np.float32)
    actual_k = min(n_clusters, len(author_ids))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    kmeans.fit(X)

    medoid_ids: list[str] = []
    for k in range(actual_k):
        cluster_idxs = np.where(kmeans.labels_ == k)[0]
        centroid = kmeans.cluster_centers_[k]
        dists = np.linalg.norm(X[cluster_idxs] - centroid, axis=1)
        medoid_ids.append(author_ids[cluster_idxs[np.argmin(dists)]])
    return medoid_ids


def select_query_candidate_author_ids(
    *,
    query_tk: np.ndarray,
    query_abs: np.ndarray,
    max_authors: int,
    max_subprofiles_per_author: int | None = None,
    weight_tk: float = WEIGHT_TK,
    weight_abs: float = WEIGHT_ABS,
) -> tuple[List[str], np.ndarray]:
    """Select top author IDs using client-side plaintext query embeddings.

    This helper is intended for privacy-preserving scenarios that need a small
    circuit pool. It avoids sending plaintext query text or query embedding to
    the server, but the selected author IDs are still an access-pattern leak.
    """
    if max_authors <= 0:
        return [], np.array([], dtype=np.float32)

    profiles, _ = _load_profiles()
    if not profiles:
        return [], np.array([], dtype=np.float32)

    q_tk = l2_normalize(np.asarray(query_tk).reshape(-1))
    q_abs = l2_normalize(np.asarray(query_abs).reshape(-1))

    scored: list[tuple[str, float]] = []
    for author_id, data in profiles.items():
        sub_profiles = data.get("sub_profiles", []) or []
        if max_subprofiles_per_author is not None:
            sub_profiles = sub_profiles[:max_subprofiles_per_author]
        if not sub_profiles:
            continue

        best_score: float | None = None
        for sp in sub_profiles:
            emb_tk = l2_normalize(_to_numpy(sp["embed_tk"]).reshape(-1))
            emb_abs = l2_normalize(_to_numpy(sp["embed_abs"]).reshape(-1))
            score = (weight_tk * float(emb_tk @ q_tk)) + (weight_abs * float(emb_abs @ q_abs))
            if best_score is None or score > best_score:
                best_score = score

        if best_score is not None:
            scored.append((author_id, best_score))

    scored.sort(key=lambda row: row[1], reverse=True)
    top = scored[:max_authors]
    return [author_id for author_id, _ in top], np.array([score for _, score in top], dtype=np.float32)
