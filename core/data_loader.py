"""Load author subprofiles from the cache.

Cache path comes from env var AUTHOR_PROFILE_CACHE_PATH.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.config import l2_normalize

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
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    """Return (sub_tk, sub_abs, author_ids, sub_to_author_idx, source)."""
    profiles, source = _load_profiles()
    if not profiles:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty, [], np.array([], dtype=np.int64), source

    author_ids = list(profiles.keys())
    if max_authors is not None:
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
