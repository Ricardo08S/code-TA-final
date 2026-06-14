"""Build query embeddings using sentence-transformers (all-MiniLM-L6-v2)."""

from __future__ import annotations

import os
import re
import time

import numpy as np

from core.config import l2_normalize

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, object] = {}


def _normalize_keywords_text(raw_keywords: str) -> str:
    tokens = re.split(r"[\n,;]+", raw_keywords or "")
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = re.sub(r"\s+", " ", token).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return " ".join(normalized)


def get_query_text() -> tuple[str, str]:
    """Load query text from env vars: QUERY_TITLE, QUERY_KEYWORDS, QUERY_ABSTRACT."""
    title = os.environ.get("QUERY_TITLE", "privacy preserving author recommendation")
    keywords_raw = os.environ.get("QUERY_KEYWORDS", "homomorphic inference")
    keywords = _normalize_keywords_text(keywords_raw)
    abstract = os.environ.get("QUERY_ABSTRACT", "")
    title_keywords_text = f"{title} {keywords}".strip()
    abstract_text = abstract.strip()
    return title_keywords_text, abstract_text


def _get_sentence_model(device: str = "cpu"):
    model = _MODEL_CACHE.get(device)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=device)
        _MODEL_CACHE[device] = model
    return model


def build_query_embeddings(
    reduce_dim: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed query text into normalized title+keywords and abstract vectors.

    Returns (q_tk, q_abs, embed_time_sec).
    """
    t0 = time.perf_counter()
    model = _get_sentence_model(device=device)
    title_keywords_text, abstract_text = get_query_text()

    embed_tk = model.encode([title_keywords_text if title_keywords_text else ""])[0]
    embed_abs = model.encode([abstract_text if abstract_text else ""])[0]

    vec_tk = l2_normalize(np.array(embed_tk))
    vec_abs = l2_normalize(np.array(embed_abs))

    if reduce_dim is not None:
        vec_tk = vec_tk[:reduce_dim]
        vec_abs = vec_abs[:reduce_dim]

    elapsed = time.perf_counter() - t0
    return vec_tk, vec_abs, elapsed
