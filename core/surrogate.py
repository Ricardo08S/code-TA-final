"""Surrogate embedding and TFHE circuit utilities.

Adapted from code-TFHE/common/surrogate.py.
Added: build_encrypted_surrogate_scores_circuit() for S3 (Phase 1 only, no top-K ranking).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import concrete.fhe as fhe
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge, RidgeCV

from core.config import WEIGHT_ABS, WEIGHT_TK


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data helpers (local, avoiding circular imports)
# ---------------------------------------------------------------------------

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


def _get_query_text() -> tuple[str, str]:
    title = os.environ.get("QUERY_TITLE", "privacy preserving author recommendation")
    keywords_raw = os.environ.get("QUERY_KEYWORDS", "homomorphic inference")
    keywords = _normalize_keywords_text(keywords_raw)
    abstract = os.environ.get("QUERY_ABSTRACT", "")
    return f"{title} {keywords}".strip(), abstract.strip()


def _load_author_profiles() -> dict:
    raw = (os.environ.get("AUTHOR_PROFILE_CACHE_PATH") or "").strip()
    if not raw:
        raise RuntimeError("AUTHOR_PROFILE_CACHE_PATH is required.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"AUTHOR_PROFILE_CACHE_PATH does not exist: {path}")

    import pickle
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            profiles = data.get("author_profiles")
            if profiles:
                return profiles
            return data
    except Exception:
        pass

    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            profiles = data.get("author_profiles")
            if profiles:
                return profiles
            return data
    except Exception as exc:
        raise RuntimeError(f"Cannot load author profiles from {path}: {exc}") from exc

    raise RuntimeError(f"Cannot load author profiles from {path}")


def _to_numpy(vec) -> np.ndarray:
    if hasattr(vec, "detach"):
        return vec.detach().cpu().numpy()
    return np.asarray(vec)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    x = vec.astype(np.float32)
    norm = np.linalg.norm(x) + 1e-12
    return x / norm


def _build_subprofile_text(sp: dict) -> tuple[str, str]:
    titles: list[str] = []
    keywords: list[str] = []
    abstracts: list[str] = []
    papers = sp.get("papers", []) or []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = (paper.get("title") or "").strip()
        if title:
            titles.append(title)
        kw = paper.get("keywords") or []
        if isinstance(kw, list):
            keywords.extend([str(x).strip() for x in kw if str(x).strip()])
        abs_text = (paper.get("abstract") or "").strip()
        if abs_text:
            abstracts.append(abs_text)
    cluster_kw = sp.get("cluster_keywords") or []
    if isinstance(cluster_kw, list):
        keywords.extend([str(x).strip() for x in cluster_kw if str(x).strip()])
    text_tk = " ".join(titles + keywords).strip()
    text_abs = " ".join(abstracts).strip()
    return text_tk, text_abs


# ---------------------------------------------------------------------------
# Surrogate result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SurrogateBuildResult:
    coef_tk_i: np.ndarray
    coef_abs_i: np.ndarray
    reps_tk_i: np.ndarray
    reps_abs_i: np.ndarray
    author_ids: list[str]
    subprofile_count: int
    metadata: dict
    build_sec: float
    pca_feat_tk_components: np.ndarray | None = None
    pca_feat_tk_mean: np.ndarray | None = None
    pca_feat_abs_components: np.ndarray | None = None
    pca_feat_abs_mean: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Dim reduction helpers
# ---------------------------------------------------------------------------

def quantize_float(vec: np.ndarray, scale: int) -> np.ndarray:
    return np.round(vec * scale).astype(np.int64)


def select_author_representatives(
    subprofiles_tk: np.ndarray,
    subprofiles_abs: np.ndarray,
    sub_to_author: np.ndarray,
    num_authors: int,
) -> tuple[np.ndarray, np.ndarray]:
    reps_tk = np.zeros((num_authors, subprofiles_tk.shape[1]), dtype=np.float32)
    reps_abs = np.zeros((num_authors, subprofiles_abs.shape[1]), dtype=np.float32)
    for author_idx in range(num_authors):
        mask = sub_to_author == author_idx
        if not np.any(mask):
            continue
        reps_tk[author_idx] = subprofiles_tk[mask].mean(axis=0)
        reps_abs[author_idx] = subprofiles_abs[mask].mean(axis=0)
    return reps_tk, reps_abs


def fit_dim_reduction(
    embeddings: np.ndarray,
    target_dim: int,
    method: str,
) -> tuple[np.ndarray, object]:
    if embeddings.shape[1] <= target_dim:
        return embeddings, None
    if method == "pca":
        m = PCA(n_components=target_dim, random_state=42)
        return m.fit_transform(embeddings).astype(np.float32), m
    if method == "svd":
        m = TruncatedSVD(n_components=target_dim, random_state=42)
        return m.fit_transform(embeddings).astype(np.float32), m
    raise ValueError(f"Unknown dim_reduction={method!r}. Choose 'pca' or 'svd'.")


def _apply_dim_reduction(embeddings: np.ndarray, target_dim: int, method: str) -> np.ndarray:
    reduced, _ = fit_dim_reduction(embeddings, target_dim, method)
    return reduced


# ---------------------------------------------------------------------------
# Train / Save / Load surrogate artifact
# ---------------------------------------------------------------------------

def train_surrogate_artifact(
    *,
    max_authors: int | None,
    max_subprofiles: int | None,
    n_features: int,
    target_dim: int,
    alpha: float | None = None,
    coef_scale: int = 8,
    profile_scale: int = 8,
    dim_reduction: str = "pca",
    feature_type: str = "hashing",
) -> SurrogateBuildResult:
    """Train Ridge regression surrogate: binary features -> reduced MiniLM embeddings."""
    t0 = time.perf_counter()
    profiles = _load_author_profiles()

    author_ids = list(profiles.keys())
    if max_authors is not None:
        author_ids = author_ids[:max_authors]

    tk_texts: list[str] = []
    abs_texts: list[str] = []
    tk_embeds_raw: list[np.ndarray] = []
    abs_embeds_raw: list[np.ndarray] = []
    sub_to_author: list[int] = []
    used_author_ids: list[str] = []

    for author_id in author_ids:
        data = profiles.get(author_id, {})
        sub_profiles = data.get("sub_profiles", []) or []
        if max_subprofiles is not None:
            sub_profiles = sub_profiles[:max_subprofiles]

        had_subprofile = False
        for sp in sub_profiles:
            text_tk, text_abs = _build_subprofile_text(sp)
            tk_texts.append(text_tk)
            abs_texts.append(text_abs)

            emb_tk = _to_numpy(sp["embed_tk"]).reshape(-1).astype(np.float32)
            emb_abs = _to_numpy(sp["embed_abs"]).reshape(-1).astype(np.float32)
            tk_embeds_raw.append(emb_tk)
            abs_embeds_raw.append(emb_abs)
            sub_to_author.append(len(used_author_ids))
            had_subprofile = True

        if had_subprofile:
            used_author_ids.append(author_id)

    if not tk_embeds_raw:
        raise RuntimeError("No subprofile targets found for surrogate training.")

    all_tk_raw = np.stack(tk_embeds_raw, axis=0)
    all_abs_raw = np.stack(abs_embeds_raw, axis=0)

    tk_reduced = _apply_dim_reduction(all_tk_raw, target_dim, dim_reduction)
    abs_reduced = _apply_dim_reduction(all_abs_raw, target_dim, dim_reduction)

    y_tk = np.stack([_l2_normalize(v) for v in tk_reduced], axis=0).astype(np.float32)
    y_abs = np.stack([_l2_normalize(v) for v in abs_reduced], axis=0).astype(np.float32)

    pca_feat_tk_components: np.ndarray | None = None
    pca_feat_tk_mean: np.ndarray | None = None
    pca_feat_abs_components: np.ndarray | None = None
    pca_feat_abs_mean: np.ndarray | None = None

    if feature_type == "binarized_pca":
        pca_tk = PCA(n_components=target_dim, random_state=42).fit(all_tk_raw)
        pca_abs = PCA(n_components=target_dim, random_state=42).fit(all_abs_raw)
        x_tk = (pca_tk.transform(all_tk_raw) > 0).astype(np.float32)
        x_abs = (pca_abs.transform(all_abs_raw) > 0).astype(np.float32)
        pca_feat_tk_components = pca_tk.components_.astype(np.float32)
        pca_feat_tk_mean = pca_tk.mean_.astype(np.float32)
        pca_feat_abs_components = pca_abs.components_.astype(np.float32)
        pca_feat_abs_mean = pca_abs.mean_.astype(np.float32)
    elif feature_type == "hashing":
        vect_tk = HashingVectorizer(
            n_features=n_features, alternate_sign=False, binary=True,
            norm=None, ngram_range=(1, 2),
        )
        vect_abs = HashingVectorizer(
            n_features=n_features, alternate_sign=False, binary=True,
            norm=None, ngram_range=(1, 2),
        )
        x_tk = vect_tk.transform(tk_texts)
        x_abs = vect_abs.transform(abs_texts)
    else:
        raise ValueError(f"Unknown feature_type={feature_type!r}.")

    _alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    if alpha is None:
        model_tk = RidgeCV(alphas=_alphas, fit_intercept=False, scoring="r2")
        model_abs = RidgeCV(alphas=_alphas, fit_intercept=False, scoring="r2")
        model_tk.fit(x_tk, y_tk)
        model_abs.fit(x_abs, y_abs)
        best_alpha_tk = float(model_tk.alpha_)
        best_alpha_abs = float(model_abs.alpha_)
        print(
            f"[surrogate] RidgeCV selected alpha: tk={best_alpha_tk} abs={best_alpha_abs}",
            flush=True,
        )
        alpha = best_alpha_tk
    else:
        model_tk = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=42)
        model_abs = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", random_state=42)
        model_tk.fit(x_tk, y_tk)
        model_abs.fit(x_abs, y_abs)

    reps_tk, reps_abs = select_author_representatives(
        subprofiles_tk=y_tk,
        subprofiles_abs=y_abs,
        sub_to_author=np.asarray(sub_to_author, dtype=np.int64),
        num_authors=len(used_author_ids),
    )

    metadata = {
        "version": 2,
        "surrogate_type": "hashing_vectorizer_ridge_linear",
        "feature_type": feature_type,
        "n_features": int(n_features),
        "ngram_range": [1, 2],
        "binary": True,
        "alternate_sign": False,
        "norm": None,
        "target_dim": int(target_dim),
        "alpha": float(alpha),
        "coef_scale": int(coef_scale),
        "profile_scale": int(profile_scale),
        "max_authors_requested": None if max_authors is None else int(max_authors),
        "max_subprofiles_requested": None if max_subprofiles is None else int(max_subprofiles),
        "weight_tk_i": int(round(WEIGHT_TK * 10)),
        "weight_abs_i": int(round(WEIGHT_ABS * 10)),
        "empty_text_policy": "zero_vector",
        "dim_reduction": dim_reduction,
    }

    return SurrogateBuildResult(
        coef_tk_i=np.round(model_tk.coef_.astype(np.float64) * coef_scale).astype(np.int64),
        coef_abs_i=np.round(model_abs.coef_.astype(np.float64) * coef_scale).astype(np.int64),
        reps_tk_i=quantize_float(reps_tk, scale=profile_scale),
        reps_abs_i=quantize_float(reps_abs, scale=profile_scale),
        author_ids=used_author_ids,
        subprofile_count=len(tk_embeds_raw),
        metadata=metadata,
        build_sec=time.perf_counter() - t0,
        pca_feat_tk_components=pca_feat_tk_components,
        pca_feat_tk_mean=pca_feat_tk_mean,
        pca_feat_abs_components=pca_feat_abs_components,
        pca_feat_abs_mean=pca_feat_abs_mean,
    )


def save_surrogate_artifact(out_dir: Path, result: SurrogateBuildResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict = {
        "coef_tk_i": result.coef_tk_i,
        "coef_abs_i": result.coef_abs_i,
        "reps_tk_i": result.reps_tk_i,
        "reps_abs_i": result.reps_abs_i,
    }
    if result.pca_feat_tk_components is not None:
        arrays["pca_feat_tk_components"] = result.pca_feat_tk_components
        arrays["pca_feat_tk_mean"] = result.pca_feat_tk_mean
        arrays["pca_feat_abs_components"] = result.pca_feat_abs_components
        arrays["pca_feat_abs_mean"] = result.pca_feat_abs_mean
    np.savez_compressed(out_dir / "surrogate_arrays.npz", **arrays)
    arrays_path = out_dir / "surrogate_arrays.npz"
    arrays_sha256 = sha256_file(arrays_path)
    payload = {
        "metadata": result.metadata,
        "author_ids": result.author_ids,
        "subprofile_count": result.subprofile_count,
        "build_sec": result.build_sec,
        "arrays_sha256": arrays_sha256,
    }
    (out_dir / "surrogate_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_surrogate_artifact(artifact_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load surrogate artifact. Returns (meta, coef_tk_i, coef_abs_i, reps_tk_i, reps_abs_i, pca_components)."""
    artifact_dir = artifact_dir.expanduser().resolve()
    arrays_path = artifact_dir / "surrogate_arrays.npz"
    meta_path = artifact_dir / "surrogate_meta.json"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays artifact: {arrays_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {meta_path}")

    arrays = np.load(arrays_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_sha = (meta.get("arrays_sha256") or "").strip()
    if expected_sha:
        actual_sha = sha256_file(arrays_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Surrogate artifact checksum mismatch. expected={expected_sha} actual={actual_sha}"
            )
    pca_components = {
        k: arrays[k] for k in
        ["pca_feat_tk_components", "pca_feat_tk_mean",
         "pca_feat_abs_components", "pca_feat_abs_mean"]
        if k in arrays
    }
    return (
        meta,
        arrays["coef_tk_i"].astype(np.int64),
        arrays["coef_abs_i"].astype(np.int64),
        arrays["reps_tk_i"].astype(np.int64),
        arrays["reps_abs_i"].astype(np.int64),
        pca_components,
    )


# ---------------------------------------------------------------------------
# Query binary features
# ---------------------------------------------------------------------------

def build_query_binary_features(n_features: int) -> np.ndarray:
    """Build binary hash features for query text (hashing vectorizer)."""
    query_tk_text, query_abs_text = _get_query_text()
    vect_tk = HashingVectorizer(
        n_features=n_features, alternate_sign=False, binary=True,
        norm=None, ngram_range=(1, 2),
    )
    vect_abs = HashingVectorizer(
        n_features=n_features, alternate_sign=False, binary=True,
        norm=None, ngram_range=(1, 2),
    )
    q_tk = (vect_tk.transform([query_tk_text]).toarray()[0] > 0)
    q_abs = (vect_abs.transform([query_abs_text]).toarray()[0] > 0)
    return np.concatenate([q_tk, q_abs]).astype(np.int16)


# ---------------------------------------------------------------------------
# Deterministic binary inputset
# ---------------------------------------------------------------------------

def build_deterministic_binary_inputset(input_len: int) -> list[np.ndarray]:
    """Build deterministic binary inputset for TFHE compile-time bounds inference."""
    if input_len <= 0:
        raise ValueError(f"input_len must be > 0, got {input_len}")

    half = max(1, input_len // 2)
    quarter = max(1, input_len // 4)

    zeros = np.zeros((input_len,), dtype=np.int16)
    ones = np.ones((input_len,), dtype=np.int16)
    alt01 = (np.arange(input_len, dtype=np.int16) % 2).astype(np.int16)
    alt10 = (1 - alt01).astype(np.int16)

    first_quarter = np.zeros((input_len,), dtype=np.int16)
    first_quarter[:quarter] = 1
    last_quarter = np.zeros((input_len,), dtype=np.int16)
    last_quarter[-quarter:] = 1

    first_half = np.zeros((input_len,), dtype=np.int16)
    first_half[:half] = 1
    last_half = np.zeros((input_len,), dtype=np.int16)
    last_half[-half:] = 1

    return [
        zeros, ones, alt01, alt10,
        first_quarter, last_quarter,
        first_half, last_half,
    ]


# ---------------------------------------------------------------------------
# S1 circuit: Phase 1 (linear embed) + Phase 2 (encrypted selection sort top-K)
# ---------------------------------------------------------------------------

def build_encrypted_surrogate_topk_circuit(
    coef_tk_i: np.ndarray,
    coef_abs_i: np.ndarray,
    reps_tk_i: np.ndarray,
    reps_abs_i: np.ndarray,
    weight_tk_i: int,
    weight_abs_i: int,
    *,
    top_k: int,
    use_gpu: bool,
) -> fhe.Circuit:
    """Build TFHE circuit: binary features -> encrypted top-K indices and scores.

    Phase 1 (linear): compute all N author scores (no TLU, GPU-friendly).
    Phase 2 (selection sort): k rounds of encrypted argmax (uses TLU via bootstrapping).
    Only top-K result is returned; client never sees all N scores.
    """
    coef_tk = coef_tk_i.astype(np.int64)
    coef_abs = coef_abs_i.astype(np.int64)
    reps_tk = reps_tk_i.astype(np.int64)
    reps_abs = reps_abs_i.astype(np.int64)
    num_authors = int(reps_tk.shape[0])
    dim = int(reps_tk.shape[1])
    n_features = int(coef_tk.shape[1])
    input_len = n_features * 2

    _inputset_preview = build_deterministic_binary_inputset(input_len)
    _max_abs_score = 1
    for _q in _inputset_preview:
        _q64 = _q.astype(np.int64)
        _emb_tk = coef_tk @ _q64[:n_features]
        _emb_abs = coef_abs @ _q64[n_features:]
        _s = (reps_tk @ _emb_tk) * np.int64(weight_tk_i) + (reps_abs @ _emb_abs) * np.int64(weight_abs_i)
        _max_abs_score = max(_max_abs_score, int(np.abs(_s).max()))
    min_score_val = np.int64(-(_max_abs_score + 1))
    import math
    _tlu_bits = math.ceil(math.log2(_max_abs_score * 2 + 2)) if _max_abs_score > 0 else 1
    print(
        f"[surrogate] topk circuit: max_abs_score={_max_abs_score} min_score={min_score_val} "
        f"tlu_diff_bits≈{_tlu_bits} (n_feat={n_features} dim={dim} authors={num_authors})",
        flush=True,
    )

    def encrypted_surrogate_topk(query_features):
        tk_features = query_features[:n_features]
        abs_features = query_features[n_features:]

        # Phase 1: linear embedding (no TLU)
        embed_tk = [np.sum(tk_features * coef_tk[d]) for d in range(dim)]
        embed_abs = [np.sum(abs_features * coef_abs[d]) for d in range(dim)]

        working = []
        for author_idx in range(num_authors):
            score_tk = np.int64(0)
            score_abs = np.int64(0)
            for d in range(dim):
                score_tk += embed_tk[d] * reps_tk[author_idx][d]
                score_abs += embed_abs[d] * reps_abs[author_idx][d]
            working.append(
                (score_tk * np.int64(weight_tk_i)) + (score_abs * np.int64(weight_abs_i))
            )

        # Phase 2: selection sort top-K (uses TLU for comparison).
        # Keep this aligned with code-TFHE/uc02_e2e_tfhe: arithmetic conditional
        # updates compile quickly for the small UC02-style benchmark.
        top_idx_list = []
        top_score_list = []

        for _ in range(top_k):
            best_score = working[0]
            best_idx = np.int64(0)

            for i in range(1, num_authors):
                is_better = working[i] > best_score
                best_score = best_score + is_better * (working[i] - best_score)
                best_idx = best_idx + is_better * (np.int64(i) - best_idx)

            top_idx_list.append(best_idx)
            top_score_list.append(best_score)

            for i in range(num_authors):
                is_selected = best_idx == np.int64(i)
                working[i] = working[i] - is_selected * (working[i] - min_score_val)

        return fhe.array(top_idx_list), fhe.array(top_score_list)

    inputset = build_deterministic_binary_inputset(input_len)
    config = fhe.Configuration(
        parameter_selection_strategy="mono",
        global_p_error=0.05,
        use_gpu=use_gpu,
    )
    compiler = fhe.Compiler(encrypted_surrogate_topk, {"query_features": "encrypted"})
    return compiler.compile(inputset, configuration=config)


# ---------------------------------------------------------------------------
# S3 circuit: Phase 1 only (no encrypted ranking — client gets all N scores)
# ---------------------------------------------------------------------------

def build_encrypted_surrogate_scores_circuit(
    coef_tk_i: np.ndarray,
    coef_abs_i: np.ndarray,
    reps_tk_i: np.ndarray,
    reps_abs_i: np.ndarray,
    weight_tk_i: int,
    weight_abs_i: int,
    *,
    use_gpu: bool,
) -> fhe.Circuit:
    """Build TFHE circuit: binary features -> all N encrypted scores (no top-K selection).

    Same Phase 1 as build_encrypted_surrogate_topk_circuit.
    Returns fhe.array of all N scores. Client decrypts all and ranks in plaintext.
    This avoids Phase 2 TLUs (no bootstrapping for ranking), trading server privacy
    for lower circuit complexity.
    """
    coef_tk = coef_tk_i.astype(np.int64)
    coef_abs = coef_abs_i.astype(np.int64)
    reps_tk = reps_tk_i.astype(np.int64)
    reps_abs = reps_abs_i.astype(np.int64)
    num_authors = int(reps_tk.shape[0])
    dim = int(reps_tk.shape[1])
    n_features = int(coef_tk.shape[1])
    input_len = n_features * 2

    _inputset_preview = build_deterministic_binary_inputset(input_len)
    _max_abs_score = 1
    for _q in _inputset_preview:
        _q64 = _q.astype(np.int64)
        _emb_tk = coef_tk @ _q64[:n_features]
        _emb_abs = coef_abs @ _q64[n_features:]
        _s = (reps_tk @ _emb_tk) * np.int64(weight_tk_i) + (reps_abs @ _emb_abs) * np.int64(weight_abs_i)
        _max_abs_score = max(_max_abs_score, int(np.abs(_s).max()))
    _score_bounds = np.array([-_max_abs_score, _max_abs_score], dtype=np.int64)

    def encrypted_surrogate_scores(query_features):
        tk_features = query_features[:n_features]
        abs_features = query_features[n_features:]

        # Phase 1: linear embedding (no TLU, no bootstrapping)
        embed_tk = [np.sum(tk_features * coef_tk[d]) for d in range(dim)]
        embed_abs = [np.sum(abs_features * coef_abs[d]) for d in range(dim)]

        scores = []
        for author_idx in range(num_authors):
            score_tk = np.int64(0)
            score_abs = np.int64(0)
            for d in range(dim):
                score_tk += embed_tk[d] * reps_tk[author_idx][d]
                score_abs += embed_abs[d] * reps_abs[author_idx][d]
            score = (score_tk * np.int64(weight_tk_i)) + (score_abs * np.int64(weight_abs_i))
            scores.append(fhe.hint(score, can_store=_score_bounds))

        return tuple(scores)

    inputset = build_deterministic_binary_inputset(input_len)
    config = fhe.Configuration(
        parameter_selection_strategy="mono",
        global_p_error=0.05,
        use_gpu=use_gpu,
    )
    compiler = fhe.Compiler(encrypted_surrogate_scores, {"query_features": "encrypted"})
    return compiler.compile(inputset, configuration=config)
