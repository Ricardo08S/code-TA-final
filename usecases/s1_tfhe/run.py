"""S1: TFHE E2E — Client encrypts binary features, Server does surrogate embed + similarity + encrypted top-K ranking.

Adapted from code-TFHE/uc02_e2e_tfhe/client.py + server.py.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import concrete.fhe as fhe
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import ARTIFACTS_DIR, scenario_output_paths
from core.surrogate import (
    build_deterministic_binary_inputset,
    build_encrypted_surrogate_topk_circuit,
    build_query_binary_features,
    load_surrogate_artifact,
    save_surrogate_artifact,
    sha256_file,
    train_surrogate_artifact,
)

LOG_PATH, RESULT_PATH = scenario_output_paths("s1")
PREFIX = "[S1]"
CIRCUIT_BUILDER_VERSION = 5
# Circuit is compiled with weight=1 for both tk and abs to keep score range small.
_CIRCUIT_WEIGHT_TK = 1
_CIRCUIT_WEIGHT_ABS = 1


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value!r}") from exc


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value!r}") from exc


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
            return True, "cuda"
        return False, "cpu (cuda unavailable)"
    except Exception:
        return False, "cpu (torch unavailable)"


def _append_csv(path: Path, columns: list[str], values: list[float]) -> None:
    from core.result_writer import append_csv
    append_csv(path, columns, values)


def _build_runtime_manifest(
    *,
    runtime_dir: Path,
    artifact_dir: Path,
    metadata: dict,
    author_ids: list[str],
    top_k: int,
    compile_sec: float,
    requested_device: str,
    resolved_device: str,
) -> dict:
    server_zip = runtime_dir / "server.zip"
    client_zip = runtime_dir / "client.zip"
    surrogate_arrays = artifact_dir / "surrogate_arrays.npz"
    surrogate_meta = artifact_dir / "surrogate_meta.json"
    return {
        "version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "circuit_type": "topk_encrypted",
        "circuit_builder_version": CIRCUIT_BUILDER_VERSION,
        "circuit_weight_tk": _CIRCUIT_WEIGHT_TK,
        "circuit_weight_abs": _CIRCUIT_WEIGHT_ABS,
        "top_k": int(top_k),
        "compile_sec": float(compile_sec),
        "compile_inputset_policy": "deterministic_binary_templates_v1",
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "author_ids": author_ids,
        "metadata": metadata,
        "runtime_files": {
            "server_zip": {"path": "server.zip", "sha256": sha256_file(server_zip)},
            "client_zip": {"path": "client.zip", "sha256": sha256_file(client_zip)},
        },
        "surrogate_artifact": {
            "artifact_dir": str(artifact_dir.resolve()),
            "arrays_path": str(surrogate_arrays.resolve()),
            "arrays_sha256": sha256_file(surrogate_arrays),
            "meta_path": str(surrogate_meta.resolve()),
            "meta_sha256": sha256_file(surrogate_meta),
        },
    }


def _ensure_client_keys(client: fhe.Client, key_dir: Path) -> None:
    if key_dir.exists() and any(key_dir.iterdir()):
        try:
            client.keys.load()
            return
        except Exception:
            pass
    client.keys.generate()


def _expected_surrogate_metadata(
    *,
    max_authors: int | None,
    max_subprofiles: int | None,
    n_features: int,
    target_dim: int,
    alpha: float | None,
    coef_scale: int,
    profile_scale: int,
    dim_reduction: str,
) -> dict:
    expected = {
        "n_features": int(n_features),
        "target_dim": int(target_dim),
        "coef_scale": int(coef_scale),
        "profile_scale": int(profile_scale),
        "max_authors_requested": None if max_authors is None else int(max_authors),
        "max_subprofiles_requested": None if max_subprofiles is None else int(max_subprofiles),
        "dim_reduction": dim_reduction,
        "feature_type": "hashing",
    }
    if alpha is not None:
        expected["alpha"] = float(alpha)
    return expected


def _surrogate_stale_reason(meta_path: Path, expected: dict) -> str | None:
    if not meta_path.exists():
        return "missing surrogate artifact"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = meta["metadata"]
    except Exception as exc:
        return f"invalid surrogate metadata: {exc}"

    for key, value in expected.items():
        if metadata.get(key) != value:
            return f"surrogate config mismatch for {key}: have={metadata.get(key)!r} want={value!r}"
    return None


def _runtime_stale_reason(
    *,
    manifest_path: Path,
    runtime_dir: Path,
    artifact_dir: Path,
    metadata: dict,
    author_ids: list[str],
    top_k: int,
    resolved_device: str,
) -> str | None:
    if not manifest_path.exists():
        return "missing runtime manifest"
    if not (runtime_dir / "server.zip").exists() or not (runtime_dir / "client.zip").exists():
        return "missing runtime package"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid runtime manifest: {exc}"

    if int(manifest.get("top_k", -1)) != int(top_k):
        return f"top_k mismatch: have={manifest.get('top_k')!r} want={top_k!r}"
    if int(manifest.get("circuit_builder_version", -1)) != CIRCUIT_BUILDER_VERSION:
        return "circuit builder version changed"
    if int(manifest.get("circuit_weight_tk", -1)) != _CIRCUIT_WEIGHT_TK:
        return f"circuit_weight_tk mismatch: have={manifest.get('circuit_weight_tk')!r} want={_CIRCUIT_WEIGHT_TK!r}"
    if int(manifest.get("circuit_weight_abs", -1)) != _CIRCUIT_WEIGHT_ABS:
        return f"circuit_weight_abs mismatch: have={manifest.get('circuit_weight_abs')!r} want={_CIRCUIT_WEIGHT_ABS!r}"
    if manifest.get("resolved_device") != resolved_device:
        return f"device mismatch: have={manifest.get('resolved_device')!r} want={resolved_device!r}"
    if manifest.get("metadata") != metadata:
        return "surrogate metadata changed"
    if list(manifest.get("author_ids") or []) != author_ids:
        return "author list changed"

    surrogate_arrays = artifact_dir / "surrogate_arrays.npz"
    surrogate_meta = artifact_dir / "surrogate_meta.json"
    surrogate_manifest = manifest.get("surrogate_artifact") or {}
    if surrogate_manifest.get("arrays_sha256") != sha256_file(surrogate_arrays):
        return "surrogate arrays checksum changed"
    if surrogate_manifest.get("meta_sha256") != sha256_file(surrogate_meta):
        return "surrogate metadata checksum changed"
    return None


def main() -> None:
    max_authors = _get_int_env("S1_MAX_AUTHORS", 16)
    max_subprofiles = _get_int_env("S1_MAX_SUBPROFILES")
    top_k = _get_int_env("S1_TOP_K", 1) or 1
    n_features = _get_int_env("S1_N_FEATURES", 64) or 64
    target_dim = _get_int_env("S1_TARGET_DIM", 16) or 16
    coef_scale = _get_int_env("S1_COEF_SCALE", 8) or 8
    profile_scale = _get_int_env("S1_PROFILE_SCALE", 1) or 1
    alpha = _get_float_env("S1_ALPHA", 0.0) or None  # None → RidgeCV, float → fixed Ridge
    dim_reduction = _get_str_env("S1_DIM_REDUCTION", "pca")
    device = _get_str_env("S1_SERVER_DEVICE", "cuda")

    artifact_dir = ARTIFACTS_DIR / "s1_surrogate"
    runtime_dir = ARTIFACTS_DIR / "s1_runtime"

    alpha_str = f"{alpha}" if alpha is not None else "auto(RidgeCV)"
    print(
        f"{PREFIX} max_authors={max_authors} top_k={top_k} n_features={n_features} "
        f"target_dim={target_dim} alpha={alpha_str} device={device}",
        flush=True,
    )

    t_total_start = time.perf_counter()

    # --- Offline: always retrain surrogate (deterministic → same data = same weights = same circuit checksum) ---
    print(f"{PREFIX} training surrogate...", flush=True)
    t_train_start = time.perf_counter()
    try:
        result = train_surrogate_artifact(
            max_authors=max_authors,
            max_subprofiles=max_subprofiles,
            n_features=n_features,
            target_dim=target_dim,
            alpha=alpha,
            coef_scale=coef_scale,
            profile_scale=profile_scale,
            dim_reduction=dim_reduction,
        )
    except Exception as exc:
        print(f"{PREFIX} ERROR training surrogate: {exc}", flush=True)
        raise
    save_surrogate_artifact(artifact_dir, result)
    t_train = time.perf_counter() - t_train_start
    print(f"{PREFIX} train done: {t_train:.3f}s authors={len(result.author_ids)}", flush=True)

    # --- Load surrogate ---
    meta, coef_tk_i, coef_abs_i, reps_tk_i, reps_abs_i, _ = load_surrogate_artifact(artifact_dir)
    metadata = meta["metadata"]
    author_ids = list(meta["author_ids"])
    weight_tk_i = int(metadata["weight_tk_i"])
    weight_abs_i = int(metadata["weight_abs_i"])
    n_features_loaded = int(metadata["n_features"])

    print(f"{PREFIX} surrogate loaded: authors={len(author_ids)} dim={target_dim}", flush=True)

    # --- Offline: compile circuit if missing ---
    manifest_path = runtime_dir / "runtime_manifest.json"
    t_compile = 0.0
    use_gpu, resolved_device = _resolve_use_gpu(device)

    stale_runtime = _runtime_stale_reason(
        manifest_path=manifest_path,
        runtime_dir=runtime_dir,
        artifact_dir=artifact_dir,
        metadata=metadata,
        author_ids=author_ids,
        top_k=top_k,
        resolved_device=resolved_device,
    )

    if stale_runtime:
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        print(f"{PREFIX} circuit stale ({stale_runtime}), compiling (use_gpu={use_gpu})...", flush=True)
        t_compile_start = time.perf_counter()
        try:
            circuit = build_encrypted_surrogate_topk_circuit(
                coef_tk_i=coef_tk_i,
                coef_abs_i=coef_abs_i,
                reps_tk_i=reps_tk_i,
                reps_abs_i=reps_abs_i,
                weight_tk_i=_CIRCUIT_WEIGHT_TK,
                weight_abs_i=_CIRCUIT_WEIGHT_ABS,
                top_k=top_k,
                use_gpu=use_gpu,
            )
        except Exception as exc:
            print(f"{PREFIX} ERROR compiling circuit: {exc}", flush=True)
            raise
        runtime_dir.mkdir(parents=True, exist_ok=True)
        circuit.server.save(runtime_dir / "server.zip")
        circuit.client.save(runtime_dir / "client.zip")
        t_compile = time.perf_counter() - t_compile_start
        print(f"{PREFIX} compile done: {t_compile:.3f}s", flush=True)

        manifest = _build_runtime_manifest(
            runtime_dir=runtime_dir,
            artifact_dir=artifact_dir,
            metadata=metadata,
            author_ids=author_ids,
            top_k=top_k,
            compile_sec=t_compile,
            requested_device=device,
            resolved_device=resolved_device,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print(f"{PREFIX} circuit found at {runtime_dir}", flush=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        t_compile = float(manifest.get("compile_sec", 0.0))

    # --- Online: embed query (binary features) ---
    t_embed_start = time.perf_counter()
    query_features = build_query_binary_features(n_features=n_features_loaded)
    t_embed = time.perf_counter() - t_embed_start
    print(f"{PREFIX} embed (binary features) done: {t_embed:.3f}s", flush=True)

    # --- Online: load client/server ---
    key_dir = runtime_dir / "client_keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    client = fhe.Client.load(runtime_dir / "client.zip", key_dir)
    server = fhe.Server.load(runtime_dir / "server.zip")
    _ensure_client_keys(client, key_dir)

    # --- Online: encrypt ---
    t_enc_start = time.perf_counter()
    enc_query = client.encrypt(query_features)
    t_enc = time.perf_counter() - t_enc_start
    print(f"{PREFIX} encrypt done: {t_enc:.3f}s", flush=True)

    # --- Online: server run ---
    t_run_start = time.perf_counter()
    enc_top_idx, enc_top_scores = server.run(enc_query, evaluation_keys=client.evaluation_keys)
    t_run = time.perf_counter() - t_run_start
    print(f"{PREFIX} server run done: {t_run:.3f}s", flush=True)

    # --- Online: decrypt top-K only ---
    t_dec_start = time.perf_counter()
    dec_top_idx, dec_top_scores = client.decrypt((enc_top_idx, enc_top_scores))
    t_dec = time.perf_counter() - t_dec_start
    print(f"{PREFIX} decrypt done: {t_dec:.3f}s", flush=True)

    rows = []
    for rank, (a_idx, a_score) in enumerate(zip(dec_top_idx.tolist(), dec_top_scores.tolist()), start=1):
        idx = int(a_idx)
        if 0 <= idx < len(author_ids):
            rows.append({
                "rank": rank,
                "author_idx": idx,
                "author_id": author_ids[idx],
                "score": float(a_score),
            })

    t_total = time.perf_counter() - t_total_start

    result = {
        "scenario": "s1",
        "top_k": rows,
        "timing": {
            "embed_sec": t_embed,
            "train_sec": t_train,
            "compile_sec": t_compile,
            "encrypt_sec": t_enc,
            "run_sec": t_run,
            "decrypt_sec": t_dec,
            "total_sec": t_total,
        },
        "config": {
            "scheme": "TFHE",
            "mode": "e2e_surrogate_topk",
            "dim": target_dim,
            "n_features": n_features_loaded,
            "max_authors": max_authors,
            "device": resolved_device,
        },
    }
    print(f"{PREFIX} top_k={rows[:3]}...", flush=True)
    from core.result_writer import write_json
    write_json(RESULT_PATH, result)
    print(f"{PREFIX} wrote {RESULT_PATH}", flush=True)

    _append_csv(
        LOG_PATH,
        ["embed_sec", "train_sec", "compile_sec", "encrypt_sec", "run_sec", "decrypt_sec", "total_sec"],
        [t_embed, t_train, t_compile, t_enc, t_run, t_dec, t_total],
    )


if __name__ == "__main__":
    main()
