"""Orchestrator: Run all benchmark scenarios sequentially.

Usage:
    python orchestrate.py [--scenarios s1,s2a,...] [--env-file .env]
    python orchestrate.py --clean [--clean-artifacts]

All scenarios:
    baseline, s1, s2a, s2b, s3, s4a, s4b, s4c, s4d, s4e
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

class _TeeWriter:
    """Writes to terminal and a log file simultaneously. Used to auto-capture full stdout."""
    def __init__(self, terminal, logfile):
        self._term = terminal
        self._log = logfile

    def write(self, data):
        self._term.write(data)
        if not self._log.closed:
            self._log.write(data)

    def flush(self):
        self._term.flush()
        if not self._log.closed:
            self._log.flush()

    @property
    def encoding(self):
        return getattr(self._term, "encoding", "utf-8")

    def isatty(self):
        return self._term.isatty() if hasattr(self._term, "isatty") else False


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
LOG_DIR = ROOT_DIR / "logs"
ORCHESTRATE_LOG_DIR = LOG_DIR / "orchestrate"
RUN_LOG_DIR = LOG_DIR / "run"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CONCRETE_ARTIFACTS_DIR = ROOT_DIR / ".artifacts"

SCENARIOS: dict[str, str] = {
    "baseline": "usecases.baseline.run",
    "s1": "usecases.s1_tfhe.run",
    "s2a": "usecases.s2a_tfhe.run",
    "s2b": "usecases.s2b_ckks.run",
    "s3": "usecases.s3_tfhe.run",
    "s4a": "usecases.s4a_tfhe.run",
    "s4b": "usecases.s4b_ckks.run",
    "s4c": "usecases.s4c_phe_paillier.run",
    "s4d": "usecases.s4d_she_bgv.run",
    "s4e": "usecases.s4e_she_bfv.run",
}

# Display names and short descriptions per scenario
SCENARIO_META: dict[str, dict] = {
    "baseline": {"scheme": "Plaintext", "dim": 384, "label": "Baseline Plaintext"},
    "s1":       {"scheme": "TFHE (Surrogate+TopK)", "dim": 16, "label": "S1 TFHE E2E Surrogate"},
    "s2a":      {"scheme": "TFHE (Encrypted TopK)", "dim": 16, "label": "S2a TFHE Encrypted Ranking"},
    "s2b":      {"scheme": "CKKS (Approx Ranking)", "dim": 64, "label": "S2b CKKS Approx Ranking"},
    "s3":       {"scheme": "TFHE (Surrogate, Client Rank)", "dim": 16, "label": "S3 TFHE Surrogate ClientRank"},
    "s4a":      {"scheme": "TFHE (per-subprofile)", "dim": 384, "label": "S4a TFHE Dot Product"},
    "s4b":      {"scheme": "CKKS (TenSEAL)", "dim": 384, "label": "S4b CKKS Dot Product"},
    "s4c":      {"scheme": "PHE Paillier", "dim": 384, "label": "S4c Paillier Dot Product"},
    "s4d":      {"scheme": "BGV (OpenFHE)", "dim": 384, "label": "S4d BGV Dot Product"},
    "s4e":      {"scheme": "BFV (TenSEAL)", "dim": 384, "label": "S4e BFV Dot Product"},
}


def _load_env_file(env_path: Path) -> None:
    """Parse and load a .env file into os.environ."""
    if not env_path.exists():
        print(f"[orchestrate] env file not found: {env_path}", flush=True)
        return
    print(f"[orchestrate] loading env from {env_path}", flush=True)
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _reset_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _setup_logger(log_dir: Path, run_timestamp: str | None = None) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"orchestrate_{timestamp}.log"

    logger = logging.getLogger("orchestrate")
    logger.setLevel(logging.DEBUG)
    _reset_logger_handlers(logger)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"[orchestrate] log file: {log_file}")
    return logger, log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all or selected benchmark scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available scenarios: {', '.join(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated list of scenarios to run. Default: all.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT_DIR / ".env",
        help="Path to .env file. Default: .env in project root.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help=(
            "Do not run any scenario. Instead read existing output/runs/latest/results "
            "or legacy output/result_*.json files and generate a consolidated RUN_SUMMARY.md."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete generated output/, logs/orchestrate/, and logs/run/ contents, then exit unless scenarios are also requested.",
    )
    parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        default=False,
        help="Delete artifacts/ and .artifacts/ contents. Combine with --clean for a full reset.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        default=False,
        help=(
            "Relaunch orchestrator in background (detached). "
            "Full output saved to logs/run/<timestamp>_full.out automatically. "
            "Prints PID and tail command, then exits."
        ),
    )
    return parser.parse_args()


def _clean_dir_contents(path: Path, *, keep_gitkeep: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if keep_gitkeep and item.name == ".gitkeep":
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    if keep_gitkeep:
        (path / ".gitkeep").touch(exist_ok=True)


def clean_generated(*, include_output_logs: bool, include_artifacts: bool) -> None:
    if include_output_logs:
        print("[orchestrate] cleaning generated logs and output...", flush=True)
        _clean_dir_contents(ORCHESTRATE_LOG_DIR)
        _clean_dir_contents(RUN_LOG_DIR)
        _clean_dir_contents(OUTPUT_DIR)
    if include_artifacts:
        print("[orchestrate] cleaning artifacts and Concrete compiler artifacts...", flush=True)
        _clean_dir_contents(ARTIFACTS_DIR)
        _clean_dir_contents(CONCRETE_ARTIFACTS_DIR, keep_gitkeep=False)
    print("[orchestrate] clean complete.", flush=True)


def _read_result_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _summary_source_dirs(output_dir: Path, runs_dir: Path) -> tuple[Path, Path]:
    latest_dir = runs_dir / "latest"
    latest_results = latest_dir / "results"
    latest_timing = latest_dir / "timing"
    if latest_results.exists():
        return latest_results, latest_timing
    return output_dir, output_dir


def _generate_run_summary(
    run_dir: Path,
    results: list[dict],
    accuracy: dict,
    run_timestamp: str,
) -> None:
    """Write RUN_SUMMARY.md to run_dir."""
    query_title = os.environ.get("QUERY_TITLE", "")
    query_kw = os.environ.get("QUERY_KEYWORDS", "")

    lines: list[str] = []
    lines.append("# RUN SUMMARY\n")
    lines.append(f"**Timestamp:** {run_timestamp}  ")
    lines.append(f"**Scenarios:** {', '.join(r['scenario'] for r in results)}  ")
    if query_title:
        lines.append(f"**Query title:** {query_title}  ")
    if query_kw:
        lines.append(f"**Query keywords:** {query_kw}  ")
    lines.append("")

    # --- Scenario results table ---
    lines.append("## Hasil Skenario\n")
    lines.append("| Skenario | Status | Total (s) | Precision@K | Recall@K | Overlap@K | Exact@K | Skema |")
    lines.append("|----------|--------|----------:|------------:|---------:|-----------|---------|-------|")
    for r in results:
        key = r["scenario"]
        meta = SCENARIO_META.get(key, {})
        if key == "baseline":
            precision = recall = overlap = exact = "(ref)"
        else:
            acc = accuracy.get(key)
            if acc:
                k = acc["k"]
                p = acc["overlap_k"] / k if k else 0.0
                rec = acc["overlap_k"] / k if k else 0.0
                precision = f"{p:.2f}"
                recall = f"{rec:.2f}"
                overlap = f"{acc['overlap_k']}/{k}"
                exact = f"{acc['exact_k']}/{k}"
            else:
                precision = recall = exact = overlap = "—"
        status = r["status"]
        elapsed = r["elapsed_sec"]
        scheme = meta.get("scheme", "—")
        lines.append(f"| {key} | {status} | {elapsed:.1f} | {precision} | {recall} | {overlap} | {exact} | {scheme} |")
    lines.append("")

    # --- Timing detail table ---
    lines.append("## Timing Detail (detik)\n")
    lines.append("| Skenario | embed | encrypt | server_run | decrypt | total |")
    lines.append("|----------|------:|--------:|-----------:|--------:|------:|")
    for r in results:
        key = r["scenario"]
        result_json = _read_result_json(run_dir / "results" / f"result_{key}.json")
        if result_json and "timing" in result_json:
            t = result_json["timing"]
            embed = t.get("embed_sec", 0)
            encrypt = t.get("encrypt_sec", t.get("keygen_sec", 0))
            server = t.get("run_sec", 0)
            decrypt = t.get("decrypt_sec", 0)
            total = t.get("total_sec", r["elapsed_sec"])
            lines.append(f"| {key} | {embed:.3f} | {encrypt:.3f} | {server:.3f} | {decrypt:.3f} | {total:.3f} |")
        else:
            lines.append(f"| {key} | — | — | — | — | {r['elapsed_sec']:.3f} |")
    lines.append("")

    # --- Performance ranking ---
    lines.append("## Ranking Performa (by total_sec)\n")
    ok_results = [r for r in results if r["status"] == "OK"]
    ok_results_sorted = sorted(ok_results, key=lambda r: r["elapsed_sec"])
    baseline_time = next((r["elapsed_sec"] for r in ok_results if r["scenario"] == "baseline"), None)
    for i, r in enumerate(ok_results_sorted):
        key = r["scenario"]
        t = r["elapsed_sec"]
        if baseline_time and key != "baseline":
            ratio = f" ({t/baseline_time:.1f}× baseline)"
        else:
            ratio = " (reference)"
        lines.append(f"{i+1}. **{key}**: {t:.1f}s{ratio}")
    lines.append("")

    # --- Accuracy ranking ---
    lines.append("## Ranking Akurasi (Exact@K vs baseline)\n")
    acc_rows = []
    for r in results:
        key = r["scenario"]
        if key == "baseline" or r["status"] != "OK":
            continue
        acc = accuracy.get(key)
        if acc:
            acc_rows.append((key, acc["exact_k"], acc["overlap_k"], acc["k"]))
    acc_rows.sort(key=lambda x: (-x[1], -x[2]))
    for key, exact, overlap, k in acc_rows:
        lines.append(f"- **{key}**: Exact@{k}={exact}/{k}, Overlap@{k}={overlap}/{k}")
    if not acc_rows:
        lines.append("*(belum ada data akurasi)*")
    lines.append("")

    # --- Insights ---
    lines.append("## Insights\n")
    _append_insights(lines, results, accuracy, baseline_time)

    content = "\n".join(lines)
    summary_path = run_dir / "RUN_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[orchestrate] wrote {summary_path}", flush=True)


def _append_insights(lines: list[str], results: list[dict], accuracy: dict, baseline_time: float | None) -> None:
    ok = {r["scenario"]: r for r in results if r["status"] == "OK"}
    err = {r["scenario"]: r for r in results if r["status"] != "OK"}

    # Speed insights
    if baseline_time:
        slowest = max((r for k, r in ok.items() if k != "baseline"), key=lambda r: r["elapsed_sec"], default=None)
        fastest_non_base = min((r for k, r in ok.items() if k != "baseline"), key=lambda r: r["elapsed_sec"], default=None)
        if slowest:
            lines.append(f"- Skenario **terlambat**: `{slowest['scenario']}` ({slowest['elapsed_sec']:.0f}s = {slowest['elapsed_sec']/baseline_time:.0f}× baseline)")
        if fastest_non_base:
            lines.append(f"- Skenario **tercepat** (non-baseline): `{fastest_non_base['scenario']}` ({fastest_non_base['elapsed_sec']:.1f}s = {fastest_non_base['elapsed_sec']/baseline_time:.1f}× baseline)")

    # Accuracy insights
    # Some scenarios still use a constrained author pool because encrypted
    # ranking/comparison circuits are not practical at full dataset size.
    _limited_pool = {"s1", "s2a", "s2b", "s3"}
    perfect = [k for k, r in ok.items() if k != "baseline" and accuracy.get(k, {}).get("exact_k") == accuracy.get(k, {}).get("k") and accuracy.get(k)]
    if perfect:
        lines.append(f"- **Akurasi sempurna (Exact@K=K)**: {', '.join(f'`{k}`' for k in perfect)}")
    low_acc = [k for k, r in ok.items() if k != "baseline" and accuracy.get(k, {}).get("exact_k", 999) == 0]
    if low_acc:
        constrained = [k for k in low_acc if k in _limited_pool]
        unconstrained = [k for k in low_acc if k not in _limited_pool]
        if constrained:
            constrained_str = ", ".join(f"`{k}`" for k in constrained)
            lines.append(
                f"- **Akurasi 0 karena pool author terbatas**: {constrained_str} "
                "— skenario ini hanya memproses bounded pool kecil, bukan semua >2000 author seperti baseline. "
                "S2a/S2b dipertahankan sebagai feasibility encrypted ranking/comparison; "
                "query-dependent private candidate retrieval berada di luar scope implementasi ini."
            )
        if unconstrained:
            lines.append(f"- **Akurasi 0 (kemungkinan skor nol/error)**: {', '.join(f'`{k}`' for k in unconstrained)} — periksa konfigurasi")

    # Error scenarios
    if err:
        for k, r in err.items():
            lines.append(f"- **[WARN] {k} GAGAL**: `{r['error'][:80]}`")

    # S4c Paillier specific note
    if "s4c" in ok and "s4b" in ok:
        ratio = ok["s4c"]["elapsed_sec"] / ok["s4b"]["elapsed_sec"]
        lines.append(f"- **Paillier (S4c)** {ratio:.0f}× lebih lambat dari CKKS (S4b) untuk dimensi yang sama (384-dim) — trade-off PHE vs SHE")

    # TFHE compilation note
    for k in ["s1", "s2a", "s3"]:
        if k in ok:
            lines.append(f"- **{k}** (TFHE): ada overhead compile circuit di waktu pertama (~menit) yang tidak termasuk di timing server_run")

    if not ok and not err:
        lines.append("*(tidak ada skenario yang selesai)*")


def _generate_all_runs_report(runs_dir: Path) -> None:
    """Scan all run folders and write all_runs_report.md aggregating every run."""
    run_folders = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name != "latest"],
        key=lambda d: d.name,
    )
    if not run_folders:
        return

    lines: list[str] = []
    lines.append("# All Runs Report\n")
    lines.append(f"- Output root: `{runs_dir}`")
    lines.append(f"- Run folders: `{len(run_folders)}`")
    lines.append("")

    # Collect all results across all runs
    all_rows: list[dict] = []
    for folder in run_folders:
        meta_path = folder / "run_meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        ts = meta.get("timestamp", folder.name)
        for r in meta.get("results", []):
            key = r["scenario"]
            elapsed = r.get("elapsed_sec", 0)
            status = r.get("status", "?")
            acc = meta.get("accuracy", {}).get(key) or {}
            overlap = acc.get("overlap_k")
            k = acc.get("k", 5)
            precision = round(overlap / k, 2) if overlap is not None and k else None
            all_rows.append({
                "run": ts,
                "scenario": key,
                "status": status,
                "elapsed_sec": elapsed,
                "overlap_k": overlap,
                "precision": precision,
                "k": k,
            })

    # --- Per-run summary table ---
    lines.append("## Ringkasan Per Run\n")
    lines.append("| Run | Skenario | Status | Total (s) | Precision@K | Overlap@K |")
    lines.append("|-----|----------|--------|----------:|------------:|-----------|")
    for row in all_rows:
        prec = f"{row['precision']:.2f}" if row["precision"] is not None else "—"
        ov = f"{row['overlap_k']}/{row['k']}" if row["overlap_k"] is not None else "—"
        lines.append(f"| {row['run']} | {row['scenario']} | {row['status']} | {row['elapsed_sec']:.1f} | {prec} | {ov} |")
    lines.append("")

    # --- Aggregate timing per scenario ---
    from collections import defaultdict
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_scenario[row["scenario"]].append(row)

    lines.append("## Rata-rata Waktu per Skenario\n")
    lines.append("| Skenario | Runs OK | Avg (s) | Min (s) | Max (s) | Avg Precision@K |")
    lines.append("|----------|--------:|--------:|--------:|--------:|----------------:|")
    for key in SCENARIOS.keys():
        rows = [r for r in by_scenario.get(key, []) if r["status"] == "OK"]
        if not rows:
            continue
        times = [r["elapsed_sec"] for r in rows]
        precs = [r["precision"] for r in rows if r["precision"] is not None]
        avg_p = f"{sum(precs)/len(precs):.2f}" if precs else "—"
        lines.append(
            f"| {key} | {len(rows)} | {sum(times)/len(times):.1f} | {min(times):.1f} | {max(times):.1f} | {avg_p} |"
        )
    lines.append("")

    report_path = runs_dir / "all_runs_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[orchestrate] wrote {report_path}", flush=True)


def _save_run_folder(
    run_dir: Path,
    results: list[dict],
    accuracy: dict,
    output_dir: Path,
    run_timestamp: str,
    source_results_dir: Path | None = None,
    source_timing_dir: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    timing_dir = run_dir / "timing"
    results_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    # Copy result JSONs for this run
    source_results_dir = source_results_dir or output_dir
    source_timing_dir = source_timing_dir or output_dir
    for r in results:
        key = r["scenario"]
        src = source_results_dir / f"result_{key}.json"
        dest = results_dir / f"result_{key}.json"
        if src.exists() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

        timing_src = source_timing_dir / f"timing_{key}.csv"
        timing_dest = timing_dir / f"timing_{key}.csv"
        if timing_src.exists() and timing_src.resolve() != timing_dest.resolve():
            shutil.copy2(timing_src, timing_dest)

    # Write RUN_SUMMARY.md
    _generate_run_summary(run_dir, results, accuracy, run_timestamp)

    # Write run_meta.json
    meta = {
        "timestamp": run_timestamp,
        "scenarios": [r["scenario"] for r in results],
        "query_title": os.environ.get("QUERY_TITLE", ""),
        "query_keywords": os.environ.get("QUERY_KEYWORDS", ""),
        "results": results,
        "accuracy": accuracy,
        "layout": {
            "results": "results/result_<scenario>.json",
            "timing": "timing/timing_<scenario>.csv",
            "logs_orchestrate": str(ORCHESTRATE_LOG_DIR),
            "logs_run": str(RUN_LOG_DIR),
            "summary": "RUN_SUMMARY.md",
        },
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Update 'latest' symlink in runs/
    latest_link = run_dir.parent / "latest"
    if latest_link.is_symlink() or latest_link.is_file():
        latest_link.unlink()
    elif latest_link.exists():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(run_dir.name)
    print(f"[orchestrate] run folder: {run_dir}", flush=True)
    print(f"[orchestrate] updated latest -> {run_dir.name}", flush=True)

    # Rebuild aggregate report across all runs
    try:
        _generate_all_runs_report(run_dir.parent)
    except Exception as exc:
        print(f"[orchestrate] WARNING: could not generate all_runs_report: {exc}", flush=True)


def main() -> None:
    args = parse_args()

    if args.clean or args.clean_artifacts:
        clean_generated(include_output_logs=args.clean, include_artifacts=args.clean_artifacts)
        if not args.scenarios and not args.summary_only:
            return

    if args.env_file and args.env_file.exists():
        _load_env_file(args.env_file)
    elif args.env_file and not args.env_file.exists():
        print(f"[orchestrate] WARNING: --env-file {args.env_file} not found, continuing without it.", flush=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Background mode: relaunch detached, print PID + tail command, then exit ---
    if args.background:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        full_log = RUN_LOG_DIR / f"run_{run_timestamp}_full.out"
        cmd = [sys.executable, str(Path(__file__).resolve())]
        if args.scenarios:
            cmd += ["--scenarios", args.scenarios]
        if args.env_file:
            cmd += ["--env-file", str(args.env_file)]
        if args.summary_only:
            cmd += ["--summary-only"]
        with open(full_log, "w") as _bg_f:
            proc = subprocess.Popen(
                cmd, stdout=_bg_f, stderr=_bg_f,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        pid_file = LOG_DIR / "nohup.pid"
        pid_file.write_text(str(proc.pid))
        print(f"[orchestrate] background PID={proc.pid}")
        print(f"[orchestrate] pid file: {pid_file}")
        print(f"[orchestrate] log: {full_log}")
        print(f"  tail -f {full_log}")
        print(f"  kill $(cat {pid_file})   # untuk stop")
        return

    output_dir = OUTPUT_DIR
    runs_dir = RUNS_DIR
    run_dir = runs_dir / run_timestamp

    # --- TeeWriter: auto-save full stdout+stderr to logs/run/<timestamp>_full.out ---
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _full_log_path = RUN_LOG_DIR / f"run_{run_timestamp}_full.out"
    _full_log_fh = open(_full_log_path, "w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(sys.stdout, _full_log_fh)
    sys.stderr = _TeeWriter(sys.stderr, _full_log_fh)

    def _close_tee():
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        if not _full_log_fh.closed:
            _full_log_fh.flush()
            _full_log_fh.close()

    logger, _ = _setup_logger(ORCHESTRATE_LOG_DIR, run_timestamp)
    logger.info(f"[orchestrate] full output log: {_full_log_path}")

    if args.scenarios:
        selected_keys = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
        unknown = [k for k in selected_keys if k not in SCENARIOS]
        if unknown:
            logger.error(f"Unknown scenarios: {unknown}. Available: {list(SCENARIOS.keys())}")
            _close_tee()
            sys.exit(1)
    else:
        selected_keys = list(SCENARIOS.keys())

    # --- Summary-only mode: read existing results, skip running ---
    if args.summary_only:
        logger.info(f"[orchestrate] --summary-only: reading existing result files (no scenarios run)")
        source_results_dir, source_timing_dir = _summary_source_dirs(output_dir, runs_dir)
        logger.info(f"[orchestrate] summary source: {source_results_dir}")
        results: list[dict] = []
        for key in selected_keys:
            result_path = source_results_dir / f"result_{key}.json"
            data = _read_result_json(result_path)
            if data:
                total_sec = data.get("timing", {}).get("total_sec", 0)
                results.append({"scenario": key, "status": "OK", "elapsed_sec": total_sec, "error": ""})
                logger.info(f"[orchestrate] loaded existing result: {key} ({total_sec:.1f}s)")
            else:
                results.append({"scenario": key, "status": "MISSING", "elapsed_sec": 0, "error": "No result file"})
                logger.info(f"[orchestrate] no result file for: {key}")
        grand_elapsed = sum(r["elapsed_sec"] for r in results)
        baseline_result_path = source_results_dir / "result_baseline.json"
        accuracy: dict[str, dict | None] = {}
        if baseline_result_path.exists():
            try:
                from core.result_writer import compute_top_k_accuracy
                for r in results:
                    k = r["scenario"]
                    if k == "baseline":
                        continue
                    accuracy[k] = compute_top_k_accuracy(baseline_result_path, source_results_dir / f"result_{k}.json")
            except Exception as exc:
                logger.warning(f"[orchestrate] Could not compute accuracy: {exc}")
        try:
            _save_run_folder(
                run_dir,
                results,
                accuracy,
                output_dir,
                run_timestamp,
                source_results_dir=source_results_dir,
                source_timing_dir=source_timing_dir,
            )
        except Exception as exc:
            logger.warning(f"[orchestrate] Could not save run folder: {exc}")
        logger.info(f"[orchestrate] summary-only run complete. See {run_dir}/RUN_SUMMARY.md")
        _close_tee()
        return

    logger.info(f"[orchestrate] running scenarios: {selected_keys}")
    os.environ["CODE_FINAL_RUN_CONTEXT"] = "orchestrate"
    os.environ["CODE_FINAL_RUN_DIR"] = str(run_dir)

    results: list[dict] = []
    grand_start = time.perf_counter()

    for scenario_key in selected_keys:
        module_path = SCENARIOS[scenario_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"[orchestrate] START scenario={scenario_key} module={module_path}")
        t_start = time.perf_counter()
        status = "OK"
        error_msg = ""
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path!r} has no main() function.")
            module.main()
            elapsed = time.perf_counter() - t_start
            logger.info(f"[orchestrate] DONE scenario={scenario_key} elapsed={elapsed:.3f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            status = "ERROR"
            error_msg = str(exc)
            logger.error(
                f"[orchestrate] ERROR scenario={scenario_key} elapsed={elapsed:.3f}s: {exc}",
                exc_info=True,
            )
        results.append({
            "scenario": scenario_key,
            "status": status,
            "elapsed_sec": elapsed,
            "error": error_msg,
        })

    grand_elapsed = time.perf_counter() - grand_start

    # --- Compute accuracy vs baseline ---
    run_results_dir = run_dir / "results"
    run_timing_dir = run_dir / "timing"
    baseline_result_path = run_results_dir / "result_baseline.json"

    accuracy: dict[str, dict | None] = {}
    if baseline_result_path.exists():
        try:
            from core.result_writer import compute_top_k_accuracy
            for r in results:
                key = r["scenario"]
                if key == "baseline":
                    continue
                scenario_result_path = run_results_dir / f"result_{key}.json"
                accuracy[key] = compute_top_k_accuracy(baseline_result_path, scenario_result_path)
        except Exception as exc:
            logger.warning(f"[orchestrate] Could not compute accuracy: {exc}")
    else:
        logger.info("[orchestrate] baseline result not found — skipping accuracy comparison")

    # --- Save run folder + RUN_SUMMARY.md ---
    try:
        _save_run_folder(
            run_dir,
            results,
            accuracy,
            output_dir,
            run_timestamp,
            source_results_dir=run_results_dir,
            source_timing_dir=run_timing_dir,
        )
    except Exception as exc:
        logger.warning(f"[orchestrate] Could not save run folder: {exc}")

    # --- Summary table ---
    logger.info(f"\n{'='*80}")
    logger.info("[orchestrate] SUMMARY  (accuracy compared vs baseline)")
    logger.info(f"{'='*80}")
    header = f"{'Scenario':<12} {'Status':<8} {'Total (s)':>10}  {'Overlap@K':>10}  {'Exact@K':>8}  Notes"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        key = r["scenario"]
        note = r["error"][:40] if r["error"] else ""
        if key == "baseline":
            acc_str_overlap = "  (ref)"
            acc_str_exact = "  (ref)"
        else:
            acc = accuracy.get(key)
            if acc:
                acc_str_overlap = f"{acc['overlap_k']}/{acc['k']}"
                acc_str_exact = f"{acc['exact_k']}/{acc['k']}"
            else:
                acc_str_overlap = "    —"
                acc_str_exact = "    —"
        logger.info(
            f"{key:<12} {r['status']:<8} {r['elapsed_sec']:>10.3f}  "
            f"{acc_str_overlap:>10}  {acc_str_exact:>8}  {note}"
        )
    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<12} {'':8} {grand_elapsed:>10.3f}")
    logger.info(f"{'='*80}")
    logger.info("  Overlap@K : berapa author dari top-K skenario ada di top-K baseline (urutan diabaikan)")
    logger.info("  Exact@K   : berapa author yang posisi ranknya sama persis dengan baseline")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_err = len(results) - n_ok
    logger.info(f"[orchestrate] finished: {n_ok}/{len(results)} OK, {n_err} errors")

    _close_tee()

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
