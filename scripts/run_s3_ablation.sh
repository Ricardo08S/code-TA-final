#!/usr/bin/env bash
# S3 Ablation: 4 configs (diagonal nf×td) × 21 query combos = 84 run
# Tiap run: baseline (all 2620 MiniLM) + S3 (surrogate TFHE, client rank)
# Output: output/ablation/s3_params/<config>/<query_label>/
# PRASYARAT: jalankan compile_s3_configs.sh terlebih dahulu!
# Estimasi: ~35s/run × 84 = ~49 menit (setelah compile selesai)

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/queries.sh

PYTHON="./venv/bin/python"
LOG="logs/run/s3_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run
echo "[s3_ablation] Start: $(date)" | tee -a "$LOG"

run_all_queries() {
    local config="$1"
    local nf="$2"
    local td="$3"

    _run() {
        local label="$1"; local title="$2"; local kw="$3"; local abs="$4"
        echo "[s3_ablation] >>> ${config}/${label}  ($(date +%H:%M:%S))" | tee -a "$LOG"
        (
            export S3_N_FEATURES="$nf"
            export S3_TARGET_DIM="$td"
            export QUERY_TITLE="$title"
            export QUERY_KEYWORDS="$kw"
            export QUERY_ABSTRACT="$abs"
            "$PYTHON" orchestrate.py \
                --scenarios baseline,s3 \
                --ablation-group s3_params \
                --ablation-config "$config" \
                --query-label "$label"
        ) 2>&1 | tee -a "$LOG"
        echo "[s3_ablation] <<< ${config}/${label} done" | tee -a "$LOG"
    }

    _run "t1_nmt_full"       "$T1_TITLE" "$T1_KW"  "$T1_ABS"
    _run "t1_nmt_title"      "$T1_TITLE" ""         ""
    _run "t1_nmt_kw"         ""          "$T1_KW"   ""
    _run "t1_nmt_abs"        ""          ""          "$T1_ABS"
    _run "t1_nmt_title_kw"   "$T1_TITLE" "$T1_KW"   ""
    _run "t1_nmt_title_abs"  "$T1_TITLE" ""          "$T1_ABS"
    _run "t1_nmt_kw_abs"     ""          "$T1_KW"   "$T1_ABS"

    _run "t2_pyrolysis_full"       "$T2_TITLE" "$T2_KW"  "$T2_ABS"
    _run "t2_pyrolysis_title"      "$T2_TITLE" ""         ""
    _run "t2_pyrolysis_kw"         ""          "$T2_KW"   ""
    _run "t2_pyrolysis_abs"        ""          ""          "$T2_ABS"
    _run "t2_pyrolysis_title_kw"   "$T2_TITLE" "$T2_KW"   ""
    _run "t2_pyrolysis_title_abs"  "$T2_TITLE" ""          "$T2_ABS"
    _run "t2_pyrolysis_kw_abs"     ""          "$T2_KW"   "$T2_ABS"

    _run "t3_ontology_full"       "$T3_TITLE" "$T3_KW"  "$T3_ABS"
    _run "t3_ontology_title"      "$T3_TITLE" ""         ""
    _run "t3_ontology_kw"         ""          "$T3_KW"   ""
    _run "t3_ontology_abs"        ""          ""          "$T3_ABS"
    _run "t3_ontology_title_kw"   "$T3_TITLE" "$T3_KW"   ""
    _run "t3_ontology_title_abs"  "$T3_TITLE" ""          "$T3_ABS"
    _run "t3_ontology_kw_abs"     ""          "$T3_KW"   "$T3_ABS"
}

# ---- 4 ablasi config (diagonal) ----
run_all_queries "nf8_td8"   8  8
run_all_queries "nf16_td16" 16 16
run_all_queries "nf32_td16" 32 16
run_all_queries "nf64_td32" 64 32   # default (cached)

echo "[s3_ablation] All 84 combos done: $(date)" | tee -a "$LOG"
