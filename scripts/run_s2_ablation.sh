#!/usr/bin/env bash
# S2 Ablation: 2 algoritma (s2a=TFHE, s2b=CKKS) × 3 reduce_dim × 21 query = 126 run
# S2a pool: 8 medoid | S2b pool: 8 medoid (disamakan untuk apple-to-apple)
# Output: output/ablation/s2_params/<s2a|s2b>_rd<dim>/<query_label>/
# Estimasi: s2a ~10s/run × 63 = ~10 mnt | s2b ~120s/run × 63 = ~126 mnt | Total ~2.5 jam

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/queries.sh

PYTHON="./venv/bin/python"
LOG="logs/run/s2_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run
echo "[s2_ablation] Start: $(date)" | tee -a "$LOG"

# ---- S2a runner: 21 query combos untuk 1 reduce_dim ----
run_s2a_queries() {
    local config="$1"
    local rd="$2"

    _run() {
        local label="$1"; local title="$2"; local kw="$3"; local abs="$4"
        echo "[s2_ablation] >>> ${config}/${label}  ($(date +%H:%M:%S))" | tee -a "$LOG"
        (
            export BASELINE_MAX_AUTHORS=8
            export BASELINE_POOL_MODE=cluster_medoid
            export S2A_REDUCE_DIM="$rd"
            export QUERY_TITLE="$title"
            export QUERY_KEYWORDS="$kw"
            export QUERY_ABSTRACT="$abs"
            "$PYTHON" orchestrate.py \
                --scenarios baseline,s2a \
                --ablation-group s2_params \
                --ablation-config "$config" \
                --query-label "$label"
        ) 2>&1 | tee -a "$LOG"
        echo "[s2_ablation] <<< ${config}/${label} done" | tee -a "$LOG"
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

# ---- S2b runner: 21 query combos untuk 1 reduce_dim ----
run_s2b_queries() {
    local config="$1"
    local rd="$2"

    _run() {
        local label="$1"; local title="$2"; local kw="$3"; local abs="$4"
        echo "[s2_ablation] >>> ${config}/${label}  ($(date +%H:%M:%S))" | tee -a "$LOG"
        (
            export BASELINE_MAX_AUTHORS=8
            export BASELINE_POOL_MODE=cluster_medoid
            export S2B_MAX_AUTHORS=8
            export S2B_REDUCE_DIM="$rd"
            export QUERY_TITLE="$title"
            export QUERY_KEYWORDS="$kw"
            export QUERY_ABSTRACT="$abs"
            "$PYTHON" orchestrate.py \
                --scenarios baseline,s2b \
                --ablation-group s2_params \
                --ablation-config "$config" \
                --query-label "$label"
        ) 2>&1 | tee -a "$LOG"
        echo "[s2_ablation] <<< ${config}/${label} done" | tee -a "$LOG"
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

# ---- 3 reduce_dim × 2 algoritma = 6 config ----
echo "[s2_ablation] === S2a (TFHE) ===" | tee -a "$LOG"
run_s2a_queries "s2a_rd8"  8
run_s2a_queries "s2a_rd16" 16   # default
run_s2a_queries "s2a_rd32" 32

echo "[s2_ablation] === S2b (CKKS) ===" | tee -a "$LOG"
run_s2b_queries "s2b_rd8"  8
run_s2b_queries "s2b_rd16" 16   # default
run_s2b_queries "s2b_rd32" 32

echo "[s2_ablation] All 126 combos done: $(date)" | tee -a "$LOG"
