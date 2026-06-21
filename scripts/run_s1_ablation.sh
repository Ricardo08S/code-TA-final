#!/usr/bin/env bash
# S1 Ablation: 6 configs (nf×td) × 21 query combos = 126 run
# Tiap run: baseline (8 medoid MiniLM) + S1 (surrogate TFHE)
# Output: output/ablation/s1_params/<config>/<query_label>/
# Estimasi: ~15s/run × 126 = ~32 menit

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/queries.sh

PYTHON="./venv/bin/python"
LOG="logs/run/s1_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run
echo "[s1_ablation] Start: $(date)" | tee -a "$LOG"

# ---- Runner: 21 query combos untuk 1 config ----
run_all_queries() {
    local config="$1"
    local nf="$2"
    local td="$3"

    _run() {
        local label="$1"; local title="$2"; local kw="$3"; local abs="$4"
        echo "[s1_ablation] >>> ${config}/${label}  ($(date +%H:%M:%S))" | tee -a "$LOG"
        (
            export BASELINE_MAX_AUTHORS=8
            export BASELINE_POOL_MODE=cluster_medoid
            export S1_N_FEATURES="$nf"
            export S1_TARGET_DIM="$td"
            export QUERY_TITLE="$title"
            export QUERY_KEYWORDS="$kw"
            export QUERY_ABSTRACT="$abs"
            "$PYTHON" orchestrate.py \
                --scenarios baseline,s1 \
                --ablation-group s1_params \
                --ablation-config "$config" \
                --query-label "$label"
        ) 2>&1 | tee -a "$LOG"
        echo "[s1_ablation] <<< ${config}/${label} done" | tee -a "$LOG"
    }

    # 3 topik × 7 kombinasi = 21 query
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

# ---- 6 ablasi config ----
run_all_queries "nf4_td4"   4  4
run_all_queries "nf8_td4"   8  4
run_all_queries "nf8_td8"   8  8
run_all_queries "nf16_td4"  16 4
run_all_queries "nf16_td8"  16 8   # default (cached)
run_all_queries "nf16_td16" 16 16

echo "[s1_ablation] All 126 combos done: $(date)" | tee -a "$LOG"
