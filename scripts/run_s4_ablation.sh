#!/usr/bin/env bash
# S4 Ablation: baseline + 5 algoritma (s4a,s4b,s4c,s4d,s4e) × 3 topik × 7 input = 21 run
# Baseline (all 2620 authors, full MiniLM) dijalankan bersama untuk evaluation metrics
# Output: output/ablation/s4_algorithms/<query_label>/
# Estimasi: ~85 menit/run × 21 = ~30 jam (s4c/Paillier ~70 menit/query, tanpa batching)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
LOG="logs/run/s4_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run

echo "[s4_ablation] Start: $(date)" | tee -a "$LOG"

# ---- Query content ----
T1_TITLE="Enhancing Domain-Specific Neural Machine Translation with Ensemble Attention Mechanisms"
T1_KW="arabic; attention mechanism;bert;indonesian;machine translation;arabic"
T1_ABS="Neural machine translation (NMT) is a method for developing a translation model using neural networks. Various modifications to NMT have been created to enhance its translation performance. The incorporation of attention techniques in NMT has become the state-of-the-art approach for various language pairs. However, this method still faces challenges when translating terms from specific domains. This research presents an approach to improve a domain-specific NMT system using an ensemble attention technique. The main idea behind our domain adaptation strategy is to combine general out-domain pre-trained models with a specific in-domain pre-trained model as an additional attention mechanism. The experimental result shows significant improvements using our proposed method, measured by BLEU metrics for translation quality. Our approach achieved BLEU scores of 29.17 for the Indonesian-English (ID-EN) task and 30.39 for the Indonesian-Arabic (ID-AR) task."

T2_TITLE="Direct pyrolysis of rock asphalt from Buton Indonesia: characterization of liquid product and the role of calcite"
T2_KW="Buton rock asphalt;Calcite;Diesel fuel;Pyrolysis"
T2_ABS="Buton rock asphalt (RA) majority composed of a mixture of calcite and petroleum-like material that is abundantly available in Buton, Indonesia. Direct pyrolysis of RA reveals the catalytic role of calcite in efficiently transforming the fused aromatic rings of asphaltene in RA into diesel-range hydrocarbons. Pyrolysis at 400 C for 180 min produced a 74% liquid product yield, significantly higher than that obtained from pyrolysis of decalcified RA (22.9%), or decalcified RA using commercial CaCO3 (34.2%). The resulting liquid hydrocarbon contains 70.30% diesel fraction, 25.72% heavy oil, and 3.98% gasoline, and exhibits a higher cetane number (69) than commercial diesel fuel. These findings highlight the importance of strong calcite/asphaltene interactions for efficient heat transfer during hydro-dearomatization of aromatic rings. Furthermore, the high thermal stability and strong basicity of calcite catalyze the cracking process into diesel-range hydrocarbons. This study offers a roadmap for the efficient conversion of low-quality Buton rock asphalt into high-quality diesel fuel through direct pyrolysis at 400 C."

T3_TITLE="Building a Spatio-Temporal Ontology for Artifacts Knowledge Management"
T3_KW="spatio-temporal ontology,historical artifacts,artifacts knowledge management"
T3_ABS="The knowledge that embedded in a historical artifact can have multidimensional information, such as time (temporal) and place (spatial) dimension. The temporal dimension indicates when the artifact had been used in the past, while spatial dimension points the location of people who had been using it at the time. Both of these information provide a general overview of the civilization conditions at the artifacts time. In most cases, the spatio-temporal information that attached to an artifact can be used to furnish the missing information of the other artifacts. If the museum managers are able to connect artifacts based on their spatio-temporal information, presenting artifacts historical value to visitors will be more continuous and complete. However, this kind of management needs could not be facilitated by any existing conventional database systems today. The author proposed an ontology approach for storing artifacts spatio-temporal information in digital form. This ontology is equipped with rules to perform reasoning thus spatio-temporal information among artifacts could be connected automatically. The result shows that the spatio-temporal ontology can be implemented in order to complete information linkage among the artifacts."

# ---- Runner ----
run_s4() {
    local label="$1"
    local title="$2"
    local kw="$3"
    local abs="$4"

    echo "" | tee -a "$LOG"
    echo "[s4_ablation] >>> $label  ($(date +%H:%M:%S))" | tee -a "$LOG"

    QUERY_TITLE="$title" \
    QUERY_KEYWORDS="$kw" \
    QUERY_ABSTRACT="$abs" \
    "$PYTHON" orchestrate.py \
        --scenarios baseline,s4a,s4b,s4c,s4d,s4e \
        --ablation-group s4_algorithms \
        --query-label "$label" 2>&1 | tee -a "$LOG"

    echo "[s4_ablation] <<< $label done  ($(date +%H:%M:%S))" | tee -a "$LOG"
}

# ---- Topic 1: NMT ----
run_s4 "t1_nmt_full"       "$T1_TITLE" "$T1_KW"  "$T1_ABS"
run_s4 "t1_nmt_title"      "$T1_TITLE" ""         ""
run_s4 "t1_nmt_kw"         ""          "$T1_KW"   ""
run_s4 "t1_nmt_abs"        ""          ""          "$T1_ABS"
run_s4 "t1_nmt_title_kw"   "$T1_TITLE" "$T1_KW"   ""
run_s4 "t1_nmt_title_abs"  "$T1_TITLE" ""          "$T1_ABS"
run_s4 "t1_nmt_kw_abs"     ""          "$T1_KW"   "$T1_ABS"

# ---- Topic 2: Pyrolysis ----
run_s4 "t2_pyrolysis_full"       "$T2_TITLE" "$T2_KW"  "$T2_ABS"
run_s4 "t2_pyrolysis_title"      "$T2_TITLE" ""         ""
run_s4 "t2_pyrolysis_kw"         ""          "$T2_KW"   ""
run_s4 "t2_pyrolysis_abs"        ""          ""          "$T2_ABS"
run_s4 "t2_pyrolysis_title_kw"   "$T2_TITLE" "$T2_KW"   ""
run_s4 "t2_pyrolysis_title_abs"  "$T2_TITLE" ""          "$T2_ABS"
run_s4 "t2_pyrolysis_kw_abs"     ""          "$T2_KW"   "$T2_ABS"

# ---- Topic 3: Ontology ----
run_s4 "t3_ontology_full"       "$T3_TITLE" "$T3_KW"  "$T3_ABS"
run_s4 "t3_ontology_title"      "$T3_TITLE" ""         ""
run_s4 "t3_ontology_kw"         ""          "$T3_KW"   ""
run_s4 "t3_ontology_abs"        ""          ""          "$T3_ABS"
run_s4 "t3_ontology_title_kw"   "$T3_TITLE" "$T3_KW"   ""
run_s4 "t3_ontology_title_abs"  "$T3_TITLE" ""          "$T3_ABS"
run_s4 "t3_ontology_kw_abs"     ""          "$T3_KW"   "$T3_ABS"

echo "" | tee -a "$LOG"
echo "[s4_ablation] All 21 combos done: $(date)" | tee -a "$LOG"
