#!/usr/bin/env bash
# S3 Compile: compile circuit untuk 3 config baru secara standalone
# WAJIB dijalankan sebelum run_s3_ablation.sh
# Config nf64_td32 sudah di-cache, tidak perlu compile ulang.
#
# Estimasi compile:
#   nf8_td8   → ~10-20 menit
#   nf16_td16 → ~30-60 menit
#   nf32_td16 → ~60-90 menit
#   Total     → ~100-170 menit (~2-3 jam)
#
# Jalankan standalone (jangan berbarengan dengan proses lain) untuk hindari OOM.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
LOG="logs/run/s3_compile_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run
echo "[s3_compile] Start: $(date)" | tee -a "$LOG"

compile_s3() {
    local config="$1"
    local nf="$2"
    local td="$3"
    echo "[s3_compile] >>> Compiling ${config}  ($(date +%H:%M:%S))" | tee -a "$LOG"
    (
        export S3_N_FEATURES="$nf"
        export S3_TARGET_DIM="$td"
        "$PYTHON" orchestrate.py --scenarios s3
    ) 2>&1 | tee -a "$LOG"
    echo "[s3_compile] <<< ${config} done  ($(date +%H:%M:%S))" | tee -a "$LOG"
}

# Compile 3 config baru (urutan dari terkecil ke terbesar)
compile_s3 "nf8_td8"   8  8
compile_s3 "nf16_td16" 16 16
compile_s3 "nf32_td16" 32 16

echo "[s3_compile] All 3 configs compiled: $(date)" | tee -a "$LOG"
echo "[s3_compile] Sekarang aman jalankan: bash scripts/run_s3_ablation.sh" | tee -a "$LOG"
