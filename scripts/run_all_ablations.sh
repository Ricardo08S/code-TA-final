#!/usr/bin/env bash
# Master ablation runner — jalan otomatis tanpa intervensi manual
# Urutan: tunggu S4 → S3 compile (standalone) → S1 → S3 run → S2
#
# Cara pakai:
#   nohup bash scripts/run_all_ablations.sh > logs/run/master_nohup.out 2>&1 &
#   echo $! > logs/master.pid

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="logs/run/master_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run

log() {
    echo "[master] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

log "========================================="
log "Master ablation run started"
log "Log: $LOG"
log "========================================="

# ── Step 1: Tunggu S4 selesai ─────────────────────────────────────────────
S4_PID_FILE="logs/nohup.pid"
if [[ -f "$S4_PID_FILE" ]]; then
    S4_PID=$(cat "$S4_PID_FILE")
    if kill -0 "$S4_PID" 2>/dev/null; then
        log "S4 masih jalan (PID=$S4_PID), menunggu..."
        while kill -0 "$S4_PID" 2>/dev/null; do
            sleep 60
        done
        log "S4 selesai."
    else
        log "S4 sudah selesai sebelumnya, lanjut."
    fi
else
    log "logs/nohup.pid tidak ditemukan, asumsikan S4 sudah selesai."
fi

# ── Step 2: S3 compile (standalone — jangan tambah proses lain) ───────────
log ""
log "========================================="
log "STEP 2/5 — S3 compile (standalone)"
log "Estimasi: 2-3 jam"
log "========================================="
bash scripts/compile_s3_configs.sh 2>&1 | tee -a "$LOG"
log "S3 compile selesai."

# ── Step 3: S1 ablation ────────────────────────────────────────────────────
log ""
log "========================================="
log "STEP 3/5 — S1 ablation (6 config × 21 query = 126 run)"
log "Estimasi: ~1 jam"
log "========================================="
bash scripts/run_s1_ablation.sh 2>&1 | tee -a "$LOG"
log "S1 selesai."

# ── Step 4: S3 run (setelah compile cache tersedia) ────────────────────────
log ""
log "========================================="
log "STEP 4/5 — S3 run (4 config × 21 query = 84 run)"
log "Estimasi: ~49 menit"
log "========================================="
bash scripts/run_s3_ablation.sh 2>&1 | tee -a "$LOG"
log "S3 run selesai."

# ── Step 5: S2 ablation (terakhir, S2b cukup berat) ──────────────────────
log ""
log "========================================="
log "STEP 5/5 — S2 ablation (6 config × 21 query = 126 run)"
log "Estimasi: ~2.5 jam"
log "========================================="
bash scripts/run_s2_ablation.sh 2>&1 | tee -a "$LOG"
log "S2 selesai."

# ── Selesai ────────────────────────────────────────────────────────────────
log ""
log "========================================="
log "SEMUA ABLASI SELESAI: $(date)"
log "Hasil: output/ablation/"
log "========================================="
