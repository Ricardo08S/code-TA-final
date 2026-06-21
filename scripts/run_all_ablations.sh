#!/usr/bin/env bash
# Master ablation runner — jalan otomatis tanpa intervensi manual
# Urutan: tunggu S4 → S3 compile (standalone) → S1 → S3 run → S2
# Error handling: kalau satu step gagal, tetap lanjut ke step berikutnya
#
# Cara pakai:
#   nohup bash scripts/run_all_ablations.sh > logs/run/master_nohup.out 2>&1 &
#   echo $! > logs/master.pid

set -uo pipefail   # -u: undefined var error; -e dihapus agar lanjut walau ada error
cd "$(dirname "$0")/.."

LOG="logs/run/master_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/run

log() {
    echo "[master] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

run_step() {
    local step="$1"
    local desc="$2"
    shift 2
    log ""
    log "========================================="
    log "$step — $desc"
    log "========================================="
    if "$@" 2>&1 | tee -a "$LOG"; then
        log "$step SELESAI."
        return 0
    else
        log "WARNING: $step GAGAL (exit code $?). Lanjut ke step berikutnya."
        return 1
    fi
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

# ── Step 2: S3 compile (standalone) ──────────────────────────────────────
S3_COMPILED=false
if run_step "STEP 2/5" "S3 compile standalone (~2-3 jam)" bash scripts/compile_s3_configs.sh; then
    S3_COMPILED=true
fi

# ── Step 3: S1 ablation ───────────────────────────────────────────────────
run_step "STEP 3/5" "S1 ablation 6 config × 21 query = 126 run (~1 jam)" bash scripts/run_s1_ablation.sh || true

# ── Step 4: S3 run (hanya kalau compile berhasil) ─────────────────────────
if [[ "$S3_COMPILED" == "true" ]]; then
    run_step "STEP 4/5" "S3 run 4 config × 21 query = 84 run (~49 menit)" bash scripts/run_s3_ablation.sh || true
else
    log ""
    log "STEP 4/5 — S3 run DI-SKIP karena compile gagal."
fi

# ── Step 5: S2 ablation ───────────────────────────────────────────────────
run_step "STEP 5/5" "S2 ablation 6 config × 21 query = 126 run (~2.5 jam)" bash scripts/run_s2_ablation.sh || true

# ── Selesai ────────────────────────────────────────────────────────────────
log ""
log "========================================="
log "MASTER SELESAI: $(date)"
log "Hasil: output/ablation/"
log "========================================="
