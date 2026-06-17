# code-final — Privacy-Preserving Author Recommendation Benchmark

Implementasi dan orkestrasi semua skenario eksperimen untuk tugas akhir:
**Perbandingan skema Homomorphic Encryption (HE) pada sistem rekomendasi author berbasis kemiripan embedding.**

---

## Daftar Skenario

| ID | Folder | Skema | Embedding | Similarity | Ranking | Dim | Pool |
|----|--------|-------|-----------|-----------|---------|-----|------|
| Baseline | `baseline` | Plaintext | Client (MiniLM) | Server (plaintext) | Client | 384 | 2620 (semua) |
| S1 | `s1_tfhe` | TFHE (Concrete) | Server (surrogate Ridge) | Server (encrypted) | Server (encrypted) | 8 | **8 cluster medoid** |
| S2a | `s2a_tfhe` | TFHE (Concrete) | Client (MiniLM) | Server (encrypted) | Server (encrypted) | 16 | **8 cluster medoid** |
| S2b | `s2b_ckks` | CKKS (OpenFHE + bootstrap) | Client (MiniLM) | Server (encrypted) | Server approx comparison | 64 | **20 cluster medoid** |
| S3 | `s3_tfhe` | TFHE (Concrete) | Server (surrogate Ridge) | Server (encrypted) | **Client** | **32** | 2620 (semua) |
| S4a | `s4a_tfhe` | TFHE (Concrete) | Client (MiniLM) | Server (encrypted) | Client | **384** | 2620 (semua) |
| S4b | `s4b_ckks` | CKKS (TenSEAL) | Client (MiniLM) | Server (encrypted) | Client | **384** | 2620 (semua) |
| S4c | `s4c_phe_paillier` | PHE Paillier | Client (MiniLM) | Server (encrypted) | Client | **384** | 2620 (semua) |
| S4d | `s4d_she_bgv` | SHE BGV (OpenFHE) | Client (MiniLM) | Server (encrypted) | Client | **384** | 2620 (semua) |
| S4e | `s4e_she_bfv` | SHE BFV (TenSEAL) | Client (MiniLM) | Server (encrypted) | Client | **384** | 2620 (semua) |

**Cluster Medoid** (S1/S2a/S2b): server menjalankan KMeans pada mean TK embedding semua 2620 author, lalu memilih satu representatif per cluster. Deterministik, query-independent, sepenuhnya server-side — privacy model tetap terjaga.
S4a–S4e menggunakan full 384-dim karena tidak ada encrypted ranking (hanya dot product, ranking di client).

---

## Struktur Folder

```
code-final/
├── orchestrate.py          # Jalankan semua skenario
├── .env.example            # Template konfigurasi
├── requirements.txt        # Dependensi
├── core/                   # Utilitas bersama
│   ├── config.py           # Konstanta (EMBED_DIM, WEIGHT_TK, dll)
│   ├── data_loader.py      # Load author subprofiles dari cache
│   ├── embedder.py         # MiniLM embedding (sentence-transformers)
│   ├── quantize.py         # quantize() / dequantize()
│   ├── scoring.py          # combine_modal_scores, aggregate_max_scores
│   ├── result_writer.py    # format_top_k, write_json, append_csv
│   ├── reduction.py        # Dimension reduction (PCA, truncate, dll)
│   └── surrogate.py        # Surrogate Ridge model + TFHE circuit builders
├── usecases/
│   ├── baseline/run.py
│   ├── s1_tfhe/run.py
│   ├── s2a_tfhe/run.py
│   ├── s2b_ckks/run.py
│   ├── s3_tfhe/run.py
│   ├── s4a_tfhe/run.py
│   ├── s4b_ckks/run.py
│   ├── s4c_phe_paillier/run.py
│   ├── s4d_she_bgv/run.py
│   └── s4e_she_bfv/run.py
├── logs/                   # Log orchestrator per run (gitignore)
├── output/                 # Hasil terbaru + output/runs/<timestamp>/ (gitignore)
└── artifacts/              # Surrogate model + compiled circuit cache (gitignore)
```

---

## Setup

### 1. Install dependensi

```bash
uv pip install -r requirements.txt
```

Untuk TFHE dengan GPU (opsional, jauh lebih cepat — ganti `concrete-python` CPU dengan versi GPU):
```bash
uv pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python
```

### 2. Buat file `.env`

```bash
cp .env.example .env
# Edit .env: isi AUTHOR_PROFILE_CACHE_PATH dan QUERY_*
```

Minimal yang wajib diisi:
```
AUTHOR_PROFILE_CACHE_PATH=/path/to/cache.pkl
QUERY_TITLE=privacy preserving author recommendation
QUERY_KEYWORDS=homomorphic encryption
```

---

## Cara Menjalankan

Semua skenario dibandingkan terhadap **baseline** (plaintext 384-dim) sebagai acuan — dari sisi waktu (seberapa lambat vs baseline) dan akurasi (apakah top-K rekomendasinya sama dengan baseline).

### Semua skenario sekaligus

```bash
./venv/bin/python orchestrate.py --env-file .env
```

### Skenario tertentu saja

```bash
./venv/bin/python orchestrate.py --env-file .env --scenarios baseline,s4b,s4c
```

### Satu skenario sebagai benchmark resmi

Gunakan orchestrator walaupun hanya menjalankan satu skenario. Hasilnya tetap
masuk ke `output/runs/<timestamp>/`.

```bash
./venv/bin/python orchestrate.py --env-file .env --scenarios s4b
```

### Satu skenario langsung untuk debug

```bash
./venv/bin/python -m usecases.s4b_ckks.run
```

Run langsung seperti ini tidak dianggap benchmark resmi. Output-nya diisolasi ke:

```text
output/manual/<timestamp>_s4b/result_s4b.json
output/manual/<timestamp>_s4b/timing_s4b.csv
```

### Generate RUN_SUMMARY.md tanpa rerun (konsolidasi hasil yang sudah ada)

```bash
./venv/bin/python orchestrate.py --env-file .env --summary-only
# atau hanya skenario tertentu:
./venv/bin/python orchestrate.py --env-file .env --scenarios baseline,s1,s2a,s3,s4a,s4b,s4c,s4d,s4e --summary-only
```

Berguna ketika hasil parsial tersebar di beberapa run (misalnya S4c selesai 72 menit yang lalu, S1/S3 selesai 20 menit yang lalu), dan kamu ingin membuat satu RUN_SUMMARY.md konsolidasi tanpa mengulangi komputasi berat.

### Background run yang aman ditinggal

```bash
# Cara mudah (direkomendasikan): flag --background sudah handle setsid + log otomatis
./venv/bin/python orchestrate.py --env-file .env --background
# → mencetak PID dan perintah tail, lalu keluar
# → output tersimpan otomatis ke logs/run/run_<timestamp>_full.out

# Monitor progress:
tail -f logs/run/run_*_full.out
```

Orchestrator selalu membuat dua log secara otomatis di folder terpisah:
- `logs/orchestrate/<timestamp>.log` — hanya event START/DONE per skenario (ringkas)
- `logs/run/<timestamp>_full.out` — seluruh output detail (semua print dari tiap run.py)

Log tidak disalin ke dalam run folder — cukup di `logs/orchestrate/` dan `logs/run/`.

### Urutan yang disarankan

Beberapa skenario membutuhkan waktu lama (S1/S3 karena TFHE compile, S4c karena operasi Paillier sangat berat). Jalankan bertahap:

```bash
# Tahap 1: relatif cepat
./venv/bin/python orchestrate.py --env-file .env --scenarios baseline,s4b,s4d,s4e

# Tahap 2: stabil tapi lebih lama
./venv/bin/python orchestrate.py --env-file .env --scenarios s4a
setsid ./venv/bin/python orchestrate.py --env-file .env --scenarios s4c \
  > logs/run_s4c.out 2>&1 < /dev/null &
tail -f logs/run_s4c.out

# Tahap 3: encrypted ranking/comparison (S1/S2a recompile circuit ~5-8 menit, S2b bootstrap ~116s)
./venv/bin/python orchestrate.py --env-file .env --scenarios s1,s2a,s2b,s3 --background
tail -f logs/run/run_*_full.out
```

Di akhir setiap run, orchestrator:
1. Membuat folder **`output/runs/<timestamp>/`**
2. Menyalin hasil JSON ke **`results/`** dan timing CSV ke **`timing/`**
3. Menulis **`RUN_SUMMARY.md`** dan **`run_meta.json`**
4. Memperbarui symlink **`output/runs/latest/`** ke run terbaru
5. Mencetak summary table di terminal

Log tersimpan terpisah — tidak di dalam run folder:
- `logs/orchestrate/orchestrate_<timestamp>.log` — event per skenario (ringkas)
- `logs/run/run_<timestamp>_full.out` — full output detail

### Membersihkan hasil run

```bash
# Hapus output/ dan logs/
./venv/bin/python orchestrate.py --clean

# Full reset: output/, logs/, artifacts/, dan .artifacts/
./venv/bin/python orchestrate.py --clean --clean-artifacts

# Hanya hapus cache/artifact compile
./venv/bin/python orchestrate.py --clean-artifacts
```

### Menyimpan PID nohup yang sedang run

```bash
echo $! > logs/nohup.pid
```

Di akhir setiap run, orchestrator mencetak summary table otomatis yang mencakup **waktu** dan **akurasi vs baseline**:

```
================================================================================
SUMMARY  (accuracy compared vs baseline)
================================================================================
Scenario     Status   Total (s)   Overlap@K   Exact@K  Notes
--------------------------------------------------------------------------------
baseline     OK            1.23      (ref)      (ref)
s4b          OK            4.87        5/5        5/5
s4c          OK           38.10        5/5        4/5
s4a          OK          142.30        5/5        3/5
s1           OK          312.50        4/5        2/5
...
--------------------------------------------------------------------------------
TOTAL                    498.00
================================================================================
  Overlap@K : berapa author dari top-K skenario ada di top-K baseline (urutan diabaikan)
  Exact@K   : berapa author yang posisi ranknya sama persis dengan baseline
```

- **Overlap@K** — mengukur apakah skenario berhasil menemukan author yang relevan (set intersection). Nilai `5/5` = semua top-5 sama persis.
- **Exact@K** — mengukur apakah urutan peringkatnya identik. Nilai `5/5` = ranking sempurna sama dengan baseline.

---

## Output

Hasil benchmark resmi ditulis langsung ke folder run timestamped. Untuk membaca
hasil final, gunakan:

```text
# Hasil run
output/runs/latest/RUN_SUMMARY.md
output/runs/latest/run_meta.json
output/runs/latest/results/result_<scenario>.json
output/runs/latest/timing/timing_<scenario>.csv

# Log (terpisah dari output, bukan di dalam output/)
logs/orchestrate/orchestrate_<timestamp>.log   # event log per run (ringkas)
logs/run/run_<timestamp>_full.out              # full output log per run
```

Run direct module/debug ditulis terpisah:

```text
output/manual/<timestamp>_<scenario>/result_<scenario>.json
output/manual/<timestamp>_<scenario>/timing_<scenario>.csv
```

Setiap skenario menghasilkan:

**`output/runs/<timestamp>/results/result_<scenario>.json`** — hasil rekomendasi:
```json
{
  "scenario": "s4b",
  "top_k": [
    {"rank": 1, "author_id": "author_123", "score": 0.87},
    ...
  ],
  "timing": {
    "embed_sec": 0.12,
    "encrypt_sec": 0.03,
    "run_sec": 1.45,
    "decrypt_sec": 0.02,
    "total_sec": 1.62
  },
  "config": {"scheme": "CKKS", "dim": 384, ...}
}
```

**`output/runs/<timestamp>/timing/timing_<scenario>.csv`** — log timing per run:
```
timestamp_iso,epoch_sec,embed_sec,encrypt_sec,run_sec,decrypt_sec,total_sec
2026-06-11T...,1749...,0.12,0.03,1.45,0.02,1.62
```

---

## Penjelasan Alur Tiap Skenario

### Baseline
```
Query text → MiniLM (384-dim) → dot product plaintext → sort → Top-K
```

### S1 — TFHE E2E Surrogate
```
OFFLINE (server, sekali):
  Subprofile texts + MiniLM cache → HashingVectorizer → Ridge.fit()
  → quantize coef → compile TFHE circuit (Phase1: embed, Phase2: top-K sort)
  → simpan server.zip + client.zip

ONLINE (per query):
  Query text → HashingVectorizer → binary features → ENCRYPT
  → server.run() [ciphertext: surrogate embed → scores → top-K sort]
  → DECRYPT → Top-K
```

### S2a — TFHE Encrypted Ranking
```
Query text → MiniLM → quantize → ENCRYPT (TFHE)
→ server: encrypted dot product × N authors → encrypted selection sort
→ DECRYPT → Top-K only (client tidak lihat semua skor)
```

### S2b — CKKS Approximate Ranking
```
Query text → MiniLM → PCA → CKKS ENCRYPT (OpenFHE)
→ server: encrypted dot product per candidate author
→ EvalBootstrap setiap encrypted author score
→ approximate pairwise comparison via polynomial sign(x) ≈ 1.5x - 0.5x³
→ DECRYPT comparison signs → vote ranking → Top-K
```

### S3 — TFHE Surrogate, Client Rank
```
Sama dengan S1 (offline surrogate + online encrypt binary features)
TAPI circuit hanya Phase 1 (embed + scores, no top-K sort)
→ semua N encrypted scores dikembalikan ke client
→ DECRYPT semua → sort plaintext → Top-K
```

### S4a–S4e — Variasi Skema, Client Rank
```
Query text → MiniLM (384-dim, full) → [quantize] → ENCRYPT (skema masing-masing)
→ server: encrypted dot product per subprofile
→ DECRYPT semua scores → aggregate max per author → sort → Top-K
```

---

## Keterbatasan

| Skenario | Keterbatasan |
|----------|-------------|
| S1 | Surrogate (HashVec 16 bucket + Ridge 8-dim) approksimasi MiniLM; `profile_scale=2` wajib kecil agar TLU tidak overflow; pool 8 author cluster medoid |
| S3 | Surrogate (HashVec **64 bucket** + Ridge **32-dim**) lebih representatif dari S1; circuit linear (no TLU) → scale bebas; compile ~80 menit (2620 author × 32 dim) |
| S1, S2a | Pool dibatasi **8 author** (cluster medoid) — TFHE TLU complexity eksponensial; circuit di-cache, hanya recompile jika config berubah |
| S2a | `enc_topk_scale=1` wajib agar TLU input ≤ 16 bit; maks 32 authors |
| S2b | Pool **20 author** (cluster medoid) — EvalBootstrap ~2.4s/author; full 2620 → ~17 jam (tidak praktis). Backend OpenFHE (bukan TenSEAL) karena hanya OpenFHE yang support bootstrapping |
| S4c | Paillier sangat lambat: ~73 menit untuk 2620 author (5466 subprofile × 384 dim = 4.2 juta modular exponentiation 2048-bit) |
| S4d | `plain_modulus` wajib `≡ 1 (mod 65536)` untuk NTT packed encoding — jika tidak, OpenFHE crash (C++ abort, tidak bisa di-catch Python) |
| S4e | `plain_modulus` wajib `≡ 1 (mod 2×poly_modulus_degree)`; plus BFV mod correction untuk skor negatif setelah dekripsi |

---

## Env Vars Lengkap

| Var | Default | Keterangan |
|-----|---------|-----------|
| `AUTHOR_PROFILE_CACHE_PATH` | — | **Wajib.** Path ke cache .pkl |
| `QUERY_TITLE` | `"privacy preserving..."` | Judul query |
| `QUERY_KEYWORDS` | `"homomorphic encryption"` | Keywords query |
| `QUERY_ABSTRACT` | `""` | Abstract query (opsional) |
| `S1_MAX_AUTHORS` | semua | Batasi jumlah author |
| `S1_TOP_K` | 1 | Jumlah rekomendasi; default disamakan dengan `code-TFHE` UC02 |
| `S1_N_FEATURES` | 16 | Bucket HashingVectorizer untuk konfigurasi UC02 |
| `S1_TARGET_DIM` | 16 | Dimensi surrogate embedding |
| `S1_COEF_SCALE` | 4 | Scale kuantisasi koefisien Ridge untuk konfigurasi UC02 |
| `S1_PROFILE_SCALE` | 2 | Scale kuantisasi profil subprofile |
| `S1_SERVER_DEVICE` | `cuda` | `cpu`, `gpu`, atau `cuda`; gunakan CUDA jika tersedia seperti UC02 |
| `S3_N_FEATURES` | 64 | Bucket HashingVectorizer query encoding (lebih besar dari S1; aman karena no TLU) |
| `S3_TARGET_DIM` | 32 | Dimensi surrogate embedding (lebih besar dari S1; circuit linear, compile ~80 menit) |
| `S3_COEF_SCALE` | 8 | Scale kuantisasi koefisien Ridge |
| `S3_PROFILE_SCALE` | 8 | Scale kuantisasi profil (bisa besar karena no TLU overflow risk) |
| `S2A_ENC_TOPK_SCALE` | 1 | Scale kuantisasi embedding S2a; **harus 1** agar output `sel()` ≤ 16 bit |
| `S2A_MAX_AUTHORS` | 8 | Ukuran pool cluster medoid untuk S2a; pool selalu dipilih via KMeans medoid |
| `S2B_MAX_AUTHORS` | 20 | Ukuran pool cluster medoid untuk S2b; dibatasi oleh biaya EvalBootstrap (~2.4s/author) |
| `S4C_KEY_BITS` | 2048 | Paillier key size |
| `S4D_MULT_DEPTH` | 2 | BGV multiply depth |
| `S4D_PLAIN_MODULUS` | auto | Prima pertama > 2×scale² (dihitung otomatis) |
| `S4E_PLAIN_MODULUS` | auto | Prima pertama > 2×scale² (dihitung otomatis) |
| `S4E_POLY_MODULUS_DEGREE` | auto | Power-of-2 ≥ 2×EMBED_DIM, min 4096 (dihitung otomatis) |
