# Data LesioScan (hanya runtime Colab)

- `metadata.csv`: metadata mentah PAD-UFES-20 (jangan ditimpa)
- `images/`: gambar asli setelah ekstraksi (jangan ditimpa)
- `metadata_clean.csv`: keluaran `scripts/clean.py` (metadata yang lolos pemeriksaan)
- `split.csv`: keluaran `scripts/prepare.py` berdasarkan metadata bersih

Jalankan: `scripts/eda.py` → `scripts/clean.py` → tinjau audit → `scripts/prepare.py` → `scripts/train.py`.
Simpan dataset dan artefak hasil ke Google Drive, bukan GitHub.
