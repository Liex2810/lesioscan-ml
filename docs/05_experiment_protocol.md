# Protokol Eksperimen dan Reproduksibilitas

- Gunakan `configs/default.yaml` yang sama atau salin konfigurasi setiap run ke `experiments/`.
- Catat versi Python, torch, torchvision, timm, tanggal eksekusi, GPU, seed, commit Git, dan urutan label.
- Simpan `data/split.csv` ke Drive dan **gunakan file identik** untuk semua model.
- Jalankan Stage 1 (head) lalu Stage 2 (fine-tuning backbone) sesuai script train.
- Bandingkan hasil **validation** dengan `scripts/benchmark.py`.
- Semua metrik dari model yang dibandingkan wajib ditampilkan; kegagalan eksperimen juga dicatat.
- Run yang namanya sama dapat menimpa checkpoint lama. Gunakan opsi `--run <nama_unik>`.
- Untuk laporan lebih kuat, implementasikan group-aware cross-validation pada development set; script saat ini hanya single holdout.

## Template log eksperimen
| Run | Git commit | Seed | Model | Epoch stage 1/2 | Batch | LR 1/2 | Macro-F1 val | Catatan |
|---|---|---|---|---|---|---|---|---|
| [belum dijalankan] | - | 42 | - | 5/20 | 8 | 0.001/0.00003 | - | - |
