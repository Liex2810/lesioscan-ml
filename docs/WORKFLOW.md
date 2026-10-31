# Alur Pengerjaan LesioScan

## 1. Persiapan sebelum dataset

1. Aktifkan GPU pada Google Colab dan periksa `torch.cuda.is_available()`.
2. Hubungkan Drive dan buat folder `LesioScan-AI/{datasets,checkpoints,experiments,models}`.
3. Clone repository ke `/content/lesioscan-ml` dan gunakan folder itu sebagai direktori kerja.
4. Pasang kebutuhan library. Pertahankan pasangan Torch/Torchvision bawaan Colab yang sudah cocok dengan GPU.
5. Baca `configs/default.yaml` dan impor package `lesioscan` dari `src`.
6. Jalankan pengujian sintetis: `PYTHONPATH=src python -m pytest tests/test_splits.py -q`.
7. Simpan catatan lingkungan dan notebook ke Drive. Tahap ini belum mengukur akurasi.

## 2. Persiapan dataset

1. Gunakan PAD-UFES-20 dari sumber resmi yang ditautkan README.
2. Periksa susunan arsip sebelum ekstraksi; jangan menebak letak file.
3. Siapkan `data/metadata.csv` serta `data/images/`.
4. Pastikan `img_id`, `patient_id`, dan `diagnostic` tersedia dan seluruh nama gambar cocok.
5. Audit gambar rusak, duplikasi gambar/lesi, pasien, dan distribusi kelas.
6. Jalankan `python scripts/prepare.py`; periksa cakupan kelas dan kelompok pasien hasil split.
7. Cadangkan `data/split.csv` ke Drive. Gunakan split yang sama untuk perbandingan model.

## 3. Pelatihan dan pemilihan model

1. Mulai dari ShuffleNet V2 untuk memperoleh satu baseline yang dapat diulang.
2. Jalankan `python scripts/train.py --model shufflenet_v2_x1_0 --run shufflenet_baseline`.
3. Periksa loss, macro-F1 validasi, dan recall per kelas.
4. Cadangkan folder run selama/selepas eksperimen; output lokal Colab bersifat sementara.
5. Setelah baseline dipahami, bandingkan DenseNet-121 dan ViT-Small pada protokol yang sama.
6. Pilih model memakai validation; jangan memakai test untuk memilih konfigurasi.

## 4. Evaluasi akhir dan integrasi

1. Bekukan pilihan model dan protokol sebelum membuka hasil holdout test.
2. Jalankan `evaluate.py` pada checkpoint terpilih dan simpan laporan.
3. Ekspor memakai `export_onnx.py`.
4. Uji kesetaraan pada gambar nyata serta latensi perangkat target sebelum integrasi Flutter.

## Batas implementasi

- Pembagian starter memakai GroupShuffleSplit berbasis pasien, belum stratifikasi kelompok atau 5-fold CV.
- Fraksi pembagian mengacu kelompok pasien; proporsi jumlah gambar dapat berbeda.
- Tambahkan patient-grouped CV pada partisi pengembangan jika penelitian memerlukan estimasi yang lebih kuat. Pertahankan holdout pasien untuk evaluasi akhir.
- Checkpoint terbaik menyimpan bobot dan konfigurasi, belum state optimizer/scaler lengkap untuk resume training.
- Ekspor menguji paritas input sintetis; ini belum cukup untuk menyatakan integrasi aplikasi tervalidasi.
- Prototipe bukan alat diagnosis medis yang tervalidasi.
