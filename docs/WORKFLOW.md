# Alur kerja LesioScan ML

1. **GitHub** menyimpan kode; **Colab** menjalankannya; **Drive** menyimpan dataset dan model.
2. **EDA**: `python scripts/eda.py` — analisis kelas, pasien, resolusi, berkas hilang/rusak.
3. **Cleaning**: `python scripts/clean.py` — catat duplikasi, metadata kosong, kelas tak dikenal, gambar hilang/rusak; output `metadata_clean.csv` dan laporan audit, data asli tidak diubah.
4. **Split**: `python scripts/prepare.py` — pembagian berdasarkan `patient_id`.
5. **Preprocessing**: `src/lesioscan/preprocessing.py` — augmentasi hanya untuk training; normalisasi ImageNet.
6. **Feature Extraction**: `src/lesioscan/feature_extraction.py` — backbone dibekukan, classifier dilatih.
7. **Fine-Tuning**: `src/lesioscan/fine_tuning.py` — backbone dibuka dan dilatih dengan learning rate lebih kecil.
7. **Validasi** pada setiap epoch, simpan `best.pt` berdasarkan macro-F1.
8. **Benchmark**: train tiga model dengan split sama, bandingkan validation macro-F1 dan recall.
9. **Evaluasi akhir** pada test set yang tidak digunakan untuk memilih model.
10. **Ekspor**: `python scripts/export_onnx.py --checkpoint <path/best.pt>`.

> Tahap sebelum dataset: repository dan Google Colab siap; dataset masih diunduh.


## Tahap rubrik tambahan
- `scripts/benchmark.py`: ringkasan metrik validasi tiga model.
- `scripts/plot_training.py`: grafik loss dan metrik validasi.
- `scripts/visualize_evaluation.py`: confusion matrix dan metrik per kelas.
- `scripts/predict.py`: prediksi satu gambar untuk pengujian teknis (bukan diagnosis).
- `scripts/benchmark_inference.py`: latensi tensor sintetis.
- `src/lesioscan/splitting.py`: memilih kandidat pembagian pasien dengan distribusi kelas lebih seimbang.
