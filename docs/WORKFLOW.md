# Alur kerja LesioScan ML

1. **GitHub** menyimpan kode; **Colab** menjalankannya; **Drive** menyimpan dataset dan model.
2. **EDA**: `python scripts/eda.py` — analisis kelas, pasien, resolusi, berkas hilang/rusak.
3. **Split**: `python scripts/prepare.py` — pembagian berdasarkan `patient_id`.
4. **Preprocessing**: `src/lesioscan/preprocessing.py` — augmentasi hanya untuk training; normalisasi ImageNet.
5. **Feature Extraction**: `src/lesioscan/feature_extraction.py` — backbone dibekukan, classifier dilatih.
6. **Fine-Tuning**: `src/lesioscan/fine_tuning.py` — backbone dibuka dan dilatih dengan learning rate lebih kecil.
7. **Validasi** pada setiap epoch, simpan `best.pt` berdasarkan macro-F1.
8. **Benchmark**: train tiga model dengan split sama, bandingkan validation macro-F1 dan recall.
9. **Evaluasi akhir** pada test set yang tidak digunakan untuk memilih model.
10. **Ekspor**: `python scripts/export_onnx.py --checkpoint <path/best.pt>`.

> Tahap sebelum dataset: repository dan Google Colab siap; dataset masih diunduh.
