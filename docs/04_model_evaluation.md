# Protokol Evaluasi LesioScan

**Status:** rancangan sebelum eksperimen; belum ada skor aktual.

## Protokol
1. Dokumentasikan asal dataset, jumlah pasien, missing images, kelas, dan pembagian data.
2. Bangun split pasien yang konsisten. Jangan memasukkan foto pasien yang sama ke split berbeda.
3. Jalankan tiga arsitektur dengan split dan metrik identik. Catat perubahan parameter bila ada.
4. Gunakan hasil **validasi** untuk memilih model/konfigurasi; jangan membuka test set untuk tuning.
5. Setelah pemilihan final, jalankan `scripts/evaluate.py` satu kali pada holdout.
6. Laporkan confusion matrix dan recall MEL, BCC, SCC dengan konteks jumlah sampel.
7. Uji inferensi/ONNX dan ukur latensi pada perangkat target terpisah.

## Metrik dan makna
- **Accuracy:** proporsi semua prediksi benar, bisa menutupi ketidakseimbangan.
- **Macro-F1:** rata-rata F1 setiap kelas tanpa memberikan bobot lebih pada kelas besar.
- **Balanced accuracy:** rata-rata recall kelas yang terwakili; jika kelas absen dari test set, interpretasi harus ditandai tidak lengkap.
- **Recall kelas:** TP/(TP+FN) untuk masing-masing kategori.
- **Confusion matrix:** mengidentifikasi pasangan label yang sering tertukar.

## Keluaran yang wajib diarsipkan
`split.csv`, `split_distribution.csv`, konfigurasi eksperimen, `history.json`, `val_metrics.json`, `test_metrics.json` setelah final, grafik, serta bobot checkpoint.

## Interpretasi hasil (isi setelah training)
- **Observasi:** [TULIS SETELAH HASIL TERSEDIA]
- **Kesalahan mayoritas:** [ISI BERDASARKAN CONFUSION MATRIX]
- **Dampak class imbalance:** [ISI DARI EDA + PER-CLASS RECALL]
- **Pemilihan model:** [JUSTIFIKASI PERFORMA DAN EFISIENSI]
- **Keterbatasan:** split tunggal, data tunggal, belum validasi eksternal, bukan klaim klinis.

## Perintah
```bash
python scripts/benchmark.py
python scripts/plot_training.py --history outputs/runs/shufflenet_v2_x1_0/history.json
python scripts/visualize_evaluation.py --metrics outputs/runs/shufflenet_v2_x1_0/val_metrics.json
```
