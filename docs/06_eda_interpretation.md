# Panduan Interpretasi EDA

Gunakan notebook `notebooks/01_exploratory_data_analysis.ipynb` setelah dataset siap. Notebook dan `scripts/eda.py` menghasilkan bukti eksplorasi. Catat:

1. Distribusi enam kelas dan perbedaan kelas mayoritas/minoritas.
2. Banyaknya pasien unik, jumlah gambar per pasien, dan risiko kebocoran jika menggunakan image-level random split.
3. Resolusi/rasio aspek dan jumlah berkas kosong, rusak, tidak ditemukan, atau nama berulang.
4. Visualisasi contoh gambar per kelas, disertai pengamatan terhadap pencahayaan/fokus/warna tanpa mengasumsikan diagnosis visual.
5. Setelah split: distribusi kelas per split, pasien unik per split, serta pemeriksaan tidak ada overlap patient ID.
6. Rekomendasi preprocessing dan sampling **berdasarkan train set saja**, bukan holdout test.

**Jangan menuliskan angka atau menyatakan kelas paling dominan sebelum menjalankan EDA.**
