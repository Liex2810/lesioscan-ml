# Penempatan Dataset

Simpan data asli di Google Drive, kemudian siapkan salinan kerja pada runtime Colab:

```text
data/
├── README.md
├── metadata.csv
└── images/
    └── <nama_file_gambar>
```

`metadata.csv` harus memuat `img_id`, `patient_id`, dan `diagnostic`. Nilai `img_id` harus cocok dengan nama file gambar termasuk ekstensi. Label yang digunakan: ACK, BCC, MEL, NEV, SCC, SEK.

`split.csv` baru dihasilkan setelah menjalankan `scripts/prepare.py`. Pembagian dilakukan berdasarkan pasien; periksa keterwakilan kelas setelah pembagian.

Jangan unggah gambar, metadata pasien, atau arsip dataset ke GitHub publik. `.gitignore` bukan pengganti pemeriksaan `git status`.
