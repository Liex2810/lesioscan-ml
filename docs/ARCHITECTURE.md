# Arsitektur Kode LesioScan

## Struktur dan tanggung jawab

| File | Fungsi |
| --- | --- |
| `README.md` | Pengantar dan petunjuk menjalankan proyek |
| `requirements.txt` | Library dan rentang versi yang dibutuhkan |
| `.gitignore` | Pola file lokal yang tidak dilacak Git |
| `configs/default.yaml` | Lokasi data, kelas, model, dan hyperparameter |
| `data/README.md` | Susunan file dataset pada runtime |
| `docs/WORKFLOW.md` | Urutan persiapan, training, evaluasi, dan ekspor |
| `scripts/prepare.py` | Memvalidasi metadata dan menyiapkan split pasien |
| `scripts/train.py` | Mengatur dua tahap transfer learning dan checkpoint |
| `scripts/evaluate.py` | Menguji checkpoint pada holdout test |
| `scripts/export_onnx.py` | Mengekspor checkpoint dan memeriksa paritas numerik |
| `src/lesioscan/__init__.py` | Penanda package internal |
| `src/lesioscan/config.py` | Pembacaan YAML, seed, device, dan direktori |
| `src/lesioscan/data.py` | Pembagian pasien, transformasi, dataset, dan loader |
| `src/lesioscan/models.py` | Pembuatan model dan pembekuan backbone |
| `src/lesioscan/engine.py` | Forward/backward satu epoch dan penyimpanan JSON |
| `src/lesioscan/metrics.py` | Accuracy, macro-F1, balanced accuracy, dan laporan kelas |
| `tests/test_splits.py` | Uji sintetis agar kelompok pasien tidak tumpang tindih |

## Hubungan modul

Pengguna menjalankan script di `scripts/`. Script memanggil fungsi di `src/lesioscan/` dan membaca konfigurasi. File modul tidak perlu dijalankan satu per satu.

- `prepare.py` memanggil `config.py` dan `data.py` untuk menghasilkan `data/split.csv`.
- `train.py` memanggil konfigurasi, dataset/loader, model, engine, dan metrik untuk menghasilkan checkpoint serta hasil validasi.
- `evaluate.py` membaca checkpoint dan data test, lalu memakai model, loader, engine, serta metrik.
- `export_onnx.py` membangun model dari checkpoint dan membandingkan hasil PyTorch dengan ONNX Runtime pada input sintetis.

## Membaca komentar per baris

Setiap baris instruksi Python memiliki komentar bahasa Indonesia di sebelah kanan. Baris kosong memisahkan kelompok pekerjaan. Nama fungsi dan pesan program dipertahankan bila tidak perlu diubah agar contoh pemakaian tetap cocok.

Contoh: `from lesioscan.models import build` mengimpor fungsi pembuat model; perintah tersebut belum melatih model. `build(..., pretrained=False)` membangun model dengan bobot acak untuk pemeriksaan teknis. `train.py` memakai konfigurasi pretrained untuk transfer learning yang sebenarnya.

## GitHub, Colab, dan Drive

GitHub menyimpan versi kode. Colab menjalankan salinan pada `/content/lesioscan-ml`. Google Drive menyimpan data dan cadangan hasil di luar disk runtime sementara. Membuat folder Drive tidak otomatis memindahkan hasil script; arahkan konfigurasi atau salin hasil secara eksplisit.

Urutan belajar: README → konfigurasi → prepare/data → models → train/engine/metrics → evaluate → export.
