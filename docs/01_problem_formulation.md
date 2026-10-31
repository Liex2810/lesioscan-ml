# Perumusan Masalah LesioScan ML

## Masalah dan konteks
LesioScan meneliti klasifikasi **enam kategori citra lesi kulit** pada PAD-UFES-20. Tujuannya adalah membandingkan kinerja model komputer visi pada *data penelitian*, bukan memastikan diagnosis seseorang. Dataset berasal dari distribusi pengambilan gambar dan label tertentu, sehingga performa di foto pengguna umum belum diketahui.

## Rumusan masalah penelitian
1. Seberapa baik model pretrained ShuffleNet V2, DenseNet-121, dan ViT-Small membedakan enam kelas pada PAD-UFES-20 menggunakan pembagian yang tidak mencampurkan pasien?
2. Apa kompromi antara Macro-F1/per-class recall dengan ukuran checkpoint dan latensi inferensi?
3. Kelas mana yang paling sering tertukar, dan bagaimana distribusi kelas memengaruhi interpretasi hasil?

## Formulasi ML
- **Masukan**: citra RGB lesi kulit yang telah diproses sesuai normalisasi ImageNet.
- **Keluaran**: enam logits yang sesuai urutan kelas `ACK, BCC, MEL, NEV, SCC, SEK`.
- **Jenis tugas**: *supervised multiclass image classification*.
- **Arsitektur**: dua CNN (ShuffleNet V2 dan DenseNet-121) serta satu Vision Transformer (ViT-Small).
- **Pendekatan**: transfer learning dua tahap (latih head, lalu fine-tune seluruh parameter).
- **Pemilihan checkpoint**: Macro-F1 validasi; selalu inspeksi recall per kelas.
- **Evaluasi akhir**: holdout pasien yang tidak digunakan saat memilih model.

## Hipotesis awal (belum dibuktikan)
Model ringan seperti ShuffleNet V2 berpotensi lebih efisien untuk perangkat mobile. Model lebih besar mungkin berbeda dalam kemampuan representasi, tetapi **tidak dapat diasumsikan lebih akurat** tanpa eksperimen adil.

## Pencegahan bias dan kebocoran data
1. Pisahkan menurut `patient_id`, bukan gambar acak.
2. Gunakan split yang sama untuk seluruh model, jangan pilih model berdasarkan test set.
3. Augmentasi hanya pada training; pelajari keputusan penyesuaian data hanya dari training set.
4. Laporkan ketidakseimbangan kelas, variasi antarpasien, serta seluruh keterbatasan data.

## Kriteria keberhasilan teknis
- Training tiga arsitektur dapat direproduksi dengan seed, konfigurasi, dan split tersimpan.
- Semua kelas terwakili dalam train/validation/test jika memungkinkan.
- Macro-F1, balanced accuracy, confusion matrix, dan recall per kelas tersedia.
- Model terpilih dapat diekspor ke ONNX dengan pemeriksaan kesetaraan numerik.

## Batasan dan etika
Prediksi bukan diagnosis medis, output softmax bukan probabilitas penyakit, dan hanya enam kelas yang dikenali. Kualitas foto dan variasi populasi dapat mengurangi performa. Jangan membuat klaim klinis tanpa validasi eksternal dan penilaian medis.
