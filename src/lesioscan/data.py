"""Pembagian berbasis pasien dan pemuatan gambar; metadata memuat nama file, pasien, dan label."""  # Dokumentasi singkat tanggung jawab modul.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import numpy as np  # Memuat numpy untuk fungsi yang digunakan pada file ini.
import pandas as pd  # Memuat pandas untuk fungsi yang digunakan pada file ini.
from PIL import Image, ImageFile  # Memuat PIL untuk fungsi yang digunakan pada file ini.
from sklearn.model_selection import GroupShuffleSplit  # Memuat sklearn.model_selection untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # Memuat torch.utils.data untuk fungsi yang digunakan pada file ini.
from torchvision import transforms as T  # Memuat torchvision untuk fungsi yang digunakan pada file ini.

def split_metadata(frame, patient_col='patient_id', label_col='diagnostic', holdout=0.15, val=0.15, seed=42):  # Membagi data berdasarkan pasien tanpa tumpang tindih train, validation, dan test.
    if not 0 < holdout < 1 or not 0 < val < 1 or holdout + val >= 1:  # Memeriksa kondisi: not 0 < holdout < 1 or not 0 < val < 1 or holdout + val >= 1.
        raise ValueError('holdout and val must be positive and sum to <1')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    df = frame.dropna(subset=[patient_col, label_col]).copy().reset_index(drop=True)  # Menghapus baris tanpa pasien/label, lalu mereset indeks salinan tabel.
    if df.empty:  # Memeriksa kondisi: df.empty.
        raise ValueError('No valid rows')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    df[patient_col] = df[patient_col].astype(str)  # Menyamakan tipe identitas pasien menjadi teks sebelum pengelompokan.
    first = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)  # Menetapkan pembagi kelompok untuk test.
    remain_idx, test_idx = next(first.split(df, groups=df[patient_col]))  # Mengambil indeks kelompok pengembangan dan kelompok test.
    remain = df.iloc[remain_idx].copy()  # Menetapkan data pengembangan setelah test dipisahkan.
    second = GroupShuffleSplit(n_splits=1, test_size=val / (1 - holdout), random_state=seed + 1)  # Menetapkan pembagi kelompok untuk validasi.
    train_idx, valid_idx = next(second.split(remain, groups=remain[patient_col]))  # Membagi kelompok pengembangan menjadi indeks train dan validasi.
    df['split'] = 'test'  # Memberi penanda test sebagai nilai awal seluruh baris.
    df.loc[remain.iloc[train_idx].index, 'split'] = 'train'  # Menandai baris kelompok pasien training.
    df.loc[remain.iloc[valid_idx].index, 'split'] = 'val'  # Menandai baris kelompok pasien validasi.
    groups = [set(df.loc[df.split == s, patient_col]) for s in ('train', 'val', 'test')]  # Menetapkan kelompok pasien setiap subset.
    if any((groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))):  # Memeriksa kondisi: any((groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))).
        raise RuntimeError('Patient leakage detected')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    return df  # Mengembalikan metadata dengan kolom penanda subset.

def augmentations(size, training=False):  # Menyusun transformasi gambar sesuai mode training atau evaluasi.
    if training:  # Memeriksa kondisi: training.
        steps = [T.RandomResizedCrop(size, scale=(0.85, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(12), T.ColorJitter(brightness=0.12, contrast=0.12)]  # Menetapkan urutan transformasi gambar.
    else:  # Menjalankan jalur alternatif saat kondisi sebelumnya tidak terpenuhi.
        steps = [T.Resize((size, size))]  # Menetapkan urutan transformasi gambar.
    return T.Compose(steps + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # Menyusun augmentasi, konversi tensor 0–1, dan normalisasi ImageNet.

class SkinDataset(Dataset):  # Mendefinisikan dataset gambar yang mengikuti antarmuka PyTorch.

    def __init__(self, df, image_root, image_col, label_col, labels, image_size, training=False):  # Menyimpan metadata, pemetaan kelas, dan transformasi dataset.
        self.rows = df.reset_index(drop=True)  # Menyimpan tabel contoh dengan indeks posisi yang berurutan.
        self.root = Path(image_root)  # Menyimpan folder induk gambar sebagai objek Path.
        self.image_col = image_col  # Menyimpan nama kolom yang berisi nama file gambar.
        self.label_col = label_col  # Menyimpan nama kolom label diagnosis.
        self.label_to_id = {label: i for i, label in enumerate(labels)}  # Memetakan setiap nama kelas ke indeks sesuai urutan konfigurasi.
        self.transform = augmentations(image_size, training)  # Menyimpan pipeline transformasi untuk mode dataset ini.

    def __len__(self):  # Mengembalikan jumlah contoh dalam dataset.
        return len(self.rows)  # Mengembalikan banyaknya contoh yang dapat dibaca loader.

    def __getitem__(self, idx):  # Membaca satu gambar dan mengembalikan tensor beserta indeks kelas.
        row = self.rows.iloc[idx]  # Mengambil satu baris metadata sesuai indeks contoh.
        path = self.root / str(row[self.image_col])  # Menggabungkan folder gambar dengan nama file pada metadata.
        with Image.open(path) as im:  # Membuka file gambar dan menutupnya otomatis setelah dibaca.
            img = im.convert('RGB')  # Menetapkan gambar RGB.
        return (self.transform(img), self.label_to_id[str(row[self.label_col])])  # Mengembalikan tensor gambar yang ditransformasi dan indeks kelasnya.

def make_loader(ds, batch_size, training=False, weighted=False, num_workers=0):  # Membuat DataLoader dan weighted sampler opsional untuk training.
    sampler = None  # Menetapkan strategi pengambilan sampel.
    if training and weighted:  # Memeriksa kondisi: training and weighted.
        labels = ds.rows[ds.label_col].astype(str)  # Menetapkan label kelas.
        freq = labels.value_counts()  # Menetapkan frekuensi setiap kelas.
        weights = torch.tensor([1.0 / freq[x] for x in labels], dtype=torch.double)  # Menetapkan bobot sampel.
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)  # Menetapkan strategi pengambilan sampel.
    return DataLoader(ds, batch_size=batch_size, shuffle=training and sampler is None, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())  # Membentuk loader batch; shuffle hanya aktif bila training tanpa sampler.

def validate_metadata(frame, cfg):  # Memeriksa kolom wajib, label, duplikasi nama, dan keberadaan gambar.
    d = cfg['data']  # Menetapkan pengaturan data.
    needed = [d['image_column'], d['patient_column'], d['label_column']]  # Menetapkan kolom wajib.
    missing = set(needed) - set(frame.columns)  # Menetapkan kolom yang belum tersedia.
    if missing:  # Memeriksa kondisi: missing.
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    frame = frame.dropna(subset=needed).copy()  # Membuang baris yang kolom wajibnya kosong sebelum validasi lanjutan.
    frame[d['label_column']] = frame[d['label_column']].astype(str).str.strip()  # Membersihkan spasi pada label dan mengubahnya menjadi teks.
    unknown = set(frame[d['label_column']]) - set(cfg['labels'])  # Menetapkan label yang tidak dikenal.
    if unknown:  # Memeriksa kondisi: unknown.
        raise ValueError(f'Unknown labels: {unknown}')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    if frame[d['image_column']].duplicated().any():  # Memeriksa kondisi: frame[d['image_column']].duplicated().any().
        raise ValueError('Duplicate image filenames in metadata; inspect before continuing')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    missing_images = [str(x) for x in frame[d['image_column']] if not (Path(d['images']) / str(x)).is_file()]  # Menetapkan gambar yang tidak ditemukan.
    if missing_images:  # Memeriksa kondisi: missing_images.
        raise FileNotFoundError(f'{len(missing_images)} images missing; first: {missing_images[:5]}')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    return frame  # Mengembalikan metadata yang lolos pemeriksaan.
