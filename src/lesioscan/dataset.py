"""Dataset + patient-disjoint split validation; images loaded on demand from Google Drive."""
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .preprocessing import CLASSES, LABEL_TO_INDEX
from .data_paths import index_images

REQUIRED = {'img_id', 'patient_id', 'diagnostic', 'split'}
SPLITS = ('train','val','test')


def load_and_validate_split(csv_path, images_root):
    df = pd.read_csv(csv_path, dtype={'img_id': 'string', 'patient_id': 'string', 'diagnostic':'string', 'split':'string'})
    if df.empty: raise ValueError('split.csv kosong')
    if not REQUIRED.issubset(df.columns): raise ValueError(f'Kolom kurang: {REQUIRED - set(df.columns)}')
    if df[list(REQUIRED)].isna().any().any(): raise ValueError('Kolom wajib mengandung null')
    if df.img_id.duplicated().any(): raise ValueError('Duplikasi img_id')
    if set(df.split) != set(SPLITS): raise ValueError(f'Split harus {SPLITS}')
    if set(df.diagnostic) != set(CLASSES): raise ValueError('Keenam kelas harus ada')
    patient_sets = {s: set(df.loc[df.split==s, 'patient_id']) for s in SPLITS}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if patient_sets[a] & patient_sets[b]: raise ValueError(f'Patient leakage: {a} dan {b}')
    distribution = pd.crosstab(df.diagnostic,df.split).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
    if (distribution == 0).any().any(): raise ValueError('Ada kelas tanpa gambar pada salah satu split')
    paths, conflicts = index_images(images_root)
    if conflicts: raise ValueError(f'Nama file gambar ambigu, contoh: {conflicts[:3]}')
    missing = sorted(set(df.img_id.astype(str)) - set(paths))
    if missing: raise FileNotFoundError(f'{len(missing)} gambar tidak ditemukan; contoh: {missing[:5]}')
    return df, paths, distribution


class LesioScanDataset(Dataset):
    def __init__(self, frame, image_paths, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        filename = str(row['img_id'])
        with Image.open(self.image_paths[filename]) as img:
            image = img.convert('RGB')
            image = self.transform(image)
        return image, LABEL_TO_INDEX[str(row['diagnostic'])]


def make_dataloaders(frame, paths, train_transform, eval_transform,
                     batch_size=16, num_workers=0, pin_memory=False):
    if batch_size <= 0: raise ValueError('batch_size harus positif')
    loaders, datasets = {}, {}
    for split in SPLITS:
        part = frame.loc[frame.split == split].copy()
        transform = train_transform if split == 'train' else eval_transform
        dataset = LesioScanDataset(part, paths, transform)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(split == 'train'), num_workers=num_workers,
                            pin_memory=pin_memory)
        loaders[split], datasets[split] = loader, dataset
    return loaders, datasets


def training_class_weights(frame):
    # Bobot hanya dari train (bukan validasi atau test).
    import torch
    train = frame.loc[frame.split == 'train', 'diagnostic'].astype(str)
    counts = Counter(train)
    if any(counts[c] == 0 for c in CLASSES): raise ValueError('Kelas train tidak lengkap')
    total = len(train)
    return torch.tensor([total/(len(CLASSES)*counts[c]) for c in CLASSES], dtype=torch.float32)
