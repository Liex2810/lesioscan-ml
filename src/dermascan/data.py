"""Patient-disjoint splitting and image loading. The metadata must contain filename, patient, label."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T


def split_metadata(frame, patient_col='patient_id', label_col='diagnostic', holdout=0.15, val=0.15, seed=42):
    if not 0 < holdout < 1 or not 0 < val < 1 or holdout + val >= 1:
        raise ValueError('holdout and val must be positive and sum to <1')
    df = frame.dropna(subset=[patient_col, label_col]).copy().reset_index(drop=True)
    if df.empty:
        raise ValueError('No valid rows')
    df[patient_col] = df[patient_col].astype(str)
    first = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    remain_idx, test_idx = next(first.split(df, groups=df[patient_col]))
    remain = df.iloc[remain_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=val / (1 - holdout), random_state=seed + 1)
    train_idx, valid_idx = next(second.split(remain, groups=remain[patient_col]))
    df['split'] = 'test'
    df.loc[remain.iloc[train_idx].index, 'split'] = 'train'
    df.loc[remain.iloc[valid_idx].index, 'split'] = 'val'
    groups = [set(df.loc[df.split == s, patient_col]) for s in ('train', 'val', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise RuntimeError('Patient leakage detected')
    return df


def augmentations(size, training=False):
    if training:
        steps = [T.RandomResizedCrop(size, scale=(0.85, 1.0)), T.RandomHorizontalFlip(),
                 T.RandomRotation(12), T.ColorJitter(brightness=0.12, contrast=0.12)]
    else:
        steps = [T.Resize((size, size))]
    return T.Compose(steps + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class SkinDataset(Dataset):
    def __init__(self, df, image_root, image_col, label_col, labels, image_size, training=False):
        self.rows = df.reset_index(drop=True)
        self.root = Path(image_root)
        self.image_col = image_col
        self.label_col = label_col
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.transform = augmentations(image_size, training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        path = self.root / str(row[self.image_col])
        with Image.open(path) as im:
            img = im.convert('RGB')
        return self.transform(img), self.label_to_id[str(row[self.label_col])]


def make_loader(ds, batch_size, training=False, weighted=False, num_workers=0):
    sampler = None
    if training and weighted:
        labels = ds.rows[ds.label_col].astype(str)
        freq = labels.value_counts()
        weights = torch.tensor([1.0 / freq[x] for x in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=training and sampler is None,
                      sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def validate_metadata(frame, cfg):
    d = cfg['data']
    needed = [d['image_column'], d['patient_column'], d['label_column']]
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')
    frame = frame.dropna(subset=needed).copy()
    frame[d['label_column']] = frame[d['label_column']].astype(str).str.strip()
    unknown = set(frame[d['label_column']]) - set(cfg['labels'])
    if unknown:
        raise ValueError(f'Unknown labels: {unknown}')
    if frame[d['image_column']].duplicated().any():
        raise ValueError('Duplicate image filenames in metadata; inspect before continuing')
    missing_images = [str(x) for x in frame[d['image_column']] if not (Path(d['images']) / str(x)).is_file()]
    if missing_images:
        raise FileNotFoundError(f'{len(missing_images)} images missing; first: {missing_images[:5]}')
    return frame
