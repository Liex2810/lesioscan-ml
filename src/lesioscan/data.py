"""Patient-disjoint splitting and image loading. The metadata must contain filename, patient, label."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .preprocessing import build_transforms


from .splitting import stratified_patient_split


def split_metadata(frame, patient_col='patient_id', label_col='diagnostic', holdout=0.15,
                   val=0.15, seed=42, n_candidates=150):
    return stratified_patient_split(frame, patient_col=patient_col, label_col=label_col,
                                    holdout=holdout, val=val, seed=seed, n_candidates=n_candidates)


def augmentations(size, training=False):
    """Backward-compatible alias for the preprocessing pipeline."""
    return build_transforms(size, training)


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
