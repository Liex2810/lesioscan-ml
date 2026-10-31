"""Exploratory Data Analysis of PAD-UFES-20 metadata and local images.
Usage: python scripts/eda.py --out outputs/eda
This does NOT touch the held-out test set or run model selection.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError
from lesioscan.config import read_config

def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis (EDA) PAD-UFES-20')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--out', default='outputs/eda')
    args = parser.parse_args()
    cfg = read_config(args.config)
    d = cfg['data']
    root = Path(d['images'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(d.get('raw_csv', d['csv']))
    image_col, patient_col, label_col = d['image_column'], d['patient_column'], d['label_column']
    required = {image_col, patient_col, label_col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')

    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_values.rename_axis('column').reset_index(name='missing_count').to_csv(out / 'missing_values.csv', index=False)
    counts = df[label_col].fillna('MISSING').astype(str).value_counts()
    counts.rename_axis('class').reset_index(name='images').to_csv(out / 'class_distribution.csv', index=False)
    patient_counts = df.groupby(patient_col, dropna=True).size()
    image_files_missing, unreadable = [], []
    widths, heights = [], []
    for item in df[image_col].dropna().astype(str).unique():
        path = root / item
        if not path.is_file():
            image_files_missing.append(item)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                widths.append(im.width)
                heights.append(im.height)
        except (OSError, ValueError, UnidentifiedImageError):
            unreadable.append(item)
    pd.DataFrame({'img_id': image_files_missing, 'issue': 'missing'}).to_csv(out / 'missing_images.csv', index=False)
    pd.DataFrame({'img_id': unreadable, 'issue': 'unreadable'}).to_csv(out / 'unreadable_images.csv', index=False)
    pd.DataFrame({'width': widths, 'height': heights}).to_csv(out / 'image_dimensions.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.reindex(cfg['labels'], fill_value=0).plot.bar(ax=ax)
    ax.set(xlabel='Kelas', ylabel='Jumlah gambar', title='Distribusi kelas PAD-UFES-20')
    fig.tight_layout(); fig.savefig(out / 'class_distribution.png', dpi=160); plt.close(fig)
    if not patient_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        patient_counts.hist(bins=min(30, max(1, patient_counts.nunique())), ax=ax)
        ax.set(xlabel='Gambar per pasien', ylabel='Jumlah pasien', title='Distribusi gambar per pasien')
        fig.tight_layout(); fig.savefig(out / 'patient_distribution.png', dpi=160); plt.close(fig)
    if widths:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(widths, heights, s=8, alpha=.5)
        ax.set(xlabel='Lebar (px)', ylabel='Tinggi (px)', title='Resolusi gambar')
        fig.tight_layout(); fig.savefig(out / 'image_resolution.png', dpi=160); plt.close(fig)
    summary = {
        'total_rows': len(df),
        'unique_images': int(df[image_col].nunique()),
        'unique_patients': int(df[patient_col].nunique()),
        'duplicate_image_ids': int(df[image_col].duplicated().sum()),
        'missing_files': len(image_files_missing),
        'unreadable_files': len(unreadable),
        'class_counts': counts.to_dict(),
    }
    (out / 'eda_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('EDA output:', out.resolve())
    print('IMPORTANT: fit preprocessing statistics and choose training strategies ONLY on train set.')

if __name__ == '__main__':
    main()
