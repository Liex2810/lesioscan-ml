import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from lesioscan.config import read_config
from lesioscan.data import split_metadata, validate_metadata

ap = argparse.ArgumentParser(description='Validate metadata and make patient-disjoint train/val/test split')
ap.add_argument('--config', default='configs/default.yaml')
a = ap.parse_args()
cfg = read_config(a.config)
d = cfg['data']
clean_path = Path(d['csv'])
if not clean_path.is_file():
    raise FileNotFoundError(f'Jalankan scripts/clean.py terlebih dahulu: {clean_path}')
frame = validate_metadata(pd.read_csv(clean_path), cfg)
out = split_metadata(frame, d['patient_column'], d['label_column'], d['holdout_fraction'], d['validation_fraction'], cfg['seed'], n_candidates=d.get('split_candidates', 150))
for label in cfg['labels']:
    print(label, {s: int(((out['split'] == s) & (out[d['label_column']] == label)).sum()) for s in ('train', 'val', 'test')})
if any(not (out['split'] == s).any() for s in ('train', 'val', 'test')):
    raise RuntimeError('Empty split')
path = Path(d['split']); path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)
print('Saved:', path, 'Rows:', len(out))
print('Selected class-aware candidate with patient-disjoint groups; check count table above.')

summary = out.groupby(['split', d['label_column']]).size().unstack(fill_value=0)
summary.to_csv(path.parent / 'split_distribution.csv')
print('Saved:', path.parent / 'split_distribution.csv')
