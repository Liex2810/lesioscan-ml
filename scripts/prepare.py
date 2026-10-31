import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from dermascan.config import read_config
from dermascan.data import split_metadata, validate_metadata

ap = argparse.ArgumentParser(description='Validate metadata and make patient-disjoint train/val/test split')
ap.add_argument('--config', default='configs/default.yaml')
a = ap.parse_args()
cfg = read_config(a.config)
d = cfg['data']
frame = validate_metadata(pd.read_csv(d['csv']), cfg)
out = split_metadata(frame, d['patient_column'], d['label_column'], d['holdout_fraction'], d['validation_fraction'], cfg['seed'])
for label in cfg['labels']:
    print(label, {s: int(((out['split'] == s) & (out[d['label_column']] == label)).sum()) for s in ('train', 'val', 'test')})
if any(not (out['split'] == s).any() for s in ('train', 'val', 'test')):
    raise RuntimeError('Empty split')
path = Path(d['split']); path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path, index=False)
print('Saved:', path, 'Rows:', len(out))
print('NOTE: Check class coverage. GroupShuffleSplit is not class-stratified.')
