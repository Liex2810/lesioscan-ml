"""Clean metadata safely, generating an audit trail; never edits original files.

Run after EDA and before prepare.py:
python scripts/clean.py
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from lesioscan.config import read_config
from lesioscan.cleaning import clean_metadata


def main():
    p = argparse.ArgumentParser(description='Pembersihan metadata dan validasi berkas gambar LesioScan')
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--out', default='outputs/cleaning')
    args = p.parse_args()
    cfg = read_config(args.config)
    d = cfg['data']
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(d.get('raw_csv', 'data/metadata.csv'))
    clean_path = Path(d['csv'])
    if raw_path.resolve() == clean_path.resolve():
        raise ValueError('raw_csv dan csv tidak boleh sama agar data asli tidak tertimpa.')
    frame = pd.read_csv(raw_path, dtype={d['image_column']: 'string', d['patient_column']: 'string'})
    cleaned, issues, stats = clean_metadata(frame, d['images'], cfg['labels'],
                    d['image_column'], d['patient_column'], d['label_column'])
    issues.to_csv(output_dir / 'cleaning_issues.csv', index=False)
    (output_dir / 'cleaning_summary.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    # Block invalid rows, but never infer corrected medical labels or patient IDs.
    if cleaned.empty or any(label not in set(cleaned[d['label_column']]) for label in cfg['labels']):
        raise ValueError('Hasil cleaning tidak memiliki semua kelas; periksa laporan sebelum split.')
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print('Metadata bersih:', clean_path)
    print('Audit:', output_dir / 'cleaning_issues.csv')
    print('WAJIB: tinjau baris yang dikarantina, terutama konflik label/pasien; jangan mengubah diagnosis otomatis.')


if __name__ == '__main__':
    main()
