"""CLI cleaning: python scripts/clean.py --metadata ... --images ... --out ..."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.cleaning import clean_dataset

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--images',required=True);p.add_argument('--out',required=True);p.add_argument('--reuse-integrity',action='store_true');a=p.parse_args()
    print(json.dumps(clean_dataset(a.metadata,a.images,a.out,a.reuse_integrity),indent=2,ensure_ascii=False))
