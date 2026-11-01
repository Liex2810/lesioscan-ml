"""CLI patient-level split: python scripts/prepare.py --metadata ... --out ..."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.splitting import make_split

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--candidates',type=int,default=500);a=p.parse_args()
    print(json.dumps(make_split(a.metadata,a.out,a.seed,a.candidates),indent=2,ensure_ascii=False))
