"""CLI EDA: python scripts/eda.py --metadata ... --images ... --out ..."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.eda_pipeline import run_eda

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--images',required=True);p.add_argument('--out',required=True);p.add_argument('--full-image-scan',action='store_true');a=p.parse_args()
    print(json.dumps(run_eda(a.metadata,a.images,a.out,a.full_image_scan),indent=2,ensure_ascii=False))
