"""python scripts/preprocess.py --split CSV --images DIR --out DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lesioscan.preprocessing_pipeline import run_preprocessing

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--sample-count',type=int,default=8)
    p.add_argument('--full-scan', action='store_true')
    a=p.parse_args()
    print(json.dumps(run_preprocessing(a.split,a.images,a.out,a.image_size,a.batch_size,a.num_workers,a.sample_count,a.full_scan),indent=2),flush=True)
