"""CLI entry point for LesioScan validation-only evaluation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.evaluation import run_evaluation

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--split',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--training-out',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--trust-own-checkpoint',action='store_true',help='Allow one-time conversion of your OWN trusted legacy checkpoint')
    a=p.parse_args()
    run_evaluation(a.split,a.images,a.training_out,a.out,batch_size=a.batch_size,workers=a.workers,
                   trusted_checkpoint=a.trust_own_checkpoint)
