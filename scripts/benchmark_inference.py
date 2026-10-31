"""Estimate per-image latency after warmup; run on TARGET device for deployment claims."""
import argparse,time,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from lesioscan.inference import load_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--runs',type=int,default=50)
    a=p.parse_args()
    if a.runs<1:raise ValueError('runs must be positive')
    model,cfg=load_checkpoint(a.checkpoint,a.device)
    x=torch.randn(1,3,cfg['image_size'],cfg['image_size'],device=a.device)
    def synchronize():
        if a.device=='cuda':torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(10):model(x)
        synchronize();times=[]
        for _ in range(a.runs):
            synchronize();start=time.perf_counter();model(x);synchronize()
            times.append((time.perf_counter()-start)*1000)
    print({'device':a.device,'batch_size':1,'mean_ms':round(statistics.mean(times),3),
           'median_ms':round(statistics.median(times),3),'runs':a.runs,
           'note':'Synthetic tensor; excludes preprocessing and camera overhead'})
if __name__=='__main__':main()
