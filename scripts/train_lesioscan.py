"""Entry point for the LesioScan two-stage ShuffleNet V2 baseline."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.training import run_training

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs-head',type=int,default=5)
    p.add_argument('--epochs-finetune',type=int,default=20)
    p.add_argument('--lr-head',type=float,default=.001)
    p.add_argument('--lr-finetune',type=float,default=.00003)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--no-class-weights',action='store_true')
    p.add_argument('--no-pretrained',action='store_true',help='For debugging only; not recommended for real training')
    a=p.parse_args()
    run_training(a.split,a.images,a.out,epochs_head=a.epochs_head,epochs_finetune=a.epochs_finetune,
                 lr_head=a.lr_head,lr_finetune=a.lr_finetune,batch_size=a.batch_size,seed=a.seed,
                 image_size=a.image_size,workers=a.workers,patience=a.patience,
                 use_class_weights=not a.no_class_weights,pretrained=not a.no_pretrained)
