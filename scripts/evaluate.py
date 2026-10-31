import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
import torch
from dermascan.config import read_config, device
from dermascan.data import SkinDataset, make_loader
from dermascan.models import build
from dermascan.engine import run_epoch, save_json
from dermascan.metrics import evaluate
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args()
cfg=read_config(a.config); d=cfg['data']; ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
df=pd.read_csv(d['split']); df=df[df.split=='test']
if set(df[d['label_column']]) != set(cfg['labels']):
    print('WARNING: Some test classes are missing; interpret metrics carefully')
ds=SkinDataset(df,d['images'],d['image_column'],d['label_column'],ckpt['labels'],ckpt['image_size'])
loader=make_loader(ds,cfg['train']['batch_size'],num_workers=cfg['train']['num_workers'])
m=build(ckpt['model'],len(ckpt['labels']),pretrained=False); m.load_state_dict(ckpt['state_dict']); dev=device(); m.to(dev)
_,true,pred=run_epoch(m,loader,dev,torch.nn.CrossEntropyLoss())
metrics=evaluate(true,pred,ckpt['labels']); save_json(Path(a.checkpoint).parent/'test_metrics.json',metrics)
print('TEST macro-F1:',metrics['macro_f1'],'\nPer-class recalls:',{k:metrics['per_class'][k]['recall'] for k in ckpt['labels']})
print('IMPORTANT: Do not repeatedly tune based on this test result.')
