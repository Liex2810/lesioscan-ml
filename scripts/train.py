import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
import torch
from lesioscan.config import read_config, seed_everything, device
from lesioscan.data import SkinDataset, make_loader
from lesioscan.models import build, ALLOWED
from lesioscan.feature_extraction import configure_feature_extraction
from lesioscan.fine_tuning import configure_fine_tuning
from lesioscan.metrics import evaluate
from lesioscan.engine import run_epoch, save_json

p = argparse.ArgumentParser(description='Two-stage transfer learning on Google Colab')
p.add_argument('--config', default='configs/default.yaml')
p.add_argument('--model', default=None, choices=ALLOWED)
p.add_argument('--run', default=None)
a = p.parse_args()
cfg = read_config(a.config); seed_everything(cfg['seed'])
d = cfg['data']; tr = cfg['train']; name = a.model or cfg['model']['name']
frame = pd.read_csv(d['split'])
if not {'train', 'val'}.issubset(set(frame['split'])):
    raise ValueError('Run scripts/prepare.py first')
for s in ('train', 'val'):
    present = set(frame.loc[frame.split == s, d['label_column']])
    if present != set(cfg['labels']):
        raise ValueError(f'{s} is missing classes: {set(cfg["labels"]) - present}; use another patient split')
loaders = {}
for split in ('train', 'val'):
    ds = SkinDataset(frame[frame.split == split], d['images'], d['image_column'], d['label_column'], cfg['labels'], d['image_size'], training=split == 'train')
    loaders[split] = make_loader(ds, tr['batch_size'], training=split == 'train', weighted=split == 'train' and tr['sampler'] == 'weighted', num_workers=tr['num_workers'])
dev = device(); print('Device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
m = build(name, len(cfg['labels']), tr['pretrained']).to(dev)
criterion = torch.nn.CrossEntropyLoss()
run = a.run or name
outdir = Path(cfg['outputs']) / 'runs' / run; outdir.mkdir(parents=True, exist_ok=True)
best_score, bad, history = -1., 0, []
# Stage 1 trains the head, stage 2 unfreezes the full backbone.
for stage, epochs, lr in [(1, tr['stage1_epochs'], tr['head_lr']), (2, tr['stage2_epochs'], tr['finetune_lr'])]:
    if stage == 1:
        configure_feature_extraction(m, name)
    else:
        configure_fine_tuning(m)
    opt = torch.optim.AdamW((x for x in m.parameters() if x.requires_grad), lr=lr, weight_decay=tr['weight_decay'])
    bad = 0
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(m, loaders['train'], dev, criterion, opt, tr['amp'])
        val_loss, y, pred = run_epoch(m, loaders['val'], dev, criterion)
        metrics = evaluate(y, pred, cfg['labels'])
        row = {'stage': stage, 'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'macro_f1': metrics['macro_f1'], 'balanced_accuracy': metrics['balanced_accuracy']}
        history.append(row)
        print(row)
        if metrics['macro_f1'] > best_score:
            best_score = metrics['macro_f1']; bad = 0
            torch.save({'model': name, 'state_dict': m.state_dict(), 'labels': cfg['labels'], 'image_size': d['image_size'], 'config': cfg}, outdir/'best.pt')
            save_json(outdir/'val_metrics.json', metrics)
        else:
            bad += 1
            if bad >= tr['patience']:
                print('Early stop stage:', stage)
                break
save_json(outdir/'history.json', history)
print('Best validation macro-F1:', best_score, 'Checkpoint:', outdir/'best.pt')
