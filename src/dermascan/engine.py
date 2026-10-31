import json
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
from .metrics import evaluate


def run_epoch(model, loader, dev, criterion, optimizer=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total, n, true, pred = 0., 0, [], []
    scaler_enabled = bool(use_amp and dev.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(dev), y.to(dev)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=dev.type, enabled=scaler_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
        total += float(loss.item()) * len(y)
        n += len(y)
        true.extend(y.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    return total / max(n, 1), true, pred


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
