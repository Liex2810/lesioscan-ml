"""LesioScan ShuffleNet V2 two-stage transfer learning; test split is never read for evaluation."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from torch import nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from tqdm.auto import tqdm

from .data_paths import index_images
from .dataset import LesioScanDataset, load_and_validate_split, training_class_weights
from .preprocessing import build_transforms, CLASSES
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(pretrained=True):
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    model = shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def set_trainable(model, stage):
    if stage == 'feature_extraction':
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
    elif stage == 'fine_tuning':
        for p in model.parameters(): p.requires_grad = True
    else:
        raise ValueError(f'Unknown stage: {stage}')


def one_epoch(model, loader, criterion, optimizer, device, stage, epoch, epochs):
    is_training = optimizer is not None
    model.train(is_training)
    # Backbone BatchNorm must not update its running statistics while frozen.
    if is_training and stage == 'feature_extraction':
        model.eval()
        model.fc.train()
    losses, y_true, y_pred = [], [], []
    desc = f'{stage} epoch {epoch}/{epochs} ' + ('train' if is_training else 'val')
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if is_training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            scores = model(images)
            loss = criterion(scores, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        losses.append((loss.item(), labels.shape[0]))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(scores.argmax(1).detach().cpu().tolist())
    if not y_true: raise RuntimeError('Empty dataloader')
    return {
        'loss': sum(l*n for l,n in losses)/sum(n for _,n in losses),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall_by_class': dict(zip(CLASSES, [float(x) for x in recall_score(y_true, y_pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)])),
    }


def save_history(rows, output):
    if not rows: return
    fields = ['stage', 'epoch', 'train_loss', 'val_loss', 'train_macro_f1', 'val_macro_f1', 'val_balanced_accuracy', 'val_accuracy']
    with open(output, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k:r[k] for k in fields} for r in rows])


def run_training(split_csv, images_root, output, epochs_head=5, epochs_finetune=20,
                 lr_head=1e-3, lr_finetune=3e-5, batch_size=16, seed=42,
                 image_size=224, workers=0, patience=6, use_class_weights=True,
                 pretrained=True):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[TRAIN] Device: {device}', flush=True)
    if device.type != 'cuda':
        print('[WARNING] GPU tidak terdeteksi. Training CPU mungkin sangat lambat.', flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, paths, distribution = load_and_validate_split(split_csv, images_root)
    distribution.to_csv(output/'split_distribution.csv')
    train_tf, eval_tf = build_transforms(image_size)
    train_frame = frame.loc[frame['split']=='train'].copy()
    val_frame = frame.loc[frame['split']=='val'].copy()
    train_dataset = LesioScanDataset(train_frame, paths, train_tf)
    val_dataset = LesioScanDataset(val_frame, paths, eval_tf)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin)
    print(f'[TRAIN] Train={len(train_dataset)} Val={len(val_dataset)} | Final test untouched={len(frame.loc[frame["split"]=="test"])}', flush=True)
    model = make_model(pretrained=pretrained).to(device)
    weights = training_class_weights(frame).to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    rows = []
    best_f1 = -1.0
    best_path = output/'best_shufflenetv2.pt'
    for stage, epochs, learning_rate in [('feature_extraction',epochs_head,lr_head), ('fine_tuning',epochs_finetune,lr_finetune)]:
        if epochs <= 0: continue
        set_trainable(model, stage)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
        no_improvement = 0
        for epoch in range(1, epochs + 1):
            train_metrics = one_epoch(model,train_loader,criterion,optimizer,device,stage,epoch,epochs)
            val_metrics = one_epoch(model,val_loader,criterion,None,device,stage,epoch,epochs)
            row = dict(stage=stage, epoch=epoch, train_loss=train_metrics['loss'], val_loss=val_metrics['loss'],
                       train_macro_f1=train_metrics['macro_f1'], val_macro_f1=val_metrics['macro_f1'],
                       val_balanced_accuracy=val_metrics['balanced_accuracy'], val_accuracy=val_metrics['accuracy'])
            rows.append(row)
            save_history(rows, output/'history.csv')
            print(f'[TRAIN] {stage} {epoch}/{epochs} | train_loss={row["train_loss"]:.4f} val_loss={row["val_loss"]:.4f} val_macro_f1={row["val_macro_f1"]:.4f}',flush=True)
            if val_metrics['macro_f1'] > best_f1 + 1e-6:
                best_f1 = float(val_metrics['macro_f1'])
                no_improvement = 0
                torch.save({
                    'model_state_dict':model.state_dict(), 'model':'shufflenet_v2_x1_0',
                    'classes':list(CLASSES), 'image_size':image_size,
                    'stage':stage, 'epoch':epoch,'seed':seed,
                    'val_metrics':val_metrics,'train_metrics':train_metrics,
                }, best_path)
            else:
                no_improvement += 1
            (output/'training_status.json').write_text(json.dumps({'stage':stage,'epoch':epoch,'epochs':epochs,'best_macro_f1':best_f1,'checkpoint':str(best_path)},indent=2),encoding='utf-8')
            if no_improvement >= patience:
                print(f'[TRAIN] Early stopping {stage} after {epoch} epochs',flush=True)
                break
        # Fine-tuning starts from the best stage-one validation checkpoint.
        if stage == 'feature_extraction' and epochs_finetune > 0 and best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
    if not best_path.exists(): raise RuntimeError('No checkpoint produced')
    summary = {'best_val_macro_f1':best_f1,'best_checkpoint':str(best_path),'history':str(output/'history.csv'),
               'total_epochs_run':len(rows),'train_images':len(train_dataset),'val_images':len(val_dataset),
               'device':str(device),'final_test_evaluated':False}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('[TRAIN] DONE',json.dumps(summary,indent=2),flush=True)
    return summary
