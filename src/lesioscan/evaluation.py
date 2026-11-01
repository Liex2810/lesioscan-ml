"""Validation-only inference and reporting for LesioScan ShuffleNet V2."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from .dataset import LesioScanDataset, load_and_validate_split
from .preprocessing import build_transforms, CLASSES
from .training import make_model


def _checkpoint(source, eval_path):
    """Prefer an ordinary weights-only checkpoint. Convert trusted legacy checkpoints once."""
    if eval_path.exists():
        return torch.load(eval_path, map_location='cpu', weights_only=True)
    if not source.is_file():
        raise FileNotFoundError(f'Best fine-tuning checkpoint not found: {source}')
    # Legacy LesioScan checkpoints include NumPy RNG state. Unrestricted pickle loading
    # is permitted ONLY for locally produced, trusted checkpoints.
    original = torch.load(source, map_location='cpu', weights_only=False)
    required = ('model', 'stage', 'epoch', 'classes', 'image_size', 'model_state_dict')
    if not all(k in original for k in required):
        raise ValueError('Checkpoint does not contain required keys')
    compact = {k:original[k] for k in required}
    compact['epoch'] = int(compact['epoch'])
    compact['image_size'] = int(compact['image_size'])
    compact['classes'] = list(compact['classes'])
    if compact['model'] != 'shufflenet_v2_x1_0' or compact['classes'] != list(CLASSES):
        raise ValueError('Checkpoint model/classes mismatch')
    torch.save(compact, eval_path)
    return torch.load(eval_path, map_location='cpu', weights_only=True)


def run_evaluation(split_csv, images_root, training_out, output, batch_size=16, workers=0,
                   trusted_checkpoint=False):
    training_out, output = Path(training_out), Path(output)
    tables, figures = output/'tables', output/'figures'
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    original = training_out/'fine_tuning'/'best_finetuning.pt'
    compact_path = training_out/'fine_tuning'/'best_finetuning_eval.pt'
    if not compact_path.exists() and not trusted_checkpoint:
        raise PermissionError('Legacy checkpoint requires --trust-own-checkpoint for one-time conversion. Only enable for YOUR OWN trusted training file.')
    print('[EVAL 5%] Reading checkpoint',flush=True)
    ckpt = _checkpoint(original,compact_path)
    if ckpt['model'] != 'shufflenet_v2_x1_0' or list(ckpt['classes']) != list(CLASSES):
        raise ValueError('Unexpected architecture or class order')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model=model.to(device).eval()
    print(f'[EVAL 15%] Model ready on {device}',flush=True)
    frame, paths, distribution=load_and_validate_split(split_csv, images_root)
    val=frame.loc[frame['split']=='val'].copy().reset_index(drop=True)
    _, transform=build_transforms(int(ckpt['image_size']))
    loader=DataLoader(LesioScanDataset(val,paths,transform),batch_size=batch_size,shuffle=False,
                      num_workers=workers,pin_memory=(device.type=='cuda'))
    print(f'[EVAL 25%] Validation images: {len(val)} (test untouched)',flush=True)
    yt, yp, prob=[], [], []
    with torch.inference_mode():
        for x,y in tqdm(loader,desc='Validation inference',unit='batch'):
            logits=model(x.to(device,non_blocking=True))
            p=logits.softmax(dim=1).cpu()
            yt.extend(y.tolist())
            yp.extend(p.argmax(dim=1).tolist())
            prob.extend(p.tolist())
    yt=np.asarray(yt,dtype=int); yp=np.asarray(yp,dtype=int); prob=np.asarray(prob)
    if len(yt)!=len(val):raise RuntimeError('Prediction count mismatch')
    print('[EVAL 65%] Predictions completed',flush=True)
    results=val[['img_id','diagnostic']].copy()
    results['true_index']=yt;results['pred_index']=yp
    results['predicted_class']=[CLASSES[i] for i in yp]
    results['confidence']=prob.max(axis=1)
    results['correct']=yt==yp
    for i,label in enumerate(CLASSES):results[f'prob_{label}']=prob[:,i]
    results.to_csv(tables/'validation_predictions.csv',index=False)
    labels=list(range(len(CLASSES)))
    report=pd.DataFrame(classification_report(yt,yp,labels=labels,target_names=list(CLASSES),output_dict=True,zero_division=0)).T
    report.to_csv(tables/'classification_report.csv')
    prec,rec,f1,support=precision_recall_fscore_support(yt,yp,labels=labels,zero_division=0)
    cls=pd.DataFrame({'class':CLASSES,'precision':prec,'recall':rec,'f1_score':f1,'support':support})
    cls.to_csv(tables/'per_class_metrics.csv',index=False)
    cm=confusion_matrix(yt,yp,labels=labels)
    cmnorm=confusion_matrix(yt,yp,labels=labels,normalize='true')
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(tables/'confusion_matrix.csv')
    pd.DataFrame(cmnorm,index=CLASSES,columns=CLASSES).to_csv(tables/'confusion_matrix_normalized.csv')
    for name,m,fmt in [('confusion_matrix',cm,'d'),('confusion_matrix_normalized',cmnorm,'.2f')]:
        fig,ax=plt.subplots(figsize=(9,7))
        ConfusionMatrixDisplay(confusion_matrix=m,display_labels=list(CLASSES)).plot(ax=ax,cmap='Blues',values_format=fmt,colorbar=True)
        ax.set_title(f'LesioScan — Validation {name.replace("_"," ").title()}')
        fig.savefig(figures/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(CLASSES,100*rec);ax.set(ylim=(0,105),ylabel='Recall (%)',title='Validation Recall per Class')
    for i,v in enumerate(rec):ax.text(i,100*v+1,f'{100*v:.1f}%',ha='center')
    fig.savefig(figures/'recall_by_class.png',dpi=160,bbox_inches='tight');plt.close(fig)
    errors=results.loc[~results.correct].copy()
    errors.groupby(['diagnostic','predicted_class']).size().reset_index(name='count').sort_values('count',ascending=False).to_csv(tables/'misclassification_pairs.csv',index=False)
    errors.sort_values('confidence',ascending=False).to_csv(tables/'high_confidence_errors.csv',index=False)
    fig,axes=plt.subplots(2,6,figsize=(18,7))
    for ax in axes.flat:ax.axis('off')
    for i,part in enumerate([results.loc[results.correct],errors]):
        samples=part.sample(n=min(6,len(part)),random_state=42)
        for j,(_,row) in enumerate(samples.iterrows()):
            ax=axes[i,j]
            with Image.open(paths[str(row.img_id)]) as im:ax.imshow(im.convert('RGB'))
            ax.set_title(f'True: {row.diagnostic} | Pred: {row.predicted_class}\nScore: {row.confidence:.2f}',fontsize=9)
            ax.axis('off')
    fig.suptitle('Validation: correct (top) and incorrect (bottom)')
    fig.savefig(figures/'prediction_samples.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('[EVAL 90%] Reports and charts saved',flush=True)
    summary={'model':'shufflenet_v2_x1_0','checkpoint':str(compact_path),'best_epoch':int(ckpt['epoch']),
             'evaluation_split':'val','validation_images':int(len(val)),
             'accuracy':float(accuracy_score(yt,yp)),
             'balanced_accuracy':float(balanced_accuracy_score(yt,yp)),
             'macro_f1':float(f1_score(yt,yp,labels=labels,average='macro',zero_division=0)),
             'test_set_evaluated':False,'classes':list(CLASSES)}
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('[EVAL 100%] DONE '+json.dumps(summary),flush=True)
    return summary
