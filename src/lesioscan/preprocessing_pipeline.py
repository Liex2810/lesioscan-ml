"""Preprocessing dry run and data audit; does not copy or modify original images."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from .dataset import load_and_validate_split, make_dataloaders, training_class_weights
from .preprocessing import CLASSES, LABEL_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD, build_transforms, denormalize


def run_preprocessing(split_csv, images_root, output, image_size=224, batch_size=16,
                      num_workers=0, sample_count=8, full_scan=False):
    output = Path(output)
    figures, tables = output/'figures', output/'tables'
    figures.mkdir(parents=True,exist_ok=True)
    tables.mkdir(parents=True,exist_ok=True)
    print('[PREPROCESS 0%] Memulai', flush=True)
    df, paths, dist = load_and_validate_split(split_csv, images_root)
    print(f'[PREPROCESS 25%] Split dan {len(paths)} file gambar tervalidasi', flush=True)
    dist.to_csv(tables/'split_class_distribution.csv')
    df.groupby('split').patient_id.nunique().reindex(['train','val','test']).rename('unique_patients').to_csv(tables/'unique_patients_per_split.csv')
    train_transform, eval_transform = build_transforms(image_size)
    loaders, datasets = make_dataloaders(df,paths,train_transform,eval_transform,batch_size,num_workers)
    weights=training_class_weights(df)
    pd.DataFrame({'class':CLASSES,'index':list(range(len(CLASSES))), 'train_count':dist['train'].tolist(), 'weight':weights.tolist()}).to_csv(tables/'label_mapping_and_weights.csv',index=False)
    print('[PREPROCESS 50%] Transformasi, DataLoader dan class weights siap', flush=True)
    # Memeriksa beberapa batch tanpa membaca seluruh dataset; sampling hanya dari TRAIN.
    x,y = next(iter(loaders['train']))
    if x.ndim!=4 or x.shape[1:]!=(3,image_size,image_size):
        raise AssertionError(f'Bentuk tensor tidak valid: {tuple(x.shape)}')
    if not torch.isfinite(x).all() or (y<0).any() or (y>=len(CLASSES)).any():
        raise AssertionError('Tensor tidak finite atau label di luar 0-5')
    print(f'[PREPROCESS 65%] Smoke test batch OK: {tuple(x.shape)}',flush=True)
    sample_count=min(sample_count,len(datasets['train']),12)
    train_frame = datasets['train'].frame
    selected = train_frame.sample(n=sample_count,random_state=42).index.tolist()
    fig, axes = plt.subplots(sample_count,3,figsize=(10,3*sample_count),squeeze=False)
    for i, idx in enumerate(selected):
        filename=str(train_frame.iloc[idx].img_id)
        with Image.open(paths[filename]) as im: original=im.convert('RGB')
        a=train_transform(original); b=eval_transform(original)
        for ax, pic, title in zip(axes[i], [original,denormalize(a).permute(1,2,0),denormalize(b).permute(1,2,0)],
                                  ['Original', 'Train augmentation', 'Evaluation']):
            ax.imshow(pic);ax.axis('off');ax.set_title(f'{title} — {train_frame.iloc[idx].diagnostic}')
    fig.suptitle('LesioScan — Preprocessing Preview (Train Only)',y=1.002)
    fig.tight_layout();fig.savefig(figures/'preprocessing_preview.png',dpi=140,bbox_inches='tight');plt.close(fig)
    print('[PREPROCESS 85%] Preview training selesai',flush=True)
    scan_rows=[]
    if full_scan:
        for filename in tqdm(df.img_id.astype(str),desc='Validasi transform seluruh gambar',unit='gambar'):
            with Image.open(paths[filename]) as im:
                t=eval_transform(im.convert('RGB'))
            scan_rows.append({'img_id':filename,'valid':bool(torch.isfinite(t).all()),'shape':str(tuple(t.shape))})
        pd.DataFrame(scan_rows).to_csv(tables/'full_transform_scan.csv',index=False)
        if not all(row['valid'] for row in scan_rows):raise AssertionError('Full scan menemukan tensor tidak valid')
    summary={'image_size':image_size,'batch_size':batch_size,'num_workers':num_workers,
             'train_images':len(datasets['train']),'val_images':len(datasets['val']),'test_images':len(datasets['test']),
             'classes':list(CLASSES),'label_mapping':LABEL_TO_INDEX,
             'imagenet_mean':list(IMAGENET_MEAN),'imagenet_std':list(IMAGENET_STD),
             'train_augmentation':True,'val_test_augmentation':False,
             'training_class_weights':weights.tolist(),'full_scan':bool(full_scan),
             'batch_shape':list(x.shape),'status':'passed'}
    (output/'preprocessing_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('[PREPROCESS 100%] Audit dan laporan tersimpan',flush=True)
    return summary
