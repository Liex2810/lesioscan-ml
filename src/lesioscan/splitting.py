"""Patient-disjoint 70/15/15 candidate-based partition with validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES=['ACK','BCC','MEL','NEV','SCC','SEK']
RATIOS={'train':.70,'val':.15,'test':.15}


def audit(split):
    if split.img_id.duplicated().any():raise ValueError('img_id duplikat')
    if split[['img_id','patient_id','diagnostic','split']].isna().any().any():raise ValueError('Missing required field')
    if set(split['split'])!=set(RATIOS):raise ValueError('Split harus train/val/test')
    sets={k:set(split.loc[split.split==k,'patient_id']) for k in RATIOS}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sets[a]&sets[b]:raise ValueError(f'Patient leakage {a}/{b}')
    table=pd.crosstab(split.diagnostic,split.split).reindex(index=CLASSES,columns=list(RATIOS),fill_value=0)
    if (table==0).any().any():raise ValueError('Salah satu kelas tidak tersedia pada salah satu split')
    return table


def make_split(metadata,output,seed=42,candidates=500):
    df=pd.read_csv(metadata)
    required=['img_id','patient_id','diagnostic']
    if df[required].isna().any().any() or df.img_id.duplicated().any():raise ValueError('Metadata belum bersih')
    if set(df.diagnostic)!=set(CLASSES):raise ValueError('Kelas tidak lengkap')
    overall=df.diagnostic.value_counts(normalize=True).reindex(CLASSES,fill_value=0).values
    best_score=np.inf; best=None
    outer=GroupShuffleSplit(n_splits=candidates,test_size=.15,random_state=seed)
    for i,(dev_idx,test_idx) in enumerate(outer.split(df,df.diagnostic,groups=df.patient_id)):
        dev=df.iloc[dev_idx];test=df.iloc[test_idx]
        inner=GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=seed+i)
        tr,val=next(inner.split(dev,dev.diagnostic,groups=dev.patient_id))
        parts={'train':dev.iloc[tr],'val':dev.iloc[val],'test':test}
        score=0
        for name,part in parts.items():
            vc=part.diagnostic.value_counts(normalize=True).reindex(CLASSES,fill_value=0).values
            counts=part.diagnostic.value_counts().reindex(CLASSES,fill_value=0)
            score+=2*abs(len(part)/len(df)-RATIOS[name])+abs(vc-overall).mean()+10*int((counts==0).sum())
        if score<best_score:best_score=score;best=parts
    if best is None or best_score>=10:raise ValueError('Tidak ditemukan split yang mencakup semua kelas. Tinjau jumlah pasien/kelas.')
    result=pd.concat([part.assign(split=name) for name,part in best.items()],ignore_index=True).sort_values('img_id').reset_index(drop=True)
    if len(result)!=len(df):raise AssertionError('Baris tidak lengkap')
    table=audit(result)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    result.to_csv(output/'split.csv',index=False);table.to_csv(output/'split_distribution.csv')
    summary={'method':'GroupShuffleSplit candidate search (not exact stratification)','seed':seed,'candidates':candidates,'best_score':float(best_score),'counts':{s:int((result.split==s).sum()) for s in RATIOS},'unique_patients':{s:int(result.loc[result.split==s,'patient_id'].nunique()) for s in RATIOS},'actual_ratios':{s:round(float((result.split==s).mean()),4) for s in RATIOS}}
    (output/'split_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,5));table.plot.bar(ax=ax);ax.set(xlabel='Kelas',ylabel='Jumlah gambar',title='Distribusi Kelas per Split');ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(output/'figures'/'split_class_distribution.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(RATIOS),[summary['counts'][s] for s in RATIOS]);ax.set(title='Jumlah Gambar per Split',ylabel='Gambar');fig.tight_layout();fig.savefig(output/'figures'/'split_sizes.png',dpi=170);plt.close(fig)
    return summary
