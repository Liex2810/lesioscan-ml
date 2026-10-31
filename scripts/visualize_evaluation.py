"""Visualize recorded per-class metrics and confusion matrix (no test tuning)."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(path:Path,out:Path):
    metrics=json.loads(path.read_text(encoding='utf-8'))
    per=metrics['per_class'];labels=[x for x in ('ACK','BCC','MEL','NEV','SCC','SEK') if x in per]
    cm=np.asarray(metrics['confusion_matrix'])
    if cm.shape!=(len(labels),len(labels)):raise ValueError('Confusion matrix shape and labels disagree')
    out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm)
    ax.set_xticks(range(len(labels)),labels=labels,rotation=45,ha='right')
    ax.set_yticks(range(len(labels)),labels=labels);ax.set(xlabel='Prediksi',ylabel='Label sebenarnya',title='Confusion Matrix')
    for i in range(len(labels)):
        for j in range(len(labels)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    x=np.arange(len(labels));w=.25
    fig,ax=plt.subplots(figsize=(9,5))
    for j,metric in enumerate(('precision','recall','f1-score')):
        ax.bar(x+(j-1)*w,[per[k][metric] for k in labels],width=w,label=metric)
    ax.set_xticks(x,labels=labels);ax.set(ylim=(0,1),ylabel='Skor',title='Metrik per Kelas');ax.legend()
    fig.tight_layout();fig.savefig(out/'per_class_metrics.png',dpi=160);plt.close(fig)
    print('Figures saved to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--out',default='outputs/evaluation')
    a=p.parse_args();render(Path(a.metrics),Path(a.out))
