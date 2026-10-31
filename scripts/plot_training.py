"""Create training history figures from a saved history.json."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(history_path:Path,out:Path):
    history=json.loads(history_path.read_text(encoding='utf-8'))
    if not history:raise ValueError('Empty training history')
    out.mkdir(parents=True,exist_ok=True)
    x=list(range(1,len(history)+1))
    for title,keys in [('loss',['train_loss','val_loss']),('validation_metrics',['macro_f1','balanced_accuracy'])]:
        fig,ax=plt.subplots(figsize=(9,5))
        for key in keys:
            if any(key in h for h in history):ax.plot(x,[h.get(key,float('nan')) for h in history],marker='.',label=key)
        ax.set(xlabel='Epoch (global)',ylabel=title,title=title.replace('_',' ').title())
        ax.legend();ax.grid(alpha=.3)
        for i in range(1,len(history)):
            if history[i].get('stage')!=history[i-1].get('stage'):ax.axvline(i+.5,linestyle='--',alpha=.5)
        fig.tight_layout();fig.savefig(out/f'{title}.png',dpi=160);plt.close(fig)
    print('Saved figures to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--out',default='outputs/plots')
    a=p.parse_args();plot(Path(a.history),Path(a.out))
