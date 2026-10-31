"""Summarize validation metrics. Selection MUST NOT use the holdout test."""
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize(root: Path, out: Path):
    rows=[]
    for path in sorted(root.glob('*/val_metrics.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        row={'run':path.parent.name,'macro_f1':data.get('macro_f1'),
             'balanced_accuracy':data.get('balanced_accuracy'), 'accuracy':data.get('accuracy')}
        for cls in ('ACK','BCC','MEL','NEV','SCC','SEK'):
            row[f'{cls}_recall']=data.get('per_class',{}).get(cls,{}).get('recall')
        model_file=path.parent/'best.pt'
        row['checkpoint_mb']=round(model_file.stat().st_size/1024**2,2) if model_file.exists() else None
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No val_metrics.json in {root}/*/; train a model first.')
    frame=pd.DataFrame(rows).sort_values(['macro_f1','balanced_accuracy'],ascending=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out,index=False)
    print(frame.to_string(index=False));print('Saved:',out)
    return frame

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',default='outputs/runs');ap.add_argument('--out',default='outputs/benchmark/validation_comparison.csv')
    args=ap.parse_args();summarize(Path(args.runs),Path(args.out))
