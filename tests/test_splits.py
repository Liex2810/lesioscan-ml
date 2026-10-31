import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from dermascan.data import split_metadata

def test_patient_disjoint():
    df=pd.DataFrame({'patient_id':[str(p) for p in range(80) for _ in range(2)],'diagnostic':['ACK','BCC']*80,'img_id':[str(i) for i in range(160)]})
    out=split_metadata(df)
    groups={s:set(out.loc[out.split==s,'patient_id']) for s in ('train','val','test')}
    assert all(groups[x].isdisjoint(groups[y]) for x,y in [('train','val'),('train','test'),('val','test')])
