import pandas as pd
import pytest
from lesioscan.splitting import stratified_patient_split


def dataset():
    rows=[]
    for label in ('ACK','BCC','MEL','NEV','SCC','SEK'):
        for i in range(25):
            for img in range(2):rows.append({'patient_id':f'{label}_{i}', 'diagnostic':label,'img_id':f'{label}_{i}_{img}.jpg'})
    return pd.DataFrame(rows)

def test_split_no_overlap_and_all_classes():
    data=dataset()
    a=stratified_patient_split(data,seed=11,n_candidates=70)
    b=stratified_patient_split(data,seed=11,n_candidates=70)
    assert a['split'].equals(b['split'])
    assert set(a['split'])=={'train','val','test'}
    for split in ('train','val','test'):
        assert set(a.loc[a.split==split,'diagnostic'])==set(data.diagnostic)
    groups={s:set(a.loc[a.split==s,'patient_id']) for s in ('train','val','test')}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])

def test_impossible_coverage_raises():
    df=pd.DataFrame({'patient_id':['a','b','c','d'],'diagnostic':['ACK','ACK','MEL','MEL']})
    with pytest.raises(ValueError,match='distinct patients'):
        stratified_patient_split(df)
