import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pytest
from PIL import Image
from lesioscan.dataset import load_and_validate_split, make_dataloaders
from lesioscan.preprocessing import build_transforms


def test_fake_dataset(tmp_path):
    classes=['ACK','BCC','MEL','NEV','SCC','SEK']; rows=[]
    for j,split in enumerate(['train','val','test']):
        for i,c in enumerate(classes):
            name=f'{split}_{c}.png'; Image.new('RGB',(120,180),(140,80,90)).save(tmp_path/name)
            rows.append(dict(img_id=name,patient_id=f'P{j}_{i}',diagnostic=c,split=split))
    csv=tmp_path/'split.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    df,paths,dist=load_and_validate_split(csv,tmp_path)
    tr,ev=build_transforms(224)
    loaders,_=make_dataloaders(df,paths,tr,ev,batch_size=3)
    x,y=next(iter(loaders['train']))
    assert tuple(x.shape)==(3,3,224,224)
    assert set(dist.columns)=={'train','val','test'}


def test_patient_leakage_fails(tmp_path):
    classes=['ACK','BCC','MEL','NEV','SCC','SEK'];rows=[]
    for j,s in enumerate(['train','val','test']):
        for i,c in enumerate(classes):
            name=f'{s}_{c}.png';Image.new('RGB',(10,10)).save(tmp_path/name)
            rows.append(dict(img_id=name,patient_id='SHARED' if i==0 else f'{s}_{i}',diagnostic=c,split=s))
    csv=tmp_path/'split.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    with pytest.raises(ValueError,match='Patient leakage'):
        load_and_validate_split(csv,tmp_path)
