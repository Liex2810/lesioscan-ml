import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lesioscan.data_paths import index_images
from lesioscan.splitting import audit
import pandas as pd
import pytest

def test_index_images_nested(tmp_path):
    (tmp_path/'imgs_part_1').mkdir()
    (tmp_path/'imgs_part_1'/'A.png').write_bytes(b'abc')
    mapping, conflicts=index_images(tmp_path)
    assert list(mapping)==['A.png'] and conflicts==[]

def test_patient_leakage_detected():
    rows=[]
    for c in ['ACK','BCC','MEL','NEV','SCC','SEK']:
        for s in ['train','val','test']:
            rows.append({'img_id':f'{c}_{s}','patient_id':f'{c}_{s}','diagnostic':c,'split':s})
    rows[0]['patient_id']=rows[1]['patient_id']
    with pytest.raises(ValueError,match='Patient leakage'):
        audit(pd.DataFrame(rows))
