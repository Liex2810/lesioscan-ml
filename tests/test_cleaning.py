from pathlib import Path
import pandas as pd
from PIL import Image
from lesioscan.cleaning import clean_metadata


def test_clean_metadata_quarantines_conflicts_and_missing(tmp_path):
    Image.new('RGB', (5, 5), 'white').save(tmp_path / 'a.png')
    Image.new('RGB', (5, 5), 'black').save(tmp_path / 'b.png')
    frame = pd.DataFrame([
        {'img_id': 'a.png', 'patient_id': 'P01', 'diagnostic': 'MEL'},
        {'img_id': 'a.png', 'patient_id': 'P01', 'diagnostic': 'MEL'},
        {'img_id': 'b.png', 'patient_id': 'P02', 'diagnostic': 'BCC'},
        {'img_id': 'b.png', 'patient_id': 'P02', 'diagnostic': 'NEV'},
        {'img_id': 'missing.png', 'patient_id': 'P03', 'diagnostic': 'MEL'},
        {'img_id': 'bad.png', 'patient_id': '', 'diagnostic': 'MEL'},
    ])
    cleaned, issues, stats = clean_metadata(frame, tmp_path, ['MEL','BCC','NEV'])
    assert list(cleaned['img_id']) == ['a.png']
    assert stats['accepted_rows'] == 1
    assert {'duplicate_record','conflicting_image_id','missing_image','missing_required'} <= set(issues['issue'])
    assert len(frame) == 6  # original untouched


def test_clean_metadata_rejects_unknown_label(tmp_path):
    Image.new('RGB', (5, 5)).save(tmp_path / 'x.png')
    raw = pd.DataFrame([{'img_id':'x.png','patient_id':'P1','diagnostic':'UNKNOWN'}])
    cleaned, issues, _ = clean_metadata(raw,tmp_path,['MEL'])
    assert cleaned.empty
    assert 'unknown_label' in set(issues.issue)
