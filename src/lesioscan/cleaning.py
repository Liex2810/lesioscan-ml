"""Auditable metadata/image validation; never modifies original patient data or pixels."""
from pathlib import Path
import pandas as pd
from PIL import Image


def clean_metadata(frame: pd.DataFrame, image_root, labels, image_col='img_id',
                   patient_col='patient_id', label_col='diagnostic', verify_images=True):
    """Return (clean_frame, issues_frame, stats). Unsafe/ambiguous rows are quarantined.

    Exact duplicate rows may be reduced to one. Conflicting image identifiers are
    quarantined together, not resolved by guessing a diagnosis or patient.
    """
    cols = (image_col, patient_col, label_col)
    missing_cols = set(cols) - set(frame.columns)
    if missing_cols:
        raise ValueError(f'Kolom metadata tidak ditemukan: {sorted(missing_cols)}')
    data = frame.copy().reset_index(drop=True)
    data.insert(0, '_source_row', data.index + 2)  # +2: CSV header at line 1
    for col in cols:
        data[col] = data[col].astype('string').str.strip()
        data[col] = data[col].replace('', pd.NA)

    original_count = len(data)
    issues = []
    rejected = set()
    def issue(indices, code, detail):
        for idx in indices:
            issues.append({'source_row': int(data.at[idx, '_source_row']),
                           'image_id': str(data.at[idx, image_col]),
                           'issue': code, 'detail': detail})
            rejected.add(int(idx))

    for col in cols:
        indices = data.index[data[col].isna()].tolist()
        issue(indices, 'missing_required', f'Kolom {col} kosong')

    unknown = data.index[data[label_col].notna() & ~data[label_col].isin(labels)].tolist()
    issue(unknown, 'unknown_label', 'Label tidak termasuk enam kategori yang dikonfigurasi')

    # Detect conflicting metadata for the same identifier, including patient / label.
    present = data.loc[data[image_col].notna()]
    for image_id, group in present.groupby(image_col, sort=False):
        if len(group) > 1:
            distinct = group[[patient_col, label_col]].drop_duplicates()
            if len(distinct) > 1:
                issue(group.index, 'conflicting_image_id', f'Konflik patient_id/diagnostic untuk {image_id}')
            else:
                # Only one copy remains; duplicates are recorded.
                issue(group.index[1:], 'duplicate_record', 'Baris duplikat identik pada kolom inti')

    root = Path(image_root)
    for image_id, group in present.groupby(image_col, sort=False):
        filename = str(image_id)
        # Disallow absolute paths or climbing out of image directory.
        path_name = Path(filename)
        if path_name.is_absolute() or '..' in path_name.parts:
            issue(group.index, 'unsafe_filename', 'Path gambar tidak aman')
            continue
        path = root / filename
        if not path.is_file():
            issue(group.index, 'missing_image', f'Tidak ditemukan: {filename}')
        elif verify_images:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert('RGB').load()
            except (OSError, ValueError, SyntaxError) as exc:
                issue(group.index, 'unreadable_image', f'{filename}: {type(exc).__name__}')

    cleaned = data.drop(index=sorted(rejected)).drop(columns=['_source_row']).reset_index(drop=True)
    issue_df = pd.DataFrame(issues, columns=['source_row', 'image_id', 'issue', 'detail'])
    stats = {'input_rows': int(original_count), 'accepted_rows': int(len(cleaned)),
             'quarantined_rows': int(len(rejected)),
             'issues_by_type': issue_df['issue'].value_counts().to_dict(),
             'classes_remaining': cleaned[label_col].value_counts().to_dict()}
    return cleaned, issue_df, stats
