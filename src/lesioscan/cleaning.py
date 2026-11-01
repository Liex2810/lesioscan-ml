"""Auditable PAD-UFES-20 metadata and image integrity cleaning."""
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from .data_paths import index_images

LABELS={'ACK','BCC','MEL','NEV','SCC','SEK'}
REQUIRED=['img_id','patient_id','diagnostic']


def clean_dataset(metadata, images_root, output, reuse_integrity=False):
    output=Path(output); tables=output/'tables';tables.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(metadata);df=raw.copy();df.columns=df.columns.str.strip().str.lower()
    if df.columns.duplicated().any():raise ValueError('Nama kolom duplikat')
    if not set(REQUIRED).issubset(df):raise ValueError(f'Kolom kurang: {set(REQUIRED)-set(df)}')
    for col in REQUIRED:
        df[col]=df[col].astype('string').str.strip().replace({'':pd.NA,'nan':pd.NA,'None':pd.NA,'null':pd.NA})
    df['diagnostic']=df.diagnostic.str.upper()
    paths,conflicts=index_images(images_root)
    pd.DataFrame(conflicts,columns=['filename','first_path','second_path']).to_csv(tables/'duplicate_filenames.csv',index=False)
    integrity_path=tables/'image_integrity.csv'
    if reuse_integrity and integrity_path.is_file():
        integrity=pd.read_csv(integrity_path)
        if set(integrity.img_id.astype(str))!=set(paths):raise ValueError('Cache integritas tidak mencakup file sekarang; ulangi tanpa --reuse-integrity')
        integrity['valid']=integrity.valid.astype(str).str.lower().eq('true')
    else:
        records=[]
        for name,path in tqdm(paths.items(),desc='Verifikasi dan decode gambar'):
            rec={'img_id':name,'valid':False,'width':None,'height':None,'format':None,'error':''}
            try:
                with Image.open(path) as im:im.verify()
                with Image.open(path) as im:
                    im.load();rec.update(valid=True,width=im.width,height=im.height,format=im.format)
            except (OSError,ValueError,SyntaxError) as exc:rec['error']=str(exc)
            records.append(rec)
        integrity=pd.DataFrame(records,columns=['img_id','valid','width','height','format','error'])
        integrity.to_csv(integrity_path,index=False)
    invalid=set(integrity.loc[~integrity.valid,'img_id'].astype(str))
    duplicates=set(x[0] for x in conflicts)
    issues=[]
    def add(mask,reason):
        for idx in df.index[mask]:issues.append({'row_index':int(idx),'img_id':str(df.at[idx,'img_id']),'reason':reason})
    add(df[REQUIRED].isna().any(axis=1),'missing_required_metadata')
    add(df.diagnostic.notna() & ~df.diagnostic.isin(LABELS),'invalid_diagnostic')
    add(df.img_id.notna() & df.img_id.duplicated(keep=False),'duplicate_image_id')
    add(df.img_id.notna() & ~df.img_id.isin(paths),'missing_image_file')
    add(df.img_id.isin(duplicates),'ambiguous_duplicate_filename')
    add(df.img_id.isin(invalid),'corrupted_image')
    ids=df.dropna(subset=['img_id']).groupby('img_id').patient_id.nunique(dropna=False)
    add(df.img_id.isin(ids.index[ids>1]),'patient_identity_conflict')
    issues_df=pd.DataFrame(issues,columns=['row_index','img_id','reason'])
    issues_df.to_csv(tables/'cleaning_issues.csv',index=False)
    reason_map=issues_df.groupby('row_index').reason.agg(lambda x:'; '.join(sorted(set(x)))) if len(issues_df) else pd.Series(dtype=str)
    df['cleaning_issue']=df.index.map(reason_map)
    quarantine=df.loc[df.cleaning_issue.notna()].copy()
    clean=df.loc[df.cleaning_issue.isna()].drop(columns='cleaning_issue').copy()
    clean.to_csv(output/'metadata_clean.csv',index=False)
    quarantine.to_csv(output/'quarantine.csv',index=False)
    integrity.loc[~integrity.valid].to_csv(tables/'corrupted_images.csv',index=False)
    integrity.loc[integrity.valid & ((integrity.width<224)|(integrity.height<224))].to_csv(tables/'low_resolution_images.csv',index=False)
    df.loc[df.img_id.notna() & ~df.img_id.isin(paths)].to_csv(tables/'missing_images.csv',index=False)
    pd.DataFrame({'img_id':sorted(set(paths)-set(df.img_id.dropna().astype(str)))}).to_csv(tables/'unreferenced_images.csv',index=False)
    before=df.diagnostic.value_counts();after=clean.diagnostic.value_counts()
    pd.concat([before.rename('before'),after.rename('after')],axis=1).fillna(0).astype(int).to_csv(tables/'class_distribution_after_cleaning.csv')
    summary={'initial_metadata_rows':len(raw),'clean_metadata_rows':len(clean),'quarantined_rows':len(quarantine),'indexed_image_files':len(paths),'image_integrity_valid':int(integrity.valid.sum()),'corrupted_image_files':int((~integrity.valid).sum()),'issue_counts':issues_df.reason.value_counts().to_dict()}
    (output/'cleaning_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary
