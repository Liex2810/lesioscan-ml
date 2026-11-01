"""Reproducible descriptive EDA, saving CSV tables and PNG charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from .data_paths import index_images

CLASSES = ['ACK','BCC','MEL','NEV','SCC','SEK']


def save_fig(fig, path):
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_eda(metadata, images_root, output, full_image_scan=False, sample_per_class=3):
    output = Path(output)
    figures, tables = output/'figures', output/'tables'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata)
    expected = {'img_id','patient_id','diagnostic'}
    if not expected.issubset(df.columns):
        raise ValueError(f'Kolom metadata kurang: {sorted(expected-set(df.columns))}')
    paths, conflicts = index_images(images_root)
    counts = df['diagnostic'].value_counts().reindex(CLASSES, fill_value=0)
    class_table = pd.DataFrame({'class':CLASSES,'images':counts.values,'percent':(counts.values/len(df)*100).round(2),'unique_patients':[df.loc[df.diagnostic==c,'patient_id'].nunique() for c in CLASSES]})
    class_table.to_csv(tables/'class_distribution.csv',index=False)
    missing = df.loc[~df['img_id'].astype(str).isin(paths)]
    missing.to_csv(tables/'missing_images.csv',index=False)
    pd.DataFrame(conflicts,columns=['filename','path_a','path_b']).to_csv(tables/'duplicate_filenames.csv',index=False)
    missing_stats = df.isna().agg(['sum','mean']).T
    missing_stats.columns = ['count','fraction']
    missing_stats['percent'] = (missing_stats.fraction*100).round(2)
    missing_stats.to_csv(tables/'missing_values.csv')
    counts_pat = df.groupby('patient_id').size().rename('image_count')
    counts_pat.to_csv(tables/'images_per_patient.csv')
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(CLASSES,counts.values);ax.invert_yaxis();ax.set_xlabel('Jumlah gambar');ax.set_title('Distribusi Kelas PAD-UFES-20')
    for i,v in enumerate(counts.values): ax.text(v+max(counts)*.01,i,str(v),va='center')
    ax.set_xlim(0,max(counts)*1.17); save_fig(fig,figures/'class_distribution.png')
    fig,ax=plt.subplots(figsize=(9,5));ax.hist(counts_pat.values,bins=min(30,max(1,len(np.unique(counts_pat)))));ax.set_title('Jumlah Gambar per Pasien');ax.set_xlabel('Gambar');ax.set_ylabel('Pasien');save_fig(fig,figures/'images_per_patient.png')
    if not missing_stats.empty:
        fig,ax=plt.subplots(figsize=(11,max(5,len(missing_stats)*.32))); values=missing_stats.sort_values('percent',ascending=True);ax.barh(values.index,values.percent);ax.set_xlim(0,105);ax.set_xlabel('Nilai kosong (%)');ax.set_title('Missing Values Metadata');save_fig(fig,figures/'missing_values.png')
    if 'age' in df:
        vals=pd.to_numeric(df['age'],errors='coerce').dropna()
        if not vals.empty:
            fig,ax=plt.subplots(figsize=(9,4));ax.hist(vals,bins=20);ax.set_xlabel('Usia (tahun)');ax.set_ylabel('Gambar');ax.set_title('Sebaran Usia (per Gambar)');save_fig(fig,figures/'age_distribution.png')
    if 'gender' in df:
        cross=pd.crosstab(df.diagnostic,df.gender.fillna('Tidak diketahui')).reindex(CLASSES,fill_value=0);cross.to_csv(tables/'gender_by_class.csv');fig,ax=plt.subplots(figsize=(9,5));cross.plot.bar(ax=ax);ax.set_title('Gender per Kelas (per Gambar)');ax.set_xlabel('Kelas');ax.tick_params(axis='x',rotation=0);save_fig(fig,figures/'gender_by_class.png')
    available=df.loc[df.img_id.astype(str).isin(paths)]
    fig, axes=plt.subplots(len(CLASSES),sample_per_class,figsize=(4*sample_per_class,3.4*len(CLASSES)),squeeze=False)
    for i,label in enumerate(CLASSES):
        rows=available.loc[available.diagnostic==label].sample(n=min(sample_per_class,(available.diagnostic==label).sum()),random_state=42)
        for j,ax in enumerate(axes[i]):
            ax.axis('off')
            if j>=len(rows):continue
            name=str(rows.iloc[j].img_id)
            try:
                with Image.open(paths[name]) as im: ax.imshow(im.convert('RGB'))
                ax.set_title(f'{label} — Sampel {j+1}',fontsize=10,pad=9)
            except (OSError,ValueError):ax.set_title(f'{label}: gagal dibaca')
    fig.suptitle('Sampel Gambar per Kelas',y=.998,fontsize=16);fig.subplots_adjust(top=.965,hspace=.35,wspace=.2);save_fig(fig,figures/'sample_images_per_class.png')
    resolution_path=tables/'image_resolution.csv'
    if full_image_scan or not resolution_path.exists():
        records=[];errors=[]
        for name, path in tqdm(
    paths.items(),
    total=len(paths),
    desc="LesioScan | Image Resolution",
    unit="gambar",
    dynamic_ncols=True,
    mininterval=0.5,
    bar_format=(
        "{desc}: {percentage:3.0f}% |{bar}| "
        "{n_fmt}/{total_fmt} gambar "
        "[{elapsed}<{remaining}]"
    )
):
            try:
                with Image.open(path) as im:width,height=im.size
                records.append({'img_id':name,'width':width,'height':height,'aspect_ratio':width/height,'megapixels':width*height/1e6})
            except (OSError,ValueError) as exc:errors.append({'img_id':name,'error':str(exc)})
        pd.DataFrame(records,columns=['img_id','width','height','aspect_ratio','megapixels']).to_csv(resolution_path,index=False)
        pd.DataFrame(errors,columns=['img_id','error']).to_csv(tables/'unreadable_images.csv',index=False)
    resolution=pd.read_csv(resolution_path)
    if not resolution.empty:
        fig,ax=plt.subplots(figsize=(9,5));ax.scatter(resolution.width,resolution.height,s=8,alpha=.35);ax.set(xlabel='Lebar (px)',ylabel='Tinggi (px)',title='Resolusi Gambar');save_fig(fig,figures/'image_resolution.png')
    summary={'metadata_rows':len(df),'unique_patients':int(df.patient_id.nunique()),'indexed_images':len(paths),'missing_image_rows':len(missing),'duplicate_image_ids':int(df.img_id.duplicated().sum()),'class_distribution':{str(k):int(v) for k,v in counts.items()},'resolution_rows':len(resolution),'full_image_scan':bool(full_image_scan)}
    (output/'eda_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary
