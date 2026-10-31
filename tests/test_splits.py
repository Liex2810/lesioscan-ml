import sys  # Memuat sys untuk fungsi yang digunakan pada file ini.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))  # Menambahkan folder src agar package dapat diimpor saat script dijalankan.
import pandas as pd  # Memuat pandas untuk fungsi yang digunakan pada file ini.
from lesioscan.data import split_metadata  # Memuat lesioscan.data untuk fungsi yang digunakan pada file ini.

def test_patient_disjoint():  # Memastikan pasien pada data buatan tidak tersebar ke subset berbeda.
    df = pd.DataFrame({'patient_id': [str(p) for p in range(80) for _ in range(2)], 'diagnostic': ['ACK', 'BCC'] * 80, 'img_id': [str(i) for i in range(160)]})  # Menetapkan tabel data.
    out = split_metadata(df)  # Menetapkan hasil pembagian atau path keluaran.
    groups = {s: set(out.loc[out.split == s, 'patient_id']) for s in ('train', 'val', 'test')}  # Menetapkan kelompok pasien setiap subset.
    assert all((groups[x].isdisjoint(groups[y]) for x, y in [('train', 'val'), ('train', 'test'), ('val', 'test')]))  # Memastikan kondisi pengujian benar; gagal bila pasien saling tumpang tindih.
