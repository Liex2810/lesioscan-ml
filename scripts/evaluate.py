import argparse  # Memuat argparse untuk fungsi yang digunakan pada file ini.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import sys  # Memuat sys untuk fungsi yang digunakan pada file ini.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))  # Menambahkan folder src agar package dapat diimpor saat script dijalankan.
import pandas as pd  # Memuat pandas untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
from lesioscan.config import read_config, device  # Memuat lesioscan.config untuk fungsi yang digunakan pada file ini.
from lesioscan.data import SkinDataset, make_loader  # Memuat lesioscan.data untuk fungsi yang digunakan pada file ini.
from lesioscan.models import build  # Memuat lesioscan.models untuk fungsi yang digunakan pada file ini.
from lesioscan.engine import run_epoch, save_json  # Memuat lesioscan.engine untuk fungsi yang digunakan pada file ini.
from lesioscan.metrics import evaluate  # Memuat lesioscan.metrics untuk fungsi yang digunakan pada file ini.
p = argparse.ArgumentParser()  # Menetapkan parser argumen atau parameter model.
p.add_argument('--checkpoint', required=True)  # Menjalankan p.add_argument untuk melanjutkan proses.
p.add_argument('--config', default='configs/default.yaml')  # Menjalankan p.add_argument untuk melanjutkan proses.
a = p.parse_args()  # Menetapkan argumen dari pengguna.
cfg = read_config(a.config)  # Menetapkan konfigurasi eksperimen.
d = cfg['data']  # Menetapkan pengaturan data.
ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)  # Menetapkan checkpoint model.
df = pd.read_csv(d['split'])  # Menetapkan tabel data.
df = df[df.split == 'test']  # Menetapkan tabel data.
if set(df[d['label_column']]) != set(cfg['labels']):  # Memeriksa kondisi: set(df[d['label_column']]) != set(cfg['labels']).
    print('WARNING: Some test classes are missing; interpret metrics carefully')  # Menampilkan informasi proses atau hasil kepada pengguna.
ds = SkinDataset(df, d['images'], d['image_column'], d['label_column'], ckpt['labels'], ckpt['image_size'])  # Menetapkan dataset gambar.
loader = make_loader(ds, cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'])  # Menetapkan pemuat batch.
m = build(ckpt['model'], len(ckpt['labels']), pretrained=False)  # Menetapkan model klasifikasi.
m.load_state_dict(ckpt['state_dict'])  # Memuat bobot checkpoint ke arsitektur model.
dev = device()  # Menetapkan perangkat CPU atau GPU.
m.to(dev)  # Menjalankan m.to untuk melanjutkan proses.
_, true, pred = run_epoch(m, loader, dev, torch.nn.CrossEntropyLoss())  # Menetapkan nilai (_, true, pred).
metrics = evaluate(true, pred, ckpt['labels'])  # Menetapkan hasil metrik.
save_json(Path(a.checkpoint).parent / 'test_metrics.json', metrics)  # Menyimpan hasil terstruktur sebagai JSON.
print('TEST macro-F1:', metrics['macro_f1'], '\nPer-class recalls:', {k: metrics['per_class'][k]['recall'] for k in ckpt['labels']})  # Menampilkan informasi proses atau hasil kepada pengguna.
print('IMPORTANT: Do not repeatedly tune based on this test result.')  # Menampilkan informasi proses atau hasil kepada pengguna.
