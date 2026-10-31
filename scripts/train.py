import argparse  # Memuat argparse untuk fungsi yang digunakan pada file ini.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import sys  # Memuat sys untuk fungsi yang digunakan pada file ini.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))  # Menambahkan folder src agar package dapat diimpor saat script dijalankan.
import pandas as pd  # Memuat pandas untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
from lesioscan.config import read_config, seed_everything, device  # Memuat lesioscan.config untuk fungsi yang digunakan pada file ini.
from lesioscan.data import SkinDataset, make_loader  # Memuat lesioscan.data untuk fungsi yang digunakan pada file ini.
from lesioscan.models import build, freeze_backbone, ALLOWED  # Memuat lesioscan.models untuk fungsi yang digunakan pada file ini.
from lesioscan.metrics import evaluate  # Memuat lesioscan.metrics untuk fungsi yang digunakan pada file ini.
from lesioscan.engine import run_epoch, save_json  # Memuat lesioscan.engine untuk fungsi yang digunakan pada file ini.
p = argparse.ArgumentParser(description='Pelatihan dua tahap LesioScan pada CPU/GPU Colab')  # Menetapkan parser argumen atau parameter model.
p.add_argument('--config', default='configs/default.yaml')  # Menjalankan p.add_argument untuk melanjutkan proses.
p.add_argument('--model', default=None, choices=ALLOWED)  # Menjalankan p.add_argument untuk melanjutkan proses.
p.add_argument('--run', default=None)  # Menjalankan p.add_argument untuk melanjutkan proses.
a = p.parse_args()  # Menetapkan argumen dari pengguna.
cfg = read_config(a.config)  # Menetapkan konfigurasi eksperimen.
seed_everything(cfg['seed'])  # Menyamakan seed generator acak sebelum eksperimen.
d = cfg['data']  # Menetapkan pengaturan data.
tr = cfg['train']  # Menetapkan pengaturan pelatihan.
name = a.model or cfg['model']['name']  # Menetapkan nama arsitektur.
frame = pd.read_csv(d['split'])  # Menetapkan tabel metadata.
if not {'train', 'val'}.issubset(set(frame['split'])):  # Memeriksa kondisi: not {'train', 'val'}.issubset(set(frame['split'])).
    raise ValueError('Run scripts/prepare.py first')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
for s in ('train', 'val'):  # Mengulangi langkah untuk s.
    present = set(frame.loc[frame.split == s, d['label_column']])  # Menetapkan nilai present.
    if present != set(cfg['labels']):  # Memeriksa kondisi: present != set(cfg['labels']).
        raise ValueError(f"{s} is missing classes: {set(cfg['labels']) - present}; use another patient split")  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
loaders = {}  # Menetapkan kumpulan loader train dan validasi.
for split in ('train', 'val'):  # Mengulangi langkah untuk split.
    ds = SkinDataset(frame[frame.split == split], d['images'], d['image_column'], d['label_column'], cfg['labels'], d['image_size'], training=split == 'train')  # Menetapkan dataset gambar.
    loaders[split] = make_loader(ds, tr['batch_size'], training=split == 'train', weighted=split == 'train' and tr['sampler'] == 'weighted', num_workers=tr['num_workers'])  # Menetapkan kumpulan loader train dan validasi.
dev = device()  # Menetapkan perangkat CPU atau GPU.
print('Device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')  # Menampilkan informasi proses atau hasil kepada pengguna.
m = build(name, len(cfg['labels']), tr['pretrained']).to(dev)  # Menetapkan model klasifikasi.
criterion = torch.nn.CrossEntropyLoss()  # Menetapkan fungsi loss klasifikasi.
run = a.run or name  # Menetapkan nama eksperimen.
outdir = Path(cfg['outputs']) / 'runs' / run  # Menetapkan folder hasil eksperimen.
outdir.mkdir(parents=True, exist_ok=True)  # Menjalankan outdir.mkdir untuk melanjutkan proses.
best_score, bad, history = (-1.0, 0, [])  # Menetapkan nilai (best_score, bad, history).
for stage, epochs, lr in [(1, tr['stage1_epochs'], tr['head_lr']), (2, tr['stage2_epochs'], tr['finetune_lr'])]:  # Mengulangi langkah untuk (stage, epochs, lr).
    if stage == 1:  # Memeriksa kondisi: stage == 1.
        freeze_backbone(m, name)  # Membekukan backbone untuk tahap pelatihan kepala klasifikasi.
    else:  # Menjalankan jalur alternatif saat kondisi sebelumnya tidak terpenuhi.
        for x in m.parameters():  # Mengulangi langkah untuk x.
            x.requires_grad = True  # Membuka seluruh parameter untuk fine-tuning tahap kedua.
    opt = torch.optim.AdamW((x for x in m.parameters() if x.requires_grad), lr=lr, weight_decay=tr['weight_decay'])  # Menetapkan optimizer AdamW.
    bad = 0  # Mereset penghitung epoch tanpa peningkatan pada tahap baru.
    for epoch in range(epochs):  # Mengulangi langkah untuk epoch.
        train_loss, _, _ = run_epoch(m, loaders['train'], dev, criterion, opt, tr['amp'])  # Menetapkan nilai (train_loss, _, _).
        val_loss, y, pred = run_epoch(m, loaders['val'], dev, criterion)  # Menetapkan nilai (val_loss, y, pred).
        metrics = evaluate(y, pred, cfg['labels'])  # Menetapkan hasil metrik.
        row = {'stage': stage, 'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'macro_f1': metrics['macro_f1'], 'balanced_accuracy': metrics['balanced_accuracy']}  # Menetapkan catatan satu epoch.
        history.append(row)  # Menambahkan catatan epoch ke riwayat eksperimen.
        print(row)  # Menampilkan informasi proses atau hasil kepada pengguna.
        if metrics['macro_f1'] > best_score:  # Memeriksa kondisi: metrics['macro_f1'] > best_score.
            best_score = metrics['macro_f1']  # Menetapkan skor validasi terbaik.
            bad = 0  # Mereset penghitung epoch tanpa peningkatan pada tahap baru.
            torch.save({'model': name, 'state_dict': m.state_dict(), 'labels': cfg['labels'], 'image_size': d['image_size'], 'config': cfg}, outdir / 'best.pt')  # Menyimpan bobot terbaik dan konfigurasi model.
            save_json(outdir / 'val_metrics.json', metrics)  # Menyimpan hasil terstruktur sebagai JSON.
        else:  # Menjalankan jalur alternatif saat kondisi sebelumnya tidak terpenuhi.
            bad += 1  # Menambah penghitung epoch tanpa peningkatan skor validasi.
            if bad >= tr['patience']:  # Memeriksa kondisi: bad >= tr['patience'].
                print('Early stop stage:', stage)  # Menampilkan informasi proses atau hasil kepada pengguna.
                break  # Menghentikan loop ketika early stopping tercapai.
save_json(outdir / 'history.json', history)  # Menyimpan hasil terstruktur sebagai JSON.
print('Best validation macro-F1:', best_score, 'Checkpoint:', outdir / 'best.pt')  # Menampilkan informasi proses atau hasil kepada pengguna.
