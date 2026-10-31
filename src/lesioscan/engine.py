import json  # Memuat json untuk fungsi yang digunakan pada file ini.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
from torch import nn  # Memuat torch untuk fungsi yang digunakan pada file ini.
from tqdm import tqdm  # Memuat tqdm untuk fungsi yang digunakan pada file ini.
from .metrics import evaluate  # Memuat metrics untuk fungsi yang digunakan pada file ini.

def run_epoch(model, loader, dev, criterion, optimizer=None, use_amp=False):  # Memproses satu epoch training atau evaluasi dan mengumpulkan prediksi.
    training = optimizer is not None  # Menetapkan status mode pelatihan.
    model.train(training)  # Memilih mode training atau evaluasi sesuai keberadaan optimizer.
    total, n, true, pred = (0.0, 0, [], [])  # Menetapkan nilai (total, n, true, pred).
    scaler_enabled = bool(use_amp and dev.type == 'cuda')  # Menetapkan status mixed precision pada CUDA.
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)  # Menetapkan pengatur skala gradien AMP.
    for x, y in tqdm(loader, leave=False):  # Mengulangi langkah untuk (x, y).
        x, y = (x.to(dev), y.to(dev))  # Menetapkan nilai (x, y).
        if training:  # Memeriksa kondisi: training.
            optimizer.zero_grad(set_to_none=True)  # Membersihkan gradien batch sebelumnya.
        with torch.set_grad_enabled(training):  # Mengaktifkan pencatatan gradien hanya selama training.
            with torch.autocast(device_type=dev.type, enabled=scaler_enabled):  # Menggunakan presisi campuran hanya ketika AMP CUDA aktif.
                logits = model(x)  # Menetapkan skor mentah kelas.
                loss = criterion(logits, y)  # Menetapkan nilai kesalahan prediksi.
            if training:  # Memeriksa kondisi: training.
                scaler.scale(loss).backward()  # Menghitung gradien loss dengan skala AMP bila aktif.
                scaler.unscale_(optimizer)  # Mengembalikan skala gradien sebelum clipping.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Membatasi norma gradien agar pembaruan tidak terlalu besar.
                scaler.step(optimizer)  # Memperbarui bobot melalui optimizer dengan penanganan AMP.
                scaler.update()  # Memperbarui faktor skala gradien.
        total += float(loss.item()) * len(y)  # Mengakumulasi loss batch dikalikan jumlah sampel agar rata-rata tepat.
        n += len(y)  # Menambahkan jumlah sampel dalam batch ke penghitung epoch.
        true.extend(y.cpu().tolist())  # Menyalin label target ke CPU dan mengumpulkannya untuk evaluasi.
        pred.extend(logits.argmax(1).cpu().tolist())  # Memilih indeks skor terbesar dan mengumpulkan prediksi pada CPU.
    return (total / max(n, 1), true, pred)  # Mengembalikan rata-rata loss per sampel, label sebenarnya, dan prediksi.

def save_json(path, payload):  # Menyimpan hasil eksperimen sebagai JSON berindentasi.
    path = Path(path)  # Menetapkan lokasi file.
    path.parent.mkdir(parents=True, exist_ok=True)  # Menjalankan path.parent.mkdir untuk melanjutkan proses.
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')  # Menjalankan path.write_text untuk melanjutkan proses.
