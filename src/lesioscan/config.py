from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import random  # Memuat random untuk fungsi yang digunakan pada file ini.
import numpy as np  # Memuat numpy untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
import yaml  # Memuat yaml untuk fungsi yang digunakan pada file ini.

def read_config(path):  # Membaca pengaturan YAML menjadi dictionary Python.
    with open(path, encoding='utf-8') as f:  # Membuka konfigurasi teks UTF-8 dan menutupnya otomatis.
        return yaml.safe_load(f)  # Membaca YAML tanpa mengeksekusi konstruksi Python arbitrer.

def seed_everything(seed):  # Mengatur seed generator acak untuk membantu pengulangan eksperimen.
    random.seed(seed)  # Menjalankan random.seed untuk melanjutkan proses.
    np.random.seed(seed)  # Menjalankan np.random.seed untuk melanjutkan proses.
    torch.manual_seed(seed)  # Menjalankan torch.manual_seed untuk melanjutkan proses.
    if torch.cuda.is_available():  # Memeriksa kondisi: torch.cuda.is_available().
        torch.cuda.manual_seed_all(seed)  # Menjalankan torch.cuda.manual_seed_all untuk melanjutkan proses.

def device():  # Memilih CUDA bila tersedia, selain itu CPU.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Mengembalikan perangkat CUDA bila tersedia, atau CPU sebagai fallback.

def ensure_path(path):  # Membuat folder beserta induknya bila belum tersedia.
    Path(path).mkdir(parents=True, exist_ok=True)  # Menjalankan Path(path).mkdir untuk melanjutkan proses.
