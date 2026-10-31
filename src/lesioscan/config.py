from pathlib import Path
import random
import numpy as np
import torch
import yaml


def read_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)
