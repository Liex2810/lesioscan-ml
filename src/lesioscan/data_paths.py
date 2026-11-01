"""Shared image-path lookup for PAD-UFES-20 nested image folders."""
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def index_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Folder gambar tidak ditemukan: {root}')
    mapping, conflicts = {}, []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.name in mapping:
                conflicts.append((path.name, str(mapping[path.name]), str(path)))
            else:
                mapping[path.name] = path
    return mapping, conflicts
