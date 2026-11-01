"""ImageNet-compatible preprocessing. Training augmentation only, never test augmentation."""
from torchvision import transforms
from PIL import Image, ImageOps

CLASSES = ('ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK')
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224):
    if image_size <= 0:
        raise ValueError('image_size harus lebih besar dari nol')
    # Pad mempertahankan seluruh lesi tanpa cropping atau distorsi aspect ratio.
    base = [transforms.Lambda(lambda image: ImageOps.pad(
        image, (image_size, image_size), color=(0, 0, 0), method=Image.Resampling.BILINEAR
    ))]
    train = transforms.Compose(base + [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=(0, 0, 0)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluation = transforms.Compose(base + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluation


def denormalize(tensor):
    import torch
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    return (tensor * std + mean).clamp(0, 1)
