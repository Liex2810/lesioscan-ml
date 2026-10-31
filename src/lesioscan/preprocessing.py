"""Preprocessing: augment training images; deterministic transforms for validation/test/inference."""
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size: int, training: bool = False):
    if training:
        ops = [T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
               T.RandomHorizontalFlip(), T.RandomRotation(12),
               T.ColorJitter(brightness=0.12, contrast=0.12)]
    else:
        ops = [T.Resize((image_size, image_size))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
