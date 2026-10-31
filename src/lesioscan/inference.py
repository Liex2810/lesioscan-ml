"""Load checkpoint and produce raw class scores; no clinical interpretation."""
from pathlib import Path
import torch
from PIL import Image
from .models import build
from .preprocessing import build_transforms


def load_checkpoint(path,device='cpu'):
    # Only open your own trusted checkpoint: torch.load can deserialize Python objects.
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=build(ckpt['model'],len(ckpt['labels']),pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model,ckpt


def predict_image(model,checkpoint,image_path,device='cpu'):
    with Image.open(image_path) as source:
        img=source.convert('RGB')
    tensor=build_transforms(checkpoint['image_size'],training=False)(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits=model(tensor)[0]
        scores=torch.softmax(logits,dim=0).cpu().tolist()
    return [{'label':name,'score':float(score)} for name,score in
            sorted(zip(checkpoint['labels'],scores),key=lambda p:p[1],reverse=True)]
