from pathlib import Path
import numpy as np
from PIL import Image
import torch
from lesioscan.inference import predict_image
from lesioscan.preprocessing import build_transforms

class Dummy(torch.nn.Module):
    def forward(self,x):
        return torch.tensor([[0.0,1.0,2.0]]).expand(x.shape[0],-1)

def test_predict_returns_ordered_scores(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(64,48),(200,120,20)).save(image)
    scores=predict_image(Dummy(),{'image_size':32,'labels':['A','B','C']},image)
    assert [s['label'] for s in scores]==['C','B','A']
    assert abs(sum(s['score'] for s in scores)-1)<1e-6

def test_deterministic_validation_transform():
    image=Image.new('RGB',(49,70),(100,30,200))
    f=build_transforms(32,training=False)
    a,b=f(image),f(image)
    assert a.shape==(3,32,32)
    assert torch.equal(a,b)
