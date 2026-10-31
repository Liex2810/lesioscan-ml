import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import onnx
import onnxruntime as ort
from dermascan.models import build
from dermascan.engine import save_json
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='deployment/model.onnx');a=p.parse_args()
ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
m=build(ckpt['model'],len(ckpt['labels']),pretrained=False)
m.load_state_dict(ckpt['state_dict']);m.eval()
x=torch.randn(1,3,ckpt['image_size'],ckpt['image_size'])
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with torch.no_grad():
    torch.onnx.export(m,x,str(out),input_names=['image'],output_names=['logits'],opset_version=17,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}})
    torch_out=m(x).numpy()
onnx.checker.check_model(onnx.load(str(out)))
s=ort.InferenceSession(str(out),providers=['CPUExecutionProvider'])
ort_out=s.run(None,{'image':x.numpy()})[0]
np.testing.assert_allclose(torch_out,ort_out,atol=1e-4,rtol=1e-3)
save_json(out.parent/'labels.json',{str(i):label for i,label in enumerate(ckpt['labels'])})
save_json(out.parent/'preprocessing.json',{'width':ckpt['image_size'],'height':ckpt['image_size'],'input_layout':'NCHW','channel_order':'RGB','pixel_range':'0..1','mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225],'output':'logits (softmax not embedded)'})
print('ONNX parity PASS:',out)
