import argparse  # Memuat argparse untuk fungsi yang digunakan pada file ini.
from pathlib import Path  # Memuat pathlib untuk fungsi yang digunakan pada file ini.
import sys  # Memuat sys untuk fungsi yang digunakan pada file ini.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))  # Menambahkan folder src agar package dapat diimpor saat script dijalankan.
import numpy as np  # Memuat numpy untuk fungsi yang digunakan pada file ini.
import torch  # Memuat torch untuk fungsi yang digunakan pada file ini.
import onnx  # Memuat onnx untuk fungsi yang digunakan pada file ini.
import onnxruntime as ort  # Memuat onnxruntime untuk fungsi yang digunakan pada file ini.
from lesioscan.models import build  # Memuat lesioscan.models untuk fungsi yang digunakan pada file ini.
from lesioscan.engine import save_json  # Memuat lesioscan.engine untuk fungsi yang digunakan pada file ini.
p = argparse.ArgumentParser()  # Menetapkan parser argumen atau parameter model.
p.add_argument('--checkpoint', required=True)  # Menjalankan p.add_argument untuk melanjutkan proses.
p.add_argument('--out', default='deployment/model.onnx')  # Menjalankan p.add_argument untuk melanjutkan proses.
a = p.parse_args()  # Menetapkan argumen dari pengguna.
ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)  # Menetapkan checkpoint model.
m = build(ckpt['model'], len(ckpt['labels']), pretrained=False)  # Menetapkan model klasifikasi.
m.load_state_dict(ckpt['state_dict'])  # Memuat bobot checkpoint ke arsitektur model.
m.eval()  # Mengaktifkan mode evaluasi model.
x = torch.randn(1, 3, ckpt['image_size'], ckpt['image_size'])  # Menetapkan tensor masukan.
out = Path(a.out)  # Menetapkan hasil pembagian atau path keluaran.
out.parent.mkdir(parents=True, exist_ok=True)  # Menjalankan out.parent.mkdir untuk melanjutkan proses.
with torch.no_grad():  # Menonaktifkan pencatatan gradien selama ekspor dan inferensi referensi.
    torch.onnx.export(m, x, str(out), input_names=['image'], output_names=['logits'], opset_version=17, dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}})  # Mengekspor graph dan bobot model ke ONNX.
    torch_out = m(x).numpy()  # Menetapkan keluaran referensi PyTorch.
onnx.checker.check_model(onnx.load(str(out)))  # Memvalidasi struktur graph ONNX hasil ekspor.
s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])  # Menetapkan sesi inferensi ONNX.
ort_out = s.run(None, {'image': x.numpy()})[0]  # Menetapkan keluaran ONNX Runtime.
np.testing.assert_allclose(torch_out, ort_out, atol=0.0001, rtol=0.001)  # Menguji kesetaraan numerik hasil PyTorch dan ONNX.
save_json(out.parent / 'labels.json', {str(i): label for i, label in enumerate(ckpt['labels'])})  # Menyimpan hasil terstruktur sebagai JSON.
save_json(out.parent / 'preprocessing.json', {'width': ckpt['image_size'], 'height': ckpt['image_size'], 'input_layout': 'NCHW', 'channel_order': 'RGB', 'pixel_range': '0..1', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'output': 'logits (softmax not embedded)'})  # Menyimpan hasil terstruktur sebagai JSON.
print('ONNX parity PASS:', out)  # Menampilkan informasi proses atau hasil kepada pengguna.
