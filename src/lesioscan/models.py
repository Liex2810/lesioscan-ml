import torch.nn as nn  # Memuat torch.nn untuk fungsi yang digunakan pada file ini.
import torchvision.models as tv  # Memuat torchvision.models untuk fungsi yang digunakan pada file ini.
import timm  # Memuat timm untuk fungsi yang digunakan pada file ini.
ALLOWED = ('shufflenet_v2_x1_0', 'densenet121', 'vit_small_patch16_224')  # Menetapkan nilai ALLOWED.

def build(name, n_classes=6, pretrained=True):  # Membangun arsitektur dengan kepala klasifikasi sesuai jumlah kelas.
    if name == 'shufflenet_v2_x1_0':  # Memeriksa kondisi: name == 'shufflenet_v2_x1_0'.
        m = tv.shufflenet_v2_x1_0(weights=tv.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)  # Menetapkan model klasifikasi.
        m.fc = nn.Linear(m.fc.in_features, n_classes)  # Menetapkan nilai m.fc.
    elif name == 'densenet121':  # Memeriksa kondisi: name == 'densenet121'.
        m = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT if pretrained else None)  # Menetapkan model klasifikasi.
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)  # Menetapkan nilai m.classifier.
    elif name == 'vit_small_patch16_224':  # Memeriksa kondisi: name == 'vit_small_patch16_224'.
        m = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=n_classes)  # Menetapkan model klasifikasi.
    else:  # Menjalankan jalur alternatif saat kondisi sebelumnya tidak terpenuhi.
        raise ValueError(f'Unknown model {name}; choose {ALLOWED}')  # Menghentikan proses ketika kondisi input atau hasil tidak valid.
    return m  # Mengembalikan arsitektur dengan kepala klasifikasi yang sudah disesuaikan.

def freeze_backbone(m, name):  # Membekukan backbone dan membuka gradien kepala klasifikasi.
    for p in m.parameters():  # Mengulangi langkah untuk p.
        p.requires_grad = False  # Menonaktifkan pembaruan gradien parameter backbone.
    head = m.fc if name == 'shufflenet_v2_x1_0' else m.classifier if name == 'densenet121' else m.get_classifier()  # Menetapkan kepala klasifikasi yang akan dilatih.
    for p in head.parameters():  # Mengulangi langkah untuk p.
        p.requires_grad = True  # Mengaktifkan gradien parameter kepala klasifikasi.
