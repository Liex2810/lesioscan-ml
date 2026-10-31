import torch.nn as nn
import torchvision.models as tv

ALLOWED = ('shufflenet_v2_x1_0', 'densenet121', 'vit_small_patch16_224')

def build(name, n_classes=6, pretrained=True):
    if name == 'shufflenet_v2_x1_0':
        m = tv.shufflenet_v2_x1_0(weights=tv.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == 'densenet121':
        m = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    elif name == 'vit_small_patch16_224':
        import timm
        m = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=n_classes)
    else:
        raise ValueError(f'Unknown model {name}; choose {ALLOWED}')
    return m


def freeze_backbone(m, name):
    for p in m.parameters():
        p.requires_grad = False
    head = m.fc if name == 'shufflenet_v2_x1_0' else m.classifier if name == 'densenet121' else m.get_classifier()
    for p in head.parameters():
        p.requires_grad = True
