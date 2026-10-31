"""Stage 2: unfreeze the pretrained network with a lower learning rate."""

def configure_fine_tuning(model):
    for parameter in model.parameters():
        parameter.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]
