"""Stage 1: freeze pretrained backbone and train only the classification head."""
from .models import freeze_backbone

def configure_feature_extraction(model, model_name: str):
    freeze_backbone(model, model_name)
    return [p for p in model.parameters() if p.requires_grad]
