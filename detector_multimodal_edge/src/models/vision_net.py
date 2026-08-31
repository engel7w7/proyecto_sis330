import torch
import torch.nn as nn
from torchvision import models

class EfficientNetVision(nn.Module):
    """
    Modelo Experto de Visión basado en EfficientNet-B0 (Lite).
    Recibe rostros recortados 224x224 y detecta inconsistencias de síntesis (Real vs Deepfake).
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetVision, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        # Adaptar para salida binaria (BCEWithLogitsLoss logits)
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
