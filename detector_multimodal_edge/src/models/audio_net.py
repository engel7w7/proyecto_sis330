import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3Audio(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(MobileNetV3Audio, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
