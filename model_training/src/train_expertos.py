"""
Módulo de Entrenamiento de Modelos Expertos (Audio y Visión) en PyTorch.
Proyecto SIS-330: Detector Móvil Multimodal de Estafas Digitales (Edge AI).

Arquitecturas:
- Experto de Audio: MobileNetV3-Small (clasificación binaria sobre espectrogramas Mel).
- Experto de Visión: EfficientNet-B0 (clasificación binaria sobre keyframes faciales).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Directorios de referencia
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(MODELS_SAVED_DIR, exist_ok=True)

# Hiperparámetros base
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 224


class MobileNetV3Audio(nn.Module):
    """
    Experto de Audio basado en MobileNetV3-Small.
    Entrada: Espectrogramas Mel normalizados de 224x224x3.
    Salida: Logit binario (0 = Real / Auténtico, 1 = Clonado / Deepfake).
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(MobileNetV3Audio, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetVision(nn.Module):
    """
    Experto de Visión basado en EfficientNet-B0.
    Entrada: Recortes faciales de 224x224x3.
    Salida: Logit binario (0 = Real / Auténtico, 1 = Sintetizado / Deepfake).
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetVision, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def calcular_metricas(preds_binary, targets):
    """
    Calcula formalmente las métricas de rendimiento en ciberseguridad:
    - Accuracy: Exactitud global.
    - Precision: TP / (TP + FP) -> Métrica crítica para evitar falsas alarmas.
    - Recall: TP / (TP + FN) -> Sensibilidad de captura de ataques.
    - F1-Score: Media armónica entre Precision y Recall.
    """
    tp = int(((preds_binary == 1) & (targets == 1)).sum())
    fp = int(((preds_binary == 1) & (targets == 0)).sum())
    fn = int(((preds_binary == 0) & (targets == 1)).sum())
    tn = int(((preds_binary == 0) & (targets == 0)).sum())

    total = len(targets)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def guardar_checkpoint(modelo, ruta):
    """Guarda el estado del modelo entrenado."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    torch.save(modelo.state_dict(), ruta)
    print(f"[OK] Checkpoint guardado en: {ruta}")


if __name__ == "__main__":
    print("Módulo de entrenamiento listo para el entorno PyTorch (env_sis421).")
    print(f"Directorio de modelos: {MODELS_SAVED_DIR}")
