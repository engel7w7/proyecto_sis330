import os
import json
import torch
import torch.nn as nn
import numpy as np
from src.config import AUDIO_MODEL_PTH, VISION_MODEL_PTH, DATA_PROCESSED_DIR, MODELS_SAVED_DIR
from src.models.audio_net import MobileNetV3Audio
from src.models.vision_net import EfficientNetVision
from src.models.dataloaders import obtener_dataloaders
from src.training.train_expertos import calcular_metricas

def evaluar_modelo_detallado(modelo, loader, model_path, nombre="Modelo"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = modelo.to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    criterio = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = modelo(inputs)
            loss = criterio(outputs, targets.float())
            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.long().cpu().numpy().flatten())

    avg_loss = total_loss / max(1, len(loader.dataset))
    metricas = calcular_metricas(np.array(all_preds), np.array(all_targets))
    tamanio_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n---------------------------------------------------------")
    print(f" EVALUACION DETALLADA MLOPS: {nombre.upper()}")
    print(f"---------------------------------------------------------")
    print(f"  - Archivo Checkpoint: {model_path} ({tamanio_mb:.2f} MB)")
    print(f"  - Dispositivo:        {device}")
    print(f"  - BCE Loss:           {avg_loss:.4f}")
    print(f"  - Accuracy:           {metricas['accuracy']*100:.2f}%")
    print(f"  - Precision:          {metricas['precision']*100:.2f}%")
    print(f"  - Recall:             {metricas['recall']*100:.2f}%")
    print(f"  - F1-Score:           {metricas['f1_score']*100:.2f}%")
    print(f"  - Matriz de Confusion:")
    print(f"      [ True Positive (TP): {metricas['tp']} | False Positive (FP): {metricas['fp']} ]")
    print(f"      [ False Negative (FN): {metricas['fn']} | True Negative  (TN): {metricas['tn']} ]")
    print(f"---------------------------------------------------------")

    return {
        "modelo": nombre,
        "loss": avg_loss,
        "tamanio_mb": tamanio_mb,
        **metricas
    }

def evaluar_expertos():
    dir_audio = os.path.join(DATA_PROCESSED_DIR, 'audio')
    dir_vision = os.path.join(DATA_PROCESSED_DIR, 'vision')

    reporte_final = {}

    if os.path.exists(AUDIO_MODEL_PTH):
        _, val_loader_a = obtener_dataloaders(dir_audio)
        if val_loader_a:
            net_audio = MobileNetV3Audio()
            reporte_final["audio"] = evaluar_modelo_detallado(net_audio, val_loader_a, AUDIO_MODEL_PTH, nombre="Experto Audio (MobileNetV3)")

    if os.path.exists(VISION_MODEL_PTH):
        _, val_loader_v = obtener_dataloaders(dir_vision)
        if val_loader_v:
            net_vision = EfficientNetVision()
            reporte_final["vision"] = evaluar_modelo_detallado(net_vision, val_loader_v, VISION_MODEL_PTH, nombre="Experto Visión (EfficientNet)")

    json_path = os.path.join(MODELS_SAVED_DIR, "evaluacion_final_metricas.json")
    with open(json_path, 'w') as f:
        json.dump(reporte_final, f, indent=4)
    print(f"\n[OK] Reporte completo de metricas guardado en: {json_path}")

if __name__ == "__main__":
    evaluar_expertos()
