"""
Módulo de entrenamiento de modelos expertos (Audio y Visión) en PyTorch.
Gestiona el bucle de entrenamiento, métricas de desempeño y guardado del mejor modelo.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.config import LEARNING_RATE, EPOCHS, AUDIO_MODEL_PTH, VISION_MODEL_PTH, DATA_PROCESSED_DIR, MODELS_SAVED_DIR
from src.models.audio_net import MobileNetV3Audio
from src.models.vision_net import EfficientNetVision
from src.models.dataloaders import obtener_dataloaders

def calcular_metricas(preds_binary, targets):
    """Calcula las métricas de clasificación: Accuracy, Precision, Recall y F1-Score."""
    tp = ((preds_binary == 1) & (targets == 1)).sum().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()
    tn = ((preds_binary == 0) & (targets == 0)).sum().item()

    total = len(targets)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def entrenar_modelo_profesional(modelo, train_loader, val_loader, ruta_guardado, epochs=EPOCHS, lr=LEARNING_RATE, nombre_modelo="Experto"):
    """
    Entrena el modelo PyTorch utilizando AdamW y ReduceLROnPlateau.
    Aplica parada temprana (Early Stopping) si la pérdida de validación no mejora.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = modelo.to(device)
    
    criterio = nn.BCEWithLogitsLoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=2)

    mejor_loss_val = float('inf')
    historial_metricas = []
    patience = 5
    patience_counter = 0

    print(f"\nEntrenando {nombre_modelo} en dispositivo: {device}")

    for epoch in range(epochs):
        modelo.train()
        running_train_loss = 0.0

        if train_loader and len(train_loader) > 0:
            for imagenes, etiquetas in tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}"):
                imagenes, etiquetas = imagenes.to(device), etiquetas.to(device).unsqueeze(1)
                
                optimizador.zero_grad()
                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas.float())
                loss.backward()
                optimizador.step()
                
                running_train_loss += loss.item() * imagenes.size(0)

        train_loss = running_train_loss / max(1, len(train_loader.dataset) if train_loader else 1)

        val_loss = 0.0
        all_preds = []
        all_targets = []

        if val_loader and len(val_loader) > 0:
            modelo.eval()
            with torch.no_grad():
                for imagenes, etiquetas in val_loader:
                    imagenes, etiquetas = imagenes.to(device), etiquetas.to(device).unsqueeze(1)
                    salidas = modelo(imagenes)
                    loss = criterio(salidas, etiquetas.float())
                    val_loss += loss.item() * imagenes.size(0)

                    probs = torch.sigmoid(salidas)
                    preds = (probs > 0.5).long()
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(etiquetas.long().cpu().numpy().flatten())

            val_loss = val_loss / len(val_loader.dataset)
            metricas = calcular_metricas(np.array(all_preds), np.array(all_targets))

            print(f"[Validación Época {epoch+1:02d}/{epochs:02d}] "
                  f"Loss: {val_loss:.4f} | "
                  f"Acc: {metricas['accuracy']*100:.2f}% | "
                  f"Prec: {metricas['precision']*100:.2f}% | "
                  f"Recall: {metricas['recall']*100:.2f}% | "
                  f"F1: {metricas['f1_score']*100:.2f}%")

            scheduler.step(val_loss)

            metricas_epoch = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metricas["accuracy"],
                "precision": metricas["precision"],
                "recall": metricas["recall"],
                "f1_score": metricas["f1_score"]
            }
            historial_metricas.append(metricas_epoch)

            if val_loss < mejor_loss_val:
                mejor_loss_val = val_loss
                os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
                torch.save(modelo.state_dict(), ruta_guardado)
                print(f" -> Guardado nuevo mejor modelo en: {ruta_guardado}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" -> Detención temprana activada tras {patience} épocas sin mejora.")
                    break
        else:
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            torch.save(modelo.state_dict(), ruta_guardado)
            print(f" -> Modelo guardado en: {ruta_guardado}")
            break

    report_path = os.path.join(MODELS_SAVED_DIR, f"reporte_{nombre_modelo.lower().replace(' ', '_')}.json")
    with open(report_path, 'w') as f:
        json.dump(historial_metricas, f, indent=4)

    return historial_metricas

def entrenar_todo():
    """Lanza el entrenamiento completo para el experto de audio y de visión."""
    dir_audio = os.path.join(DATA_PROCESSED_DIR, 'audio')
    dir_vision = os.path.join(DATA_PROCESSED_DIR, 'vision')

    print("--- Entrenando Experto de Audio (MobileNetV3) ---")
    t_loader_a, v_loader_a = obtener_dataloaders(dir_audio)
    net_audio = MobileNetV3Audio()
    entrenar_modelo_profesional(net_audio, t_loader_a, v_loader_a, AUDIO_MODEL_PTH, nombre_modelo="Experto_Audio")

    print("--- Entrenando Experto de Visión (EfficientNet) ---")
    t_loader_v, v_loader_v = obtener_dataloaders(dir_vision)
    net_vision = EfficientNetVision()
    entrenar_modelo_profesional(net_vision, t_loader_v, v_loader_v, VISION_MODEL_PTH, nombre_modelo="Experto_Vision")

if __name__ == "__main__":
    entrenar_todo()

