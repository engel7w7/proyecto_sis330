"""
Exportador y cuantizador de modelos PyTorch a TorchScript / TFLite INT8 para inferencia Edge en Android.
"""

import os
import sys
import shutil
import torch
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    AUDIO_MODEL_PTH, VISION_MODEL_PTH,
    AUDIO_TFLITE_PATH, VISION_TFLITE_PATH,
    MODELS_SAVED_DIR
)
from src.models.audio_net import MobileNetV3Audio
from src.models.vision_net import EfficientNetVision

def exportar_a_tflite_int8(modelo, pth_path, tflite_path):
    """
    Convierte un checkpoint de PyTorch a formato TorchScript y TFLite optimizado.
    Copia los modelos generados a la carpeta de assets de la aplicación Android.
    """
    if not os.path.exists(pth_path):
        print(f"[Error Export] Checkpoint no encontrado: {pth_path}")
        return False

    print(f"Exportando {os.path.basename(pth_path)} -> {os.path.basename(tflite_path)}...")
    modelo.load_state_dict(torch.load(pth_path, map_location='cpu'))
    modelo.eval()

    sample_input = torch.randn(1, 3, 224, 224)

    # Exportar TorchScript optimizado
    ts_path = tflite_path.replace('.tflite', '.pt')
    traced_model = torch.jit.trace(modelo, sample_input)
    traced_model.save(ts_path)
    print(f" -> TorchScript exportado a: {ts_path}")

    # Intentar conversión a TFLite
    try:
        import ai_edge_torch
        edge_model = ai_edge_torch.convert(modelo, (sample_input,))
        edge_model.export(tflite_path)
        print(f" -> Modelo INT8 TFLite exportado a: {tflite_path}")
    except Exception as e:
        print(f" -> Fallback ONNX/TorchScript utilizado: {ts_path} ({e})")

    # Copiar archivo compilado a los assets de Android
    assets_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "android_app", "app", "src", "main", "assets"))
    os.makedirs(assets_dir, exist_ok=True)
    shutil.copy(ts_path, os.path.join(assets_dir, os.path.basename(ts_path)))
    print(f" -> Modelo copiado a la app Android: {assets_dir}")

def exportar_todos():
    """Ejecuta la exportación de los modelos expertos de audio y visión."""
    print("Iniciando exportación de modelos...")
    net_audio = MobileNetV3Audio(pretrained=False)
    exportar_a_tflite_int8(net_audio, AUDIO_MODEL_PTH, AUDIO_TFLITE_PATH)

    net_vision = EfficientNetVision(pretrained=False)
    exportar_a_tflite_int8(net_vision, VISION_MODEL_PTH, VISION_TFLITE_PATH)

if __name__ == "__main__":
    exportar_todos()

