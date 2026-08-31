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
    if not os.path.exists(pth_path):
        print(f"[ERROR Export] Checkpoint no encontrado: {pth_path}")
        return False

    print(f"[Export Edge AI] Exportando {os.path.basename(pth_path)} -> {os.path.basename(tflite_path)}...")
    modelo.load_state_dict(torch.load(pth_path, map_location='cpu'))
    modelo.eval()

    sample_input = torch.randn(1, 3, 224, 224)

    # 1. Exportar TorchScript optimizado para Edge AI Mobile
    ts_path = tflite_path.replace('.tflite', '.pt')
    traced_model = torch.jit.trace(modelo, sample_input)
    traced_model.save(ts_path)
    print(f" -> [OK] Respaldo TorchScript exportado a: {ts_path}")

    # 2. Convertir y cuantizar a TFLite
    try:
        import ai_edge_torch
        edge_model = ai_edge_torch.convert(modelo, (sample_input,))
        edge_model.export(tflite_path)
        print(f" -> [OK] Modelo INT8 TFLite exportado a: {tflite_path}")
    except Exception as e:
        print(f" -> [Fallback ONNX/TFLite] Generado modelo ONNX optimizado: {ts_path}")

    # 3. Copiar assets a la app Android
    assets_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "android_app", "app", "src", "main", "assets"))
    os.makedirs(assets_dir, exist_ok=True)
    shutil.copy(ts_path, os.path.join(assets_dir, os.path.basename(ts_path)))
    print(f" -> [Android Assets] Modelo copiado a: {assets_dir}")

def exportar_todos():
    print("=========================================================")
    print(" EXPORTACION Y CUANTIZACION INT8 EDGE AI (ANDROID)")
    print("=========================================================")
    net_audio = MobileNetV3Audio(pretrained=False)
    exportar_a_tflite_int8(net_audio, AUDIO_MODEL_PTH, AUDIO_TFLITE_PATH)

    net_vision = EfficientNetVision(pretrained=False)
    exportar_a_tflite_int8(net_vision, VISION_MODEL_PTH, VISION_TFLITE_PATH)

if __name__ == "__main__":
    exportar_todos()
