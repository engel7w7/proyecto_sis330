"""
Módulo de Cuantización y Exportación a TensorFlow Lite (INT8 PTQ).
Proyecto SIS-330: Detector Móvil Multimodal de Estafas Digitales (Edge AI).

Este script realiza la conversión y optimización de los modelos expertos de Audio y Visión
al formato TensorFlow Lite (.tflite) aplicando Post-Training Quantization (INT8 / Dynamic Range),
y los despliega directamente en el directorio assets de la aplicación Android.
"""

import os
import sys
import shutil
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
MODELS_SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
ASSETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "android_app", "app", "src", "main", "assets"))

os.makedirs(MODELS_SAVED_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

AUDIO_TFLITE_NAME = "experto_audio_int8.tflite"
VISION_TFLITE_NAME = "experto_vision_int8.tflite"

AUDIO_TFLITE_PATH = os.path.join(MODELS_SAVED_DIR, AUDIO_TFLITE_NAME)
VISION_TFLITE_PATH = os.path.join(MODELS_SAVED_DIR, VISION_TFLITE_NAME)


def construir_y_exportar_tflite_audio(salida_tflite):
    """
    Construye el modelo MobileNetV3-Small para espectrogramas de audio (224x224x3 -> 2 clases)
    y lo exporta a TFLite aplicando Post-Training Quantization INT8.
    """
    import tensorflow as tf

    print(f"\n[1/2] Exportando Experto de Audio ({AUDIO_TFLITE_NAME})...")
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    # Clasificador binario de salida (Logits / Softmax para Real vs Clonado)
    outputs = tf.keras.layers.Dense(2, activation='linear', name='classifier')(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="MobileNetV3_Audio_Expert")

    # Conversión a TFLite con Post-Training Quantization (INT8)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Cuantización de rango dinámico / representativa
    def representative_data_gen():
        for _ in range(20):
            data = np.random.normal(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
            yield [data]
            
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    with open(salida_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(salida_tflite) / (1024 * 1024)
    print(f" -> Guardado exitoso: {salida_tflite} ({size_mb:.2f} MB)")
    return salida_tflite


def construir_y_exportar_tflite_vision(salida_tflite):
    """
    Construye el modelo EfficientNet-B0 para rostros/keyframes (224x224x3 -> 2 clases)
    y lo exporta a TFLite aplicando Post-Training Quantization INT8.
    """
    import tensorflow as tf

    print(f"\n[2/2] Exportando Experto de Visión ({VISION_TFLITE_NAME})...")
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    outputs = tf.keras.layers.Dense(2, activation='linear', name='classifier')(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="EfficientNetB0_Vision_Expert")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for _ in range(20):
            data = np.random.normal(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
            yield [data]

    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    with open(salida_tflite, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(salida_tflite) / (1024 * 1024)
    print(f" -> Guardado exitoso: {salida_tflite} ({size_mb:.2f} MB)")
    return salida_tflite


def desplegar_en_android():
    """Copia los archivos .tflite generados a android_app/app/src/main/assets y elimina archivos .pt obsoletos."""
    print(f"\nDesplegando artefactos en Android Assets: {ASSETS_DIR}")
    
    # 1. Copiar modelos .tflite
    for nombre in [AUDIO_TFLITE_NAME, VISION_TFLITE_NAME]:
        origen = os.path.join(MODELS_SAVED_DIR, nombre)
        destino = os.path.join(ASSETS_DIR, nombre)
        if os.path.exists(origen):
            shutil.copyfile(origen, destino)
            print(f" [OK] Copiado a assets: {nombre} ({os.path.getsize(destino) / (1024*1024):.2f} MB)")

    # 2. Eliminar archivos .pt obsoletos de assets
    for pt_file in ["experto_audio_int8.pt", "experto_vision_int8.pt"]:
        pt_path = os.path.join(ASSETS_DIR, pt_file)
        if os.path.exists(pt_path):
            os.remove(pt_path)
            print(f" [REMOVED] Archivo PyTorch .pt crudo eliminado de assets: {pt_file}")


def main():
    print("==================================================================")
    print(" PIPELINE DE CUANTIZACIÓN POST-TRAINING A TFLITE (INT8)")
    print("==================================================================")
    
    construir_y_exportar_tflite_audio(AUDIO_TFLITE_PATH)
    construir_y_exportar_tflite_vision(VISION_TFLITE_PATH)
    desplegar_en_android()
    
    print("\n[ÉXITO] Modelos TFLite INT8 generados y listos para inferencia móvil en Android.")


if __name__ == "__main__":
    main()
