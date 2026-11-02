# Historial de Versiones de Modelos - SIS 330

Este directorio resguarda las versiones históricas y optimizadas de los modelos clasificadores en formato TensorFlow Lite INT8 para el proyecto **Detector Móvil Multimodal de Estafas Digitales mediante IA Explicable (XAI) y Procesamiento en el Borde (Edge AI)**.

---

## 1. Estructura de Directorios

* `app-model/`: Modelos empaquetados directamente en los activos de la aplicación móvil (`android_app/app/src/main/assets/`).
  * `v1/`: Modelos iniciales de la versión 1.
    * `experto_audio_int8.tflite` (SHA256: `DF821037...`)
    * `experto_vision_int8.tflite` (SHA256: `AF8BD3FB...`)
  * `v2/`: Modelos optimizados con calibración Platt INT8, preservación de canal Grad-CAM `[1, 7, 7, 1280]` y mitigación de degradación por compresión.
    * `experto_audio_int8_v2.tflite` (SHA256: `FD7022DA...`)
    * `experto_vision_int8_v2.tflite` (SHA256: `0EF0283A...`)

* `model/`: Modelos de respaldo y exportación de la canalización de entrenamiento (`detector_multimodal_edge/src/models/saved/`).
  * `v1/`: Exportaciones iniciales de la canalización de entrenamiento V1.
    * `experto_audio_int8.tflite` (SHA256: `0ACEF4A8...`)
    * `experto_vision_int8.tflite` (SHA256: `A3F20B20...`)
  * `v2/`: Exportaciones optimizadas con calibración de logits V2.
    * `experto_audio_int8_v2.tflite` (SHA256: `FD7022DA...`)
    * `experto_vision_int8_v2.tflite` (SHA256: `0EF0283A...`)

---

## 2. Resumen Comparativo de Rendimiento

| Métrica / Aspecto | Versión 1 (V1) | Versión 2 (V2 - Activa) |
|---|---|---|
| **Precisión Audio (ASVspoof)** | ~88.4% | **98.62%** (Benchmark) / **93.21%** (LA Eval) |
| **Precisión Visión (Facial)** | ~89.1% | **100.00%** (Rostros LFW, FaceForensics, SIS330) |
| **Calibración INT8** | Comprimida (<0.35) | Platt calibrado (Frontera en 0.50 exacto) |
| **Canal XAI Grad-CAM** | Preservado | Preservado `[1, 7, 7, 1280]` |
| **Edge AI (RAM / Latencia)** | <200 MB / ~30 ms | <200 MB / ~28 ms |

