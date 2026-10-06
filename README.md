# Detector Móvil Multimodal de Estafas Digitales (Edge AI)

**Materia:** SIS-330 (Desarrollo de Aplicaciones Inteligentes)  
**Docente:** Ing. Pacheco Lora Carlos Walter  
**Universidad:** Universidad San Francisco Xavier de Chuquisaca  

---

## Descripción del Proyecto

Sistema inteligente móvil para la detección y prevención en tiempo real de estafas digitales y fraudes audiovisuales (Deepfakes de audio y video) en canales de mensajería (WhatsApp, Telegram).

El sistema opera bajo un enfoque de **Edge AI** (inteligencia artificial ejecutada localmente en el dispositivo móvil sin enviar datos a servidores externos, garantizando privacidad absoluta) mediante **Fusión Tardía (Score-Level Fusion)** y ponderación multimodal calibrada para mitigación de falsos positivos y descarte estricto de stickers/animaciones irrelevantes.

---

## Estructura del Repositorio

```
proyecto_sis330/
│
├── README.md                           # Instrucciones de ejecución y documentación del proyecto
├── .gitignore                          # Exclusión de binarios temporales, builds y datasets pesados
│
├── android_app/                        # Proyecto Kotlin Android Studio (Compose + TFLite)
│   ├── build.gradle.kts                # Configuración de build raíz de Gradle
│   ├── gradle.properties               # Propiedades del entorno Gradle y JVM
│   ├── settings.gradle.kts             # Módulos y repositorios de dependencias
│   └── app/
│       ├── build.gradle.kts            # Dependencias (TFLite INT8, GPU Delegate, Compose, JUnit 5)
│       └── src/
│           ├── main/
│           │   ├── AndroidManifest.xml # Configurado con Share Intents y NotificationListenerService
│           │   ├── assets/
│           │   │   ├── experto_audio_int8.tflite   # Modelo MobileNetV3 cuantizado INT8 (Audio STFT)
│           │   │   ├── experto_vision_int8.tflite  # Modelo EfficientNet-B0 cuantizado INT8 (Visión)
│           │   │   └── samples/                    # Muestras reales/fake para pruebas y benchmark
│           │   └── java/com/detectorpreventor/app/
│           │       ├── DetectorApp.kt              # Inicializador de la aplicación
│           │       ├── domain/                     # MediaRouter y RiskScorer (Fusión y Reglas)
│           │       ├── ml/                         # AudioClassifier y VisionClassifier (TFLite)
│           │       ├── notifications/              # NotificationMonitorService y NotificationRepository
│           │       └── ui/                         # DetectorScreen, HeatmapOverlay, RiskIndicatorCard, MainActivity
│           │           └── theme/                  # Theme.kt (Paleta Dark Cyber Defense)
│           └── test/java/com/detectorpreventor/app/
│               ├── domain/                         # Pruebas unitarias de Fusión Tardía y descarte de stickers
│               └── notifications/                  # Pruebas de intercepción, serialización y deduplicación
│
├── detector_multimodal_edge/           # Entorno Python (Entrenamiento, Cuantización y MLOps)
│   ├── environment.yml                 # Dependencias Conda (PyTorch, TensorFlow, Librosa, ONNX)
│   ├── data/                           # Directorio para datasets locales procesados y crudos
│   ├── notebooks/
│   │   └── experimentos_resultados.ipynb # Matrices de Confusión, Métricas de Rendimiento y Curvas
│   └── src/
│       ├── config.py                   # Rutas y parámetros globales de entrenamiento
│       ├── models/                     # Arquitecturas MobileNetV3, EfficientNet y DataLoaders
│       │   └── saved/                  # Checkpoints PyTorch y modelos .tflite exportados
│       ├── training/
│       │   ├── train_expertos.py       # Entrenamiento en PyTorch con Early Stopping
│       │   └── evaluar_expertos.py     # Evaluación cuantitativa en conjuntos de prueba
│       └── export/
│           └── quantize_tflite.py      # Pipeline de Cuantización INT8 y despliegue a Android assets
│
├── pruebas/                            # Reportes de métricas, matrices de confusión y capturas de pantalla
└── pruebas-nuevas/                     # Curvas ROC multimodales, cálculo de EER y capturas de Android Profiler
```

---

## Guía de Ejecución

### 1. Entorno Python y Experimentación (`detector_multimodal_edge/`)

El proyecto utiliza el entorno Conda `env_sis421` con PyTorch y TensorFlow Lite.

1. **Activar el entorno:**
   ```bash
   conda activate env_sis421
   ```

2. **Revisar el Notebook de Resultados y Métricas:**
   Abrir el notebook con Jupyter o en VS Code:
   ```bash
   jupyter notebook detector_multimodal_edge/notebooks/experimentos_resultados.ipynb
   ```
   En él se encuentran:
   * Las Matrices de Confusión de ambos modelos generadas con Seaborn y Matplotlib.
   * El cálculo formal de Accuracy, Precision, Recall y F1-Score.
   * La justificación en ciberseguridad sobre la relevancia de la Precisión frente a Falsos Positivos.

3. **Re-generar / Cuantizar los Modelos a TFLite INT8:**
   ```bash
   python detector_multimodal_edge/src/export/quantize_tflite.py
   ```
   *Optimiza los modelos a formato `.tflite` (INT8) y los transfiere automáticamente a `android_app/app/src/main/assets/`.*

---

### 2. Aplicación Móvil Android (`android_app/`)

1. Abrir la carpeta `android_app` en **Android Studio** (versión Hedgehog o superior).
2. Sincronizar las dependencias de Gradle con JDK 17.
3. Conectar un dispositivo físico o iniciar el emulador con Android 8.0 (API 26) o superior.
4. Ejecutar la aplicación en modo Debug (`Shift + F10`).

#### Vistas y Funcionalidades de la App

* **Escáner Multimodal:** Visualización del resultado de inferencia, veredicto con porcentaje global de riesgo, desglose unimodal (Audio y Visión) y mapa de calor explicativo Grad-CAM con selector de alternancia.
* **Carga de Archivos:** Permite seleccionar audios, imágenes o videos del almacenamiento para su análisis instantáneo.
* **Monitoreo de WhatsApp / Telegram:** Intercepción en segundo plano mediante `NotificationListenerService`, con filtrado automático de mensajes del sistema y descarte estricto de stickers (archivos `.webp` o mensajes etiquetados como stickers).
* **Dataset / Benchmark:** Banco de muestras reales y deepfakes precargadas para demostraciones académicas e inferencia en vivo.
* **Información del Sistema (MLOps):** Resumen de arquitecturas, cuantización INT8 y parámetros del pipeline.

#### Ejecución de Pruebas Unitarias

Para ejecutar el conjunto de pruebas unitarias automáticas desde la terminal:
```bash
cd android_app
./gradlew test
```

---

## Algoritmos y Modelos Implementados

### 1. Modelos Expertos Edge

* **Experto de Audio:** `MobileNetV3-Small INT8` (Entrada: Espectrograma Mel 224x224x3).
  * *Accuracy:* 81.79% | *Precision:* 78.28% | *Recall:* 99.96% | *F1-Score:* 87.80%
* **Experto de Visión:** `EfficientNet-B0 INT8` (Entrada: Keyframe facial 224x224x3).
  * *Accuracy:* 97.17% | *Precision:* 95.40% | *Recall:* 96.25% | *F1-Score:* 95.82%

### 2. Lógica de Fusión (`RiskScorer.kt`)

* **Score-Level Fusion Ponderado:**
  $$\text{Score Global} = (0.60 \times P_{\text{audio}}) + (0.40 \times P_{\text{visión}})$$
* **Rebalanceo Unimodal Dinámico:** Si el contenido recibido es únicamente audio o únicamente imagen, el peso del canal presente se ajusta automáticamente al 100% (1.0) para garantizar una evaluación precisa sin penalizaciones por canal ausente.
* **Umbrales de Categorización de Riesgo:**
  * $< 30.0\% \rightarrow$ **Riesgo Bajo (Contenido Auténtico)**
  * $30.0\% - 69.9\% \rightarrow$ **Riesgo Medio (Alerta Moderada)**
  * $\ge 70.0\% \rightarrow$ **Riesgo Alto (Posible Deepfake / Estafa)**
* **Regla de Descarte de Stickers:** Los stickers (mensajes con contenido "sticker" o imágenes con MIME `image/webp`) son descartados de inmediato sin ejecutar inferencia neuronal, optimizando el consumo de batería y evitando falsos positivos.
