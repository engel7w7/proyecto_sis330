# Detector Móvil Multimodal de Estafas Digitales mediante IA Explicable (XAI) y Procesamiento en el Borde (Edge AI)

**Universidad:** Universidad Mayor Real y Pontificia de San Francisco Xavier de Chuquisaca  
**Facultad:** Facultad de Ciencias y Tecnología  
**Carrera:** Ingeniería en Ciencias de la Computación  
**Materia:** SIS-330 (Desarrollo de Aplicaciones Inteligentes) - Grupo 1  
**Universitario:** Romero Morales Jhojan Erick  
**Docente:** Ing. Pacheco Lora Carlos Walter  
**Gestión:** 2026 | Sucre – Bolivia  

---

## Descripción del Proyecto

Sistema inteligente móvil para la detección y prevención en tiempo real de estafas digitales y fraudes audiovisuales (Deepfakes de audio y video) en canales de mensajería (WhatsApp, Telegram).

El sistema opera bajo un enfoque de **Edge AI** (inteligencia artificial ejecutada localmente en el dispositivo móvil sin enviar datos a servidores externos, garantizando privacidad absoluta) mediante **Fusión Tardía (Score-Level Fusion)** y ponderación multimodal calibrada para mitigación de falsos positivos, descarte automático ante ausencia de rostros humanos y descarte estricto de stickers/animaciones irrelevantes.

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
├── detector_multimodal_edge/           # Entorno Python (Preprocesamiento, Entrenamiento y MLOps)
│   ├── environment.yml                 # Dependencias Conda (PyTorch, TensorFlow, Librosa, OpenCV)
│   ├── preprocess.py                   # Script raíz para ejecución rápida del Pipeline ETL
│   ├── data/
│   │   └── processed/                  # Archivos procesados organizados para PyTorch DataLoaders
│   │       ├── audio/ (real / fake)    # Espectrogramas térmicos Magma PNG (224x224x3)
│   │       ├── vision/ (real / fake)   # Rostros recortados normalizados JPG (224x224x3)
│   │       └── manifest_resumen.json   # Manifiesto JSON con metadatos del procesamiento
│   ├── notebooks/
│   │   └── experimentos_resultados.ipynb # Matrices de Confusión, Métricas de Rendimiento y Curvas
│   └── src/
│       ├── config.py                   # Rutas y parámetros globales de entrenamiento
│       ├── preprocessing/              # Módulo ETL de preprocesamiento de datasets
│       │   └── preprocess_datasets.py  # Extracción, transformación STFT Magma y recorte facial
│       ├── models/                     # Arquitecturas MobileNetV3, EfficientNet y DataLoaders
│       │   ├── audio_net.py            # Backbone MobileNetV3-Small para espectrogramas
│       │   ├── vision_net.py           # Backbone EfficientNet-B0 para fotogramas faciales
│       │   ├── dataloaders.py          # PyTorch Dataset y DataLoader con división 80/20
│       │   └── saved/                  # Checkpoints PyTorch y modelos .tflite exportados
│       ├── training/
│       │   ├── train_expertos.py       # Entrenamiento en PyTorch con Early Stopping
│       │   └── evaluar_expertos.py     # Evaluación cuantitativa en conjuntos de prueba
│       └── export/
│           └── quantize_tflite.py      # Pipeline de Cuantización INT8 y despliegue a Android assets
│
├── datasets/                           # Datasets crudos de referencia (ASVspoof 2019, FaceForensics)
├── pruebas/                            # Reportes de métricas, matrices de confusión y capturas de pantalla
└── pruebas-nuevas/                     # Curvas ROC multimodales, cálculo de EER y capturas de Android Profiler
```

---

## Guía de Ejecución

### 1. Entorno Python y Experimentación (`detector_multimodal_edge/`)

El proyecto utiliza el entorno Conda `env_sis421` con PyTorch, TensorFlow Lite, OpenCV y Librosa/Soundfile.

1. **Activar el entorno:**
   ```bash
   conda activate env_sis421
   ```

2. **Pipeline ETL de Preprocesamiento de Datasets (`preprocess.py`):**
   Para transformar los datasets crudos en tensores periciales normalizados y listos para los DataLoaders de PyTorch:
   ```bash
   cd detector_multimodal_edge
   python preprocess.py --samples_per_modality 300
   ```
   * **Audio:** Ingesta archivos FLAC de ASVspoof 2019 (protocolo LA CM dev), normaliza a 16 kHz (ventana fija de 3.0 s / 48,000 muestras), calcula la STFT logarítmica y aplica el **mapa térmico cromático Magma RGB (224x224x3)**.
   * **Visión:** Ingesta secuencias MP4 de FaceForensics++ c23, extrae keyframes equidistantes, detecta rostros mediante Haar Cascade con un **margen de seguridad del 15%** y aplica **descarte pericial** ante la ausencia de rostro humano (224x224x3).
   * **Salida Estructurada:** Genera los archivos en `data/processed/audio/` y `data/processed/vision/`, además de emitir el manifiesto `data/processed/manifest_resumen.json`.

3. **Carga y Verificación con PyTorch DataLoaders:**
   Los archivos procesados son consumidos automáticamente por `src.models.dataloaders.MultimodalDeepfakeDataset`:
   ```bash
   python -c "from src.models.dataloaders import obtener_dataloaders; tl, vl = obtener_dataloaders('data/processed/audio', batch_size=16); print('DataLoader Audio listo:', len(tl))"
   ```

4. **Revisar el Notebook de Resultados y Métricas:**
   Abrir el notebook con Jupyter o en VS Code:
   ```bash
   jupyter notebook detector_multimodal_edge/notebooks/experimentos_resultados.ipynb
   ```
   En él se encuentran:
   * Las Matrices de Confusión de ambos modelos generadas con Seaborn y Matplotlib.
   * El cálculo formal de Accuracy, Precision, Recall y F1-Score.
   * La justificación en ciberseguridad sobre la relevancia de la Precisión frente a Falsos Positivos.

5. **Re-generar / Cuantizar los Modelos a TFLite INT8:**
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
* **Descarte Automático sin Rostro:** Si una imagen o fotograma analizado no contiene ningún rostro humano identificable (o es un meme/paisaje), el sistema descarta el análisis visual pericial para evitar falsos positivos.
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

* **Experto de Audio:** `MobileNetV3-Small INT8` (Entrada: Espectrograma térmico cromático Magma 224x224x3).
  * *Accuracy:* 81.79% | *Precision:* 78.28% | *Recall:* 99.96% | *F1-Score:* 87.80%
  * *Justificación del Mapa Térmico Magma:* Permite aprovechar el preentrenamiento de MobileNetV3 en ImageNet (3 canales RGB), resaltando armónicos de alta energía en amarillo brillante y formantes de voz en magenta frente al fondo acústico índigo oscuro.
* **Experto de Visión:** `EfficientNet-B0 INT8` (Entrada: Keyframe facial centrado con margen del 15% 224x224x3).
  * *Accuracy:* 97.17% | *Precision:* 95.40% | *Recall:* 96.25% | *F1-Score:* 95.82%
  * *Pipeline Biométrico:* Detección con Haar Cascade y recorte adaptativo con margen de contexto del 15%. Descarte automático cuando no se localiza fisonomía facial.

### 2. Lógica de Fusión (`RiskScorer.kt`)

* **Score-Level Fusion Ponderado:**
  $$\text{Score Global} = (0.55 \times P_{\text{audio}}) + (0.45 \times P_{\text{visión}})$$
* **Regla de 3/5 para Video:** Para la detección sostenida en fotogramas de video, basta con que la pista de audio sea clasificada como clonada o que la mayoría de los frames (al menos 3 de 5) presenten alteraciones visuales para emitir una alerta crítica de Deepfake.
* **Explicabilidad XAI (Grad-CAM):** Extracción de gradientes de la última capa convolucional para generar mapas térmicos cromáticos (Saliency Heatmaps) que justifican visualmente ante el usuario la detección de anomalías.
* **Rebalanceo Unimodal Dinámico:** Si el contenido recibido es únicamente audio o únicamente imagen, el peso del canal presente se ajusta automáticamente al 100% (1.0) para garantizar una evaluación precisa sin penalizaciones por canal ausente.
* **Umbrales de Categorización de Riesgo:**
  * $< 30.0\% \rightarrow$ **Riesgo Bajo (Contenido Auténtico)**
  * $30.0\% - 69.9\% \rightarrow$ **Riesgo Medio (Alerta Moderada)**
  * $\ge 70.0\% \rightarrow$ **Riesgo Alto (Posible Deepfake / Estafa)**
* **Regla de Descarte de Stickers y Contenido sin Rostro:** Los stickers (mensajes con contenido "sticker" o imágenes con MIME `image/webp`) y las imágenes sin rostro detectable son descartados antes de la inferencia pesada, ahorrando batería y CPU.
