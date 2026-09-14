# Detector Móvil Multimodal de Estafas Digitales (Edge AI)
**Materia:** SIS-330 (Taller de Grado / Sistemas Expertos)  
**Docente:** Ing. Pacheco  
**Universidad:** Universidad Autónoma Gabriel René Moreno (UAGRM)

---

## 📌 Descripción del Proyecto
Sistema inteligente móvil para la detección y prevención en tiempo real de estafas digitales y fraudes audiovisuales (Deepfakes de audio y video) en canales de mensajería (WhatsApp, Telegram).

El sistema opera bajo un enfoque de **Edge AI** (inteligencia artificial ejecutada localmente en el dispositivo móvil sin enviar datos a servidores externos, garantizando privacidad absoluta) mediante **Fusión Tardía (Score-Level Fusion)** y la **Regla 3/5** para mitigación de falsos positivos.

---

## 📂 Estructura del Repositorio

```
proyecto_sis330/
│
├── README.md                           # Instrucciones de ejecución y documentación
├── .gitignore                          # Exclusión de binarios temporales y datasets pesados
│
├── android_app/                        # Proyecto Kotlin Android Studio
│   ├── build.gradle.kts                # Configuración de build raíz
│   └── app/
│       ├── build.gradle.kts            # Configuración con namespace com.sis330.detector
│       └── src/main/
│           ├── AndroidManifest.xml     # Configurado con Share Intents (WhatsApp/Telegram)
│           ├── assets/
│           │   ├── experto_audio_int8.tflite  # Modelo MobileNetV3 cuantizado INT8
│           │   └── experto_vision_int8.tflite # Modelo EfficientNet-B0 cuantizado INT8
│           └── java/com/sis330/detector/
│               ├── MainActivity.kt     # UI Jetpack Compose en Modo Oscuro y alertas
│               ├── RiskScorer.kt       # Algoritmo de Score-Level Fusion + Regla 3/5
│               └── ml/
│                   ├── AudioClassifier.kt # Inferencia TFLite para espectrogramas Mel
│                   └── VisionClassifier.kt# Inferencia TFLite para keyframes faciales
│
└── model_training/                     # Entorno Python (Experimentación y MLOps)
    ├── environment.yml                 # Dependencias Conda (env_sis421)
    ├── data/                           # Directorio para datasets locales
    │   └── .keep
    ├── notebooks/
    │   └── experimentos_resultados.ipynb # Matrices de Confusión, Métricas y Justificación
    └── src/
        ├── train_expertos.py           # Código de entrenamiento en PyTorch
        └── quantize_tflite.py          # Pipeline de Cuantización INT8 y despliegue a Android
```

---

## 🚀 Guía de Ejecución

### 1. Entorno Python y Experimentación (`model_training/`)

El proyecto utiliza el entorno Conda `env_sis421` con PyTorch y TensorFlow Lite.

1. **Activar el entorno:**
   ```bash
   conda activate env_sis421
   ```

2. **Revisar el Notebook de Resultados y Métricas:**
   Abrir el notebook con Jupyter o en VS Code:
   ```bash
   jupyter notebook model_training/notebooks/experimentos_resultados.ipynb
   ```
   En él se encuentran:
   * Las **Matrices de Confusión** de ambos modelos graficadas con `seaborn` y `matplotlib`.
   * El cálculo e impresión formal de **Accuracy, Precision, Recall y F1-Score**.
   * La **justificación teórica en ciberseguridad** de por qué *Precision* es la métrica más crítica.

3. **Re-generar / Cuantizar los Modelos a TFLite INT8:**
   ```bash
   python model_training/src/quantize_tflite.py
   ```
   *Convierte y optimiza los modelos a formato `.tflite` (INT8) y los copia automáticamente a `android_app/app/src/main/assets/`.*

---

### 2. Aplicación Móvil Android (`android_app/`)

1. Abrir la carpeta `android_app` en **Android Studio** (versión Hedgehog o superior).
2. Dejar que Gradle sincronice las dependencias automáticamente (JDK 17).
3. Conectar un dispositivo Android o iniciar el emulador (Android 8.0 / API 26 o superior).
4. Presionar **Run 'app'** (`Shift + F10`).

#### Demostración en Vivo:
* **Intercepción automática:** Al recibir o compartir un audio (`.opus`, `.mp3`) o imagen/video desde WhatsApp hacia la app, se activa el análisis multimodal.
* **UI Modo Oscuro:** Muestra en pantalla el texto prominente: `"Riesgo de Estafa: [Score Global]%"`, el desglose de inferencia y los botones de mitigación:
  * 🔴 **"Bloquear y Reportar"** (Acción de peligro / aislamiento del contacto).
  * ⚪ **"Descartar Análisis"** (Confirmación de medio seguro).
  * 🔄 **"Alternar Escenario de Prueba"** (Permite simular en vivo casos seguros y casos de estafa para evaluación docente).

---

## 🧠 Algoritmos Implementados

### 1. Modelos Expertos Edge
* **Experto de Audio:** `MobileNetV3-Small INT8` (Entrada: Espectrograma Mel 224x224x3).
  * *Accuracy:* 81.79% | *Precision:* 78.28% | *Recall:* 99.96% | *F1-Score:* 87.80%
* **Experto de Visión:** `EfficientNet-B0 INT8` (Entrada: Keyframe facial 224x224x3).
  * *Accuracy:* 97.17% | *Precision:* 95.40% | *Recall:* 96.25% | *F1-Score:* 95.82%

### 2. Lógica de Fusión (`RiskScorer.kt`)
* **Regla 3/5:** Exige que al menos 3 de los 5 keyframes analizados superen el umbral de fraude para validar manipulación facial, evitando falsos positivos causados por iluminación o compresión.
* **Score-Level Fusion:**
  $$\text{Score Global} = (0.55 \times P_{\text{audio}}) + (0.45 \times P_{\text{visión}})$$
* **Veredicto:**
  * $< 35\% \rightarrow$ **Seguro**
  * $35\% - 70\% \rightarrow$ **Sospechoso**
  * $> 70\% \rightarrow$ **Estafa Inminente**
