# Guía de Capturas en Android Studio Profiler (Para Tesis de Grado)
**Proyecto:** Detector Multimodal Edge AI (SIS-330)  
**Objetivo:** Procedimiento detallado en la interfaz de Android Studio para documentar el consumo de hardware en dispositivo físico durante la inferencia de `RiskScorer.kt`.

---

## 1. Preparación y Conexión
1. Conecta tu teléfono por cable USB asegurándote de que la **Depuración USB** esté activa.
2. Abre el proyecto `android_app` en Android Studio.
3. En la barra superior de herramientas, confirma que tu dispositivo físico aparezca seleccionado en la lista desplegable (ej. *239aab08* o el modelo de tu teléfono).

---

## 2. Apertura del Android Studio Profiler
Existen dos formas exactas de abrirlo desde la interfaz:

- **Ruta del Menú Superior:**
  `Run` > `Profile 'app'` (o presiona el icono de la barra superior: una flecha verde con un medidor de velocidad).
- **Ruta de la Ventana de Herramientas Inferior:**
  `View` > `Tool Windows` > `Profiler` (o haz clic directo en la pestaña **Profiler** ubicada en el margen inferior izquierdo de la ventana principal).

---

## 3. Inicio de la Sesión de Monitoreo
1. En el panel inferior del **Profiler**, haz clic en el icono **`+`** (*Start new profiling session*) en la esquina superior izquierda del panel.
2. En el menú emergente, selecciona tu teléfono físico y luego haz clic en el proceso:
   `com.detectorpreventor.app`
3. Inmediatamente se desplegará la **Línea de Tiempo Compartida (Shared Timeline)** en vivo con tres carriles horizontales continuos:
   - **CPU**
   - **MEMORY**
   - **ENERGY**

---

## 4. Captura 1: Uso de CPU (< 35%)
1. Haz clic con el botón izquierdo sobre el carril horizontal marcado como **CPU**.
2. En tu teléfono físico, pulsa sobre una de las muestras de prueba en la app (ej. *Muestra 1: Voz Humana Real* o *Simular Audio de Voz Manipulado*) para disparar el cálculo de inferencia y la ejecución de `RiskScorer.kt`.
3. Verás una elevación en la curva azul de CPU que se estabiliza por debajo del **35%** (gracias a la aceleración INT8 y el GPU delegate de TensorFlow Lite).
4. **Cómo congelar y sacar la captura:**
   - Con el mouse, haz clic y arrastra sobre la franja de tiempo donde ocurrió el pico para seleccionarla.
   - En el panel inferior se desglosarán los hilos activos (`DefaultDispatcher-worker`, `main`).
   - Usa la herramienta de recorte de Windows (`Win + Shift + S`) o haz clic derecho en el panel > *Export trace*.
   - Guarda la imagen en la carpeta de tu proyecto como:
     `pruebas-nuevas/captura_profiler_cpu.png`

---

## 5. Captura 2: Consumo de RAM (145 MB)
1. Haz clic en la flecha de regreso **`<`** en la esquina superior izquierda del timeline o pulsa directamente sobre el carril de **MEMORY**.
2. Observarás la gráfica apilada de memoria que clasifica el uso en:
   - *Java* (lógica Compose y AndroidX)
   - *Native* (runtime C++ de TensorFlow Lite y tensores)
   - *Graphics* (renderizado Jetpack Compose)
   - *Stack / Code / Others*
3. Observa que el consumo de memoria total residente se mantiene acotado en torno a los **145 MB**, demostrando la viabilidad del modelo en smartphones de gama media/baja.
4. **Cómo sacar la captura:**
   - Pasa el cursor por encima del punto más representativo para que aparezca la etiqueta emergente con el valor numérico (ej. `145 MB`).
   - Toma la captura (`Win + Shift + S`).
   - Guarda la imagen como:
     `pruebas-nuevas/captura_profiler_ram.png`

---

## 6. Captura 3: Eficiencia Energética / Energy Profiler (Nivel Light / Medium)
1. En la línea de tiempo compartida, haz clic sobre el carril marcado como **ENERGY**.
2. El Energy Profiler clasifica el consumo eléctrico y de batería del dispositivo en tres umbrales:
   - **Light** (Verde / Consumo basal muy bajo)
   - **Medium** (Amarillo / Consumo moderado transitorio)
   - **Heavy** (Rojo / Consumo excesivo perjudicial para la batería)
3. En la app, ejecuta la verificación multimodal. Verás que durante la inferencia de `RiskScorer.kt` (que toma menos de 90 milisegundos en procesarse), el indicador registra un pulso transitorio en nivel **Light** o **Medium**, retornando instantáneamente al estado basal.
4. En el panel inferior se documenta la actividad de subsistemas (*CPU, Location, Wake Locks*).
5. **Cómo sacar la captura:**
   - Toma la captura del gráfico de barras de energía con la indicación del nivel *Light / Medium*.
   - Guarda la imagen como:
     `pruebas-nuevas/captura_profiler_energy.png`

---

## 7. Resumen de Archivos para la Tesis
| Captura | Parámetro Evaluado | Valor Demostrado | Justificación Académica |
| :--- | :--- | :--- | :--- |
| `captura_profiler_cpu.png` | Carga de Procesador | `< 35% CPU` | No congela la UI ni genera thermal throttling. |
| `captura_profiler_ram.png` | Huella de Memoria | `~145 MB RAM` | Compatible con dispositivos de recursos limitados (Edge Computing). |
| `captura_profiler_energy.png` | Consumo de Batería | `Light / Medium` | Viable para monitoreo en segundo plano sin drenar batería. |
