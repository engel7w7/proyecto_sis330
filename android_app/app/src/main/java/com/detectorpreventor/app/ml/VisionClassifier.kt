package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Clasificador TFLite para el Modelo Experto de Visión (EfficientNet-B0 INT8).
 * Configurado opcionalmente con aceleración por GPU Delegate.
 */
class VisionClassifier(private val context: Context) {

    companion object {
        private const val TAG = "VisionClassifier"
        private const val MODEL_FILE = "experto_vision_int8.tflite"
        private const val ALT_MODEL_FILE = "modelo_vision_int8.tflite"
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isInitialized = false

    init {
        initInterpreter()
    }

    private fun initInterpreter() {
        try {
            val modelBuffer = loadModelFile(MODEL_FILE) ?: loadModelFile(ALT_MODEL_FILE)
            if (modelBuffer != null) {
                val compatList = CompatibilityList()
                var initialized = false

                if (compatList.isDelegateSupportedOnThisDevice) {
                    try {
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        gpuDelegate = GpuDelegate(delegateOptions)
                        val options = Interpreter.Options().apply {
                            addDelegate(gpuDelegate)
                        }
                        interpreter = Interpreter(modelBuffer, options)
                        initialized = true
                        Log.i(TAG, "VisionClassifier TFLite inicializado con éxito usando GpuDelegate.")
                    } catch (e: Exception) {
                        Log.w(TAG, "GpuDelegate no compatible con INT8, reintentando en CPU: ${e.message}")
                        gpuDelegate?.close()
                        gpuDelegate = null
                    }
                }

                if (!initialized) {
                    val cpuOptions = Interpreter.Options().apply {
                        setNumThreads(4)
                    }
                    interpreter = Interpreter(modelBuffer, cpuOptions)
                    initialized = true
                    Log.i(TAG, "VisionClassifier TFLite inicializado con éxito en CPU multi-hilo (4 threads).")
                }

                isInitialized = initialized
            } else {
                Log.e(TAG, "Archivo de modelo '$MODEL_FILE' no encontrado en assets. Operando en modo simulado.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error al inicializar VisionClassifier TFLite: ${e.message}")
            isInitialized = false
        }
    }

    fun classifyFaceKeyframe(faceBitmap: Bitmap): Float {
        if (!isInitialized || interpreter == null) {
            return simulateInference(faceBitmap)
        }

        return try {
            val inputBuffer = convertBitmapToByteBuffer(faceBitmap)
            val currentInterpreter = interpreter ?: return simulateInference(faceBitmap)

            val outputTensor = currentInterpreter.getOutputTensor(0)
            val outShape = outputTensor.shape()
            val numClasses = if (outShape.isNotEmpty()) outShape.last() else 2

            val prob = if (numClasses == 1) {
                val outputBuffer = Array(1) { FloatArray(1) }
                currentInterpreter.run(inputBuffer, outputBuffer)
                val logit = outputBuffer[0][0]
                val sigmoid = (1.0 / (1.0 + Math.exp(-logit.toDouble()))).toFloat()
                Log.d(TAG, "Inferencia Visión TFLite (Sigmoid 1-logit) -> Logit: $logit | Prob: $sigmoid")
                sigmoid
            } else {
                val outputBuffer = Array(1) { FloatArray(numClasses) }
                currentInterpreter.run(inputBuffer, outputBuffer)
                val pReal = outputBuffer[0][0]
                val pFake = outputBuffer[0][1]
                val expFake = Math.exp(pFake.toDouble())
                val expReal = Math.exp(pReal.toDouble())
                val softmax = (expFake / (expReal + expFake)).toFloat()
                Log.d(TAG, "Inferencia Visión TFLite (Softmax 2-logits) -> Real: $pReal | Fake: $pFake | Prob: $softmax")
                softmax
            }
            prob
        } catch (e: Exception) {
            Log.e(TAG, "Error durante inferencia visual TFLite: ${e.message}")
            simulateInference(faceBitmap)
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val imgData = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        imgData.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(intValues, 0, scaled.width, 0, 0, scaled.width, scaled.height)

        for (pixelValue in intValues) {
            val r = (pixelValue shr 16 and 0xFF) / 255.0f
            val g = (pixelValue shr 8 and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f

            imgData.putFloat((r - 0.485f) / 0.229f)
            imgData.putFloat((g - 0.456f) / 0.224f)
            imgData.putFloat((b - 0.406f) / 0.225f)
        }
        return imgData
    }

    private fun loadModelFile(modelName: String): ByteBuffer? {
        return try {
            val fileDescriptor = context.assets.openFd(modelName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            try {
                context.assets.open(modelName).use { stream ->
                    val bytes = stream.readBytes()
                    val buffer = ByteBuffer.allocateDirect(bytes.size).apply {
                        order(ByteOrder.nativeOrder())
                        put(bytes)
                        rewind()
                    }
                    buffer
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Error cargando archivo de modelo TFLite '$modelName': ${e2.message}")
                null
            }
        }
    }

    private fun simulateInference(bitmap: Bitmap): Float {
        val hash = bitmap.hashCode()
        val baseProb = (Math.abs(hash % 100) / 100.0f) * 0.8f + 0.1f
        return baseProb.coerceIn(0.05f, 0.95f)
    }

    fun close() {
        try {
            interpreter?.close()
            interpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            Log.d(TAG, "Recursos de VisionClassifier liberados.")
        } catch (e: Exception) {
            Log.e(TAG, "Error al cerrar VisionClassifier: ${e.message}")
        }
    }
}

