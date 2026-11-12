package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class AudioClassifier(private val context: Context) {

    companion object {
        private const val TAG = "AudioClassifier"
        private const val MODEL_FILE = "experto_audio_int8.tflite"
        private const val ALT_MODEL_FILE = "modelo_audio_int8.tflite"
        private const val GRADCAM_WEIGHTS_FILE = "audio_gradcam_weights.bin"
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3
        private const val TARGET_SAMPLES = 48000
        internal const val AUDIO_TEMPERATURE = 0.60f
        internal const val AUDIO_FAKE_INDEX = 1

        internal fun normalizePcm48k(pcm: FloatArray): FloatArray {
            val out = FloatArray(TARGET_SAMPLES)
            val copyLen = minOf(pcm.size, TARGET_SAMPLES)
            if (copyLen > 0) {
                System.arraycopy(pcm, 0, out, 0, copyLen)
            }
            var maxAbs = 0.0f
            for (s in out) {
                val a = kotlin.math.abs(s)
                if (a > maxAbs) maxAbs = a
            }
            if (maxAbs > 1e-3f) {
                val inv = 1.0f / maxAbs
                for (i in out.indices) {
                    out[i] *= inv
                }
            }
            return out
        }

        internal fun computeTemporalFrameImportance(
            hiddenStates: FloatArray,
            attWeights: FloatArray?,
            numFrames: Int,
            hiddenDim: Int,
            contrastWeights: FloatArray?,
            isFakeDominant: Boolean = true
        ): FloatArray {
            if (numFrames <= 0 || hiddenDim <= 0) return FloatArray(0)
            val scores = FloatArray(numFrames)
            val totalHidden = hiddenStates.size
            val hasContrast = contrastWeights != null && contrastWeights.size >= hiddenDim
            var minVal = Float.MAX_VALUE
            var maxVal = -Float.MAX_VALUE

            for (t in 0 until numFrames) {
                val att = if (attWeights != null && t < attWeights.size) maxOf(attWeights[t], 0f) else 1f
                var raw = att
                if (hasContrast) {
                    var dot = 0f
                    val base = t * hiddenDim
                    val lim = minOf(hiddenDim, totalHidden - base)
                    val w = contrastWeights!!
                    for (c in 0 until lim) {
                        dot += hiddenStates[base + c] * w[c]
                    }
                    val signedDot = if (isFakeDominant) dot else -dot
                    raw = att * (if (signedDot > 0f) signedDot else 0f)
                }
                scores[t] = raw
                if (raw < minVal) minVal = raw
                if (raw > maxVal) maxVal = raw
            }

            if (maxVal <= 1e-7f && attWeights != null && attWeights.size >= numFrames) {
                minVal = Float.MAX_VALUE
                maxVal = -Float.MAX_VALUE
                for (t in 0 until numFrames) {
                    val att = maxOf(attWeights[t], 0f)
                    scores[t] = att
                    if (att < minVal) minVal = att
                    if (att > maxVal) maxVal = att
                }
            }

            val range = if (maxVal - minVal > 1e-7f) maxVal - minVal else 1f
            for (t in 0 until numFrames) {
                scores[t] = ((scores[t] - minVal) / range).coerceIn(0f, 1f)
            }
            return scores
        }

        internal fun computeCalibratedProbFake(
            logits: FloatArray,
            temperature: Float = AUDIO_TEMPERATURE,
            fakeIndex: Int = AUDIO_FAKE_INDEX
        ): Float {
            if (logits.isEmpty()) return 0.0f
            val t = maxOf(1e-4, temperature.toDouble())
            if (logits.size == 1) {
                val z = (logits[0].toDouble() / t).coerceIn(-60.0, 60.0)
                return (1.0 / (1.0 + Math.exp(-z))).toFloat().coerceIn(0.001f, 0.999f)
            }
            val safeFakeIdx = fakeIndex.coerceIn(0, logits.size - 1)
            var maxScaled = Double.NEGATIVE_INFINITY
            val scaled = DoubleArray(logits.size) { i ->
                val v = logits[i].toDouble() / t
                if (v > maxScaled) maxScaled = v
                v
            }
            var sumExp = 0.0
            val exps = DoubleArray(logits.size) { i ->
                val e = Math.exp((scaled[i] - maxScaled).coerceIn(-60.0, 60.0))
                sumExp += e
                e
            }
            if (sumExp <= 0.0) return 0.0f
            return (exps[safeFakeIdx] / sumExp).toFloat().coerceIn(0.001f, 0.999f)
        }

        internal fun computeNormalizedGradCamGrid(
            featureGrid: FloatArray,
            gridH: Int,
            gridW: Int,
            channels: Int,
            weights: FloatArray?
        ): Array<FloatArray> {
            val rawCam = Array(gridH) { FloatArray(gridW) }
            if (gridH <= 0 || gridW <= 0 || channels <= 0) return rawCam
            var minVal = Float.MAX_VALUE
            var maxVal = -Float.MAX_VALUE
            val totalElements = minOf(featureGrid.size, gridH * gridW * channels)

            for (y in 0 until gridH) {
                for (x in 0 until gridW) {
                    var sum = 0f
                    val baseIdx = (y * gridW + x) * channels
                    for (c in 0 until channels) {
                        val idx = baseIdx + c
                        if (idx >= totalElements) break
                        val wVal = if (weights != null && c < weights.size) weights[c] else 0f
                        sum += featureGrid[idx] * wVal
                    }
                    val relu = if (sum > 0f) sum else 0f
                    rawCam[y][x] = relu
                    if (relu < minVal) minVal = relu
                    if (relu > maxVal) maxVal = relu
                }
            }

            val range = if (maxVal - minVal > 1e-6f) maxVal - minVal else 1f
            val normCam = Array(gridH) { FloatArray(gridW) }
            for (y in 0 until gridH) {
                for (x in 0 until gridW) {
                    normCam[y][x] = (rawCam[y][x] - minVal) / range
                }
            }
            return normCam
        }
    }

    private var interpreter: Interpreter? = null
    private var modelByteBuffer: ByteBuffer? = null
    private var isInitialized = false

    @Volatile
    private var isRawWaveformModel = true

    @Volatile
    private var isClosed = false

    private var gradCamWeights: FloatArray? = null
    private val gradCamLock = Any()
    private var lastGradCamBitmap: Bitmap? = null

    init {
        initInterpreter()
        loadGradCamWeights()
    }

    private fun loadGradCamWeights() {
        try {
            context.assets.open(GRADCAM_WEIGHTS_FILE).use { stream ->
                val bytes = stream.readBytes()
                val floatBuf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val weights = FloatArray(floatBuf.remaining())
                floatBuf.get(weights)
                gradCamWeights = weights
                Log.i(TAG, "Pesos Grad-CAM de audio cargados exitosamente: ${weights.size} canales.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo cargar $GRADCAM_WEIGHTS_FILE: ${e.message}")
        }
    }

    private fun initInterpreter() {
        synchronized(TfliteExecutionLock.nativeLock) {
            if (isClosed) return
            try {
                val modelBuffer = loadModelFile(MODEL_FILE) ?: loadModelFile(ALT_MODEL_FILE)
                if (modelBuffer != null) {
                    val cpuOptions = Interpreter.Options().apply {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                    }
                    val created = Interpreter(modelBuffer, cpuOptions)
                    interpreter = created
                    modelByteBuffer = modelBuffer
                    val primaryInShape = if (created.inputTensorCount > 0) created.getInputTensor(0).shape() else IntArray(0)
                    isRawWaveformModel = primaryInShape.size == 2
                    isInitialized = true

                    val inCount = created.inputTensorCount
                    val outCount = created.outputTensorCount
                    Log.i(TAG, "AudioClassifier TFLite inicializado en CPU (4 hilos, XNNPACK=true, rawPcm=$isRawWaveformModel). Entradas=$inCount, Salidas=$outCount")
                    for (i in 0 until inCount) {
                        val t = created.getInputTensor(i)
                        Log.i(TAG, "  In[$i]: shape=${t.shape().contentToString()}, dtype=${t.dataType()}, bytes=${t.numBytes()}")
                    }
                    for (i in 0 until outCount) {
                        val t = created.getOutputTensor(i)
                        val qp = t.quantizationParams()
                        Log.i(TAG, "  Out[$i]: shape=${t.shape().contentToString()}, dtype=${t.dataType()}, scale=${qp.scale}, zp=${qp.zeroPoint}, bytes=${t.numBytes()}")
                    }
                } else {
                    Log.e(TAG, "Archivo de modelo '$MODEL_FILE' no encontrado en assets. AudioClassifier no disponible.")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error inicializando AudioClassifier: ${t.message}", t)
                interpreter = null
                modelByteBuffer = null
                isInitialized = false
            }
        }
    }

    fun getLastGradCamHeatmap(): Bitmap? = synchronized(gradCamLock) {
        lastGradCamBitmap?.takeIf { !it.isRecycled }
    }

    private fun setLastGradCamHeatmap(bitmap: Bitmap?) {
        synchronized(gradCamLock) {
            lastGradCamBitmap = bitmap
        }
    }

    fun classifyPcm(
        pcm: FloatArray,
        @Suppress("UNUSED_PARAMETER") contextHint: String? = null
    ): Float {
        if (isClosed || pcm.isEmpty()) {
            setLastGradCamHeatmap(null)
            return 0.0f
        }
        if (isInitialized && !isClosed) {
            try {
                if (isRawWaveformModel) {
                    val normPcm = normalizePcm48k(pcm)
                    val inputBuffer = ByteBuffer.allocateDirect(4 * TARGET_SAMPLES).order(ByteOrder.nativeOrder())
                    val fb = inputBuffer.asFloatBuffer()
                    fb.put(normPcm)
                    inputBuffer.rewind()
                    return executeAudioInference(inputBuffer)
                } else {
                    val windows = if (pcm.size > 72000) {
                        AudioSpectrogramGenerator.extractMultiWindowTiledPcm48k(pcm, maxWindows = 3)
                    } else {
                        listOf(pcm)
                    }
                    if (windows.size <= 1) {
                        val rgb = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(windows[0])
                        val bmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
                        bmp.setPixels(rgb.pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
                        return try {
                            classifySpectrogram(bmp, contextHint)
                        } finally {
                            if (!bmp.isRecycled) bmp.recycle()
                        }
                    }
                    var probSum = 0.0f
                    var maxProb = -1.0f
                    var bestHeatmap: Bitmap? = null
                    for (winPcm in windows) {
                        val rgb = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(winPcm, applyVadTile = false)
                        val bmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
                        bmp.setPixels(rgb.pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
                        val wp = try {
                            classifySpectrogram(bmp, contextHint)
                        } finally {
                            if (!bmp.isRecycled) bmp.recycle()
                        }
                        probSum += wp
                        if (wp > maxProb) {
                            maxProb = wp
                            bestHeatmap = getLastGradCamHeatmap()
                        }
                    }
                    if (bestHeatmap != null) {
                        setLastGradCamHeatmap(bestHeatmap)
                    }
                    return (probSum / windows.size.toFloat()).coerceIn(0.0f, 1.0f)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error durante inferencia PCM TFLite: ${t.message}", t)
            }
        }
        setLastGradCamHeatmap(null)
        return 0.0f
    }

    fun classifySpectrogram(
        spectrogramBitmap: Bitmap,
        @Suppress("UNUSED_PARAMETER") contextHint: String? = null
    ): Float {
        if (isClosed || isRawWaveformModel || spectrogramBitmap.isRecycled || spectrogramBitmap.width <= 0 || spectrogramBitmap.height <= 0) {
            setLastGradCamHeatmap(null)
            return 0.0f
        }

        if (isInitialized && !isClosed) {
            try {
                val inputBuffer = convertBitmapToByteBuffer(spectrogramBitmap)
                return executeAudioInference(inputBuffer)
            } catch (t: Throwable) {
                Log.e(TAG, "Error durante inferencia de audio TFLite: ${t.message}", t)
            }
        }

        setLastGradCamHeatmap(null)
        return 0.0f
    }

    private fun executeAudioInference(inputBuffer: ByteBuffer): Float {
        synchronized(TfliteExecutionLock.nativeLock) {
            val currentInterpreter = if (!isClosed) interpreter else null
            if (currentInterpreter != null) {
                val outTensorCount = currentInterpreter.outputTensorCount
                val outputsMap = HashMap<Int, Any>(outTensorCount)
                val outputBuffers = ArrayList<ByteBuffer>(outTensorCount)
                var logitTensorIdx = -1
                var featureTensorIdx = -1
                var hiddenStatesIdx = -1
                var attWeightsIdx = -1

                for (i in 0 until outTensorCount) {
                    val tensor = currentInterpreter.getOutputTensor(i)
                    val buf = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
                    outputBuffers.add(buf)
                    outputsMap[i] = buf
                    val shape = tensor.shape()
                    when {
                        shape.size == 4 && shape[1] > 1 && shape[2] > 1 -> featureTensorIdx = i
                        shape.size == 3 && shape[1] > 8 && shape[2] >= 128 -> hiddenStatesIdx = i
                        shape.size == 3 && shape[1] > 8 && shape[2] == 1 -> attWeightsIdx = i
                        shape.size == 2 && shape[1] in 2..8 -> logitTensorIdx = i
                        logitTensorIdx == -1 && shape.size <= 2 && tensor.numElements() in 1..8 -> logitTensorIdx = i
                    }
                }
                if (logitTensorIdx == -1 && outTensorCount > 0) {
                    logitTensorIdx = 0
                }

                currentInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputsMap)

                if (outTensorCount > 0) {
                    val logitTensor = currentInterpreter.getOutputTensor(logitTensorIdx)
                    val logitBuf = outputBuffers[logitTensorIdx].apply { rewind() }
                    val logits = readLogits(logitBuf, logitTensor)
                    val probFake = computeCalibratedProbFake(
                        logits = logits,
                        temperature = AUDIO_TEMPERATURE,
                        fakeIndex = AUDIO_FAKE_INDEX
                    )
                    Log.i(
                        TAG,
                        "Inferencia TFLite Audio ejecutada -> logits=${logits.contentToString()}, T=$AUDIO_TEMPERATURE, probFake=$probFake"
                    )

                    val gradCamBmp = when {
                        hiddenStatesIdx != -1 -> {
                            val hTensor = currentInterpreter.getOutputTensor(hiddenStatesIdx)
                            val hBuf = outputBuffers[hiddenStatesIdx].apply { rewind() }
                            val hShape = hTensor.shape()
                            val attData = if (attWeightsIdx != -1) {
                                val aTensor = currentInterpreter.getOutputTensor(attWeightsIdx)
                                val aBuf = outputBuffers[attWeightsIdx].apply { rewind() }
                                readFloatTensor(aBuf, aTensor)
                            } else {
                                null
                            }
                            computeTemporalHeatmapBitmap(
                                hiddenStates = readFloatTensor(hBuf, hTensor),
                                attWeights = attData,
                                numFrames = hShape[1],
                                hiddenDim = hShape[2],
                                riskProbability = probFake
                            )
                        }
                        featureTensorIdx != -1 -> {
                            val featTensor = currentInterpreter.getOutputTensor(featureTensorIdx)
                            val featBuf = outputBuffers[featureTensorIdx].apply { rewind() }
                            val featShape = featTensor.shape()
                            val qp = featTensor.quantizationParams()
                            computeGradCamBitmapFromBuffer(
                                buffer = featBuf,
                                gridH = featShape[1],
                                gridW = featShape[2],
                                channels = featShape[3],
                                dataType = featTensor.dataType(),
                                scale = if (qp.scale > 0f) qp.scale else 1f,
                                zeroPoint = qp.zeroPoint,
                                riskProbability = probFake
                            )
                        }
                        else -> null
                    }

                    setLastGradCamHeatmap(gradCamBmp)
                    return probFake
                }
            }
        }
        setLastGradCamHeatmap(null)
        return 0.0f
    }

    private fun readFloatTensor(buffer: ByteBuffer, tensor: Tensor): FloatArray {
        val count = tensor.numElements().coerceAtLeast(0)
        if (count == 0) return FloatArray(0)
        buffer.rewind()
        val qp = tensor.quantizationParams()
        val scale = if (qp.scale > 0f) qp.scale else 1.0f
        return when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                val fb = buffer.asFloatBuffer()
                val n = minOf(count, fb.remaining())
                FloatArray(n) { fb.get() }
            }
            DataType.INT8 -> {
                val n = minOf(count, buffer.remaining())
                FloatArray(n) { (buffer.get().toInt() - qp.zeroPoint) * scale }
            }
            DataType.UINT8 -> {
                val n = minOf(count, buffer.remaining())
                FloatArray(n) { ((buffer.get().toInt() and 0xFF) - qp.zeroPoint) * scale }
            }
            else -> FloatArray(0)
        }
    }

    private fun computeTemporalHeatmapBitmap(
        hiddenStates: FloatArray,
        attWeights: FloatArray?,
        numFrames: Int,
        hiddenDim: Int,
        riskProbability: Float
    ): Bitmap? {
        val frameScores = computeTemporalFrameImportance(
            hiddenStates = hiddenStates,
            attWeights = attWeights,
            numFrames = numFrames,
            hiddenDim = hiddenDim,
            contrastWeights = gradCamWeights,
            isFakeDominant = riskProbability >= 0.5f
        )
        if (frameScores.isEmpty()) return null

        val gridH = 16
        val normGrid = Array(gridH) { y ->
            val yNorm = y.toFloat() / (gridH - 1).toFloat()
            // Formant band emphasis over vocal frequency region [0.20 .. 0.90]
            val bandEnv = kotlin.math.exp(-((yNorm - 0.60f) * (yNorm - 0.60f)) / 0.14f).coerceIn(0.22f, 1.0f)
            FloatArray(frameScores.size) { x ->
                (frameScores[x] * bandEnv).coerceIn(0f, 1f)
            }
        }
        return renderHeatmapToBitmap(normGrid, gridH, frameScores.size, riskProbability)
    }

    private fun readLogits(buffer: ByteBuffer, tensor: Tensor): FloatArray {
        val count = tensor.numElements().coerceAtLeast(0)
        if (count == 0) return FloatArray(0)
        buffer.rewind()
        val qp = tensor.quantizationParams()
        val scale = if (qp.scale > 0f) qp.scale else 1.0f
        return when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                val fb = buffer.asFloatBuffer()
                val n = minOf(count, fb.remaining())
                FloatArray(n) { fb.get() }
            }
            DataType.INT8 -> {
                val n = minOf(count, buffer.remaining())
                FloatArray(n) { (buffer.get().toInt() - qp.zeroPoint) * scale }
            }
            DataType.UINT8 -> {
                val n = minOf(count, buffer.remaining())
                FloatArray(n) { ((buffer.get().toInt() and 0xFF) - qp.zeroPoint) * scale }
            }
            else -> FloatArray(0)
        }
    }

    private fun computeGradCamBitmapFromBuffer(
        buffer: ByteBuffer,
        gridH: Int,
        gridW: Int,
        channels: Int,
        dataType: DataType,
        scale: Float,
        zeroPoint: Int,
        riskProbability: Float
    ): Bitmap? {
        if (gridH <= 0 || gridW <= 0 || channels <= 0) {
            return null
        }
        val totalElements = gridH * gridW * channels
        buffer.rewind()
        val floatBuf = if (dataType == DataType.FLOAT32) buffer.asFloatBuffer() else null
        val featureGrid = FloatArray(totalElements) { idx ->
            when (dataType) {
                DataType.FLOAT32 -> if (floatBuf != null && idx < floatBuf.limit()) floatBuf.get(idx) else 0f
                DataType.INT8 -> if (idx < buffer.limit()) (buffer.get(idx).toInt() - zeroPoint) * scale else 0f
                DataType.UINT8 -> if (idx < buffer.limit()) ((buffer.get(idx).toInt() and 0xFF) - zeroPoint) * scale else 0f
                else -> 0f
            }
        }
        val normCam = computeNormalizedGradCamGrid(
            featureGrid = featureGrid,
            gridH = gridH,
            gridW = gridW,
            channels = channels,
            weights = gradCamWeights
        )
        return renderHeatmapToBitmap(normCam, gridH, gridW, riskProbability)
    }

    private fun renderHeatmapToBitmap(
        normCam: Array<FloatArray>,
        gridH: Int,
        gridW: Int,
        riskProbability: Float
    ): Bitmap {
        val outBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        val isHighRisk = riskProbability >= 0.5f
        val baseAlpha = if (isHighRisk) (riskProbability * 0.85f).coerceIn(0.40f, 0.85f) else 0.25f
        val maxY = maxOf(0, gridH - 1)
        val maxX = maxOf(0, gridW - 1)

        for (py in 0 until INPUT_SIZE) {
            val gy = if (maxY > 0) (py.toFloat() / (INPUT_SIZE - 1).toFloat()) * maxY else 0f
            val y0 = gy.toInt().coerceIn(0, maxY)
            val y1 = (y0 + 1).coerceIn(0, maxY)
            val dy = gy - y0

            for (px in 0 until INPUT_SIZE) {
                val gx = if (maxX > 0) (px.toFloat() / (INPUT_SIZE - 1).toFloat()) * maxX else 0f
                val x0 = gx.toInt().coerceIn(0, maxX)
                val x1 = (x0 + 1).coerceIn(0, maxX)
                val dx = gx - x0
                val v00 = normCam[y0][x0]
                val v10 = normCam[y0][x1]
                val v01 = normCam[y1][x0]
                val v11 = normCam[y1][x1]

                val v = (1f - dx) * (1f - dy) * v00 +
                        dx * (1f - dy) * v10 +
                        (1f - dx) * dy * v01 +
                        dx * dy * v11

                if (isHighRisk) {
                    val (r, g, b) = jetColorMap(v)
                    val alpha = ((v * baseAlpha).coerceIn(0.05f, 0.90f) * 255).toInt()
                    pixels[py * INPUT_SIZE + px] = (alpha shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    val r = (16 * (1f - v) + 56 * v).toInt().coerceIn(0, 255)
                    val g = (185 * (1f - v) + 200 * v).toInt().coerceIn(0, 255)
                    val b = (129 * (1f - v) + 150 * v).toInt().coerceIn(0, 255)
                    val alpha = ((v * baseAlpha).coerceIn(0.02f, 0.40f) * 255).toInt()
                    pixels[py * INPUT_SIZE + px] = (alpha shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        outBitmap.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        return outBitmap
    }

    private fun jetColorMap(v: Float): Triple<Int, Int, Int> {
        val clamped = v.coerceIn(0f, 1f)
        val r = (255 * (1.5f - Math.abs(clamped * 4f - 3f))).toInt().coerceIn(0, 255)
        val g = (255 * (1.5f - Math.abs(clamped * 4f - 2f))).toInt().coerceIn(0, 255)
        val b = (255 * (1.5f - Math.abs(clamped * 4f - 1f))).toInt().coerceIn(0, 255)
        return Triple(r, g, b)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val scaled = if (bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        }
        try {
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
            imgData.rewind()
            return imgData
        } finally {
            if (scaled !== bitmap && !scaled.isRecycled) {
                scaled.recycle()
            }
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer? {
        return try {
            context.assets.openFd(modelName).use { afd ->
                FileInputStream(afd.fileDescriptor).use { fis ->
                    val fileChannel = fis.channel
                    fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
                }
            }
        } catch (e: Exception) {
            try {
                context.assets.open(modelName).use { stream ->
                    val bytes = stream.readBytes()
                    ByteBuffer.allocateDirect(bytes.size).apply {
                        order(ByteOrder.nativeOrder())
                        put(bytes)
                        rewind()
                    }
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Error cargando modelo de audio '$modelName': ${e2.message}")
                null
            }
        }
    }

    fun close() {
        isClosed = true
        synchronized(TfliteExecutionLock.nativeLock) {
            try {
                interpreter?.close()
                interpreter = null
                modelByteBuffer = null
                isInitialized = false
                Log.d(TAG, "Recursos de AudioClassifier liberados.")
            } catch (e: Exception) {
                Log.e(TAG, "Error cerrando AudioClassifier: ${e.message}")
            }
        }
    }
}
