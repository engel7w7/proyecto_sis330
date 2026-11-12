package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs

class VisionClassifier(private val context: Context) {

    data class VideoExtractionResult(
        val crops: List<Bitmap>,
        val qualities: List<Float>,
        val hasFace: Boolean
    )

    companion object {
        private const val TAG = "VisionClassifier"
        private const val MODEL_FILE = "experto_vision_int8.tflite"
        private const val ALT_MODEL_FILE = "modelo_vision_int8.tflite"
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3
        private const val VISION_TEMPERATURE = 0.62f
        private const val VISION_FAKE_INDEX = 1
        private const val PORTRAIT_FACE_RATIO_THRESHOLD = 0.42f
        private const val PORTRAIT_SQUARE_ASPECT_THRESHOLD = 1.35f
        private const val PORTRAIT_EXPAND_MARGIN = 0.55f
        private const val SCENE_FACE_MARGIN = 0.15f

        data class CropRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
            val width: Int get() = right - left
            val height: Int get() = bottom - top
        }

        /**
         * Encuadre adaptativo alineado con el entrenamiento multi-dominio:
         * - Retratos cercanos (frac >= 0.42): conserva cabeza/cabello/contorno igual que SDXL/StyleGAN/JamieWithofs.
         * - Escenas amplias o fotogramas de video: aplica recorte BlazeFace + 15% igual que FF++/Thien0103.
         */
        fun computeAdaptiveFaceCropRect(
            imgWidth: Int,
            imgHeight: Int,
            faceLeft: Int,
            faceTop: Int,
            faceRight: Int,
            faceBottom: Int,
            forceTightCrop: Boolean = false
        ): CropRect {
            val safeW = maxOf(1, imgWidth)
            val safeH = maxOf(1, imgHeight)
            val bw = maxOf(1, faceRight - faceLeft)
            val bh = maxOf(1, faceBottom - faceTop)
            val faceFrac = maxOf(bw.toFloat() / safeW, bh.toFloat() / safeH)
            val aspect = maxOf(safeW, safeH).toFloat() / minOf(safeW, safeH).toFloat()

            if (!forceTightCrop && faceFrac >= PORTRAIT_FACE_RATIO_THRESHOLD) {
                if (aspect <= PORTRAIT_SQUARE_ASPECT_THRESHOLD) {
                    return CropRect(0, 0, safeW, safeH)
                }
                val mx = (bw * PORTRAIT_EXPAND_MARGIN).toInt()
                val my = (bh * PORTRAIT_EXPAND_MARGIN).toInt()
                val left = (faceLeft - mx).coerceIn(0, safeW - 1)
                val top = (faceTop - my).coerceIn(0, safeH - 1)
                val right = (faceRight + mx).coerceIn(left + 1, safeW)
                val bottom = (faceBottom + my).coerceIn(top + 1, safeH)
                return CropRect(left, top, right, bottom)
            }

            val mx = (bw * SCENE_FACE_MARGIN).toInt()
            val my = (bh * SCENE_FACE_MARGIN).toInt()
            val left = (faceLeft - mx).coerceIn(0, safeW - 1)
            val top = (faceTop - my).coerceIn(0, safeH - 1)
            val right = (faceRight + mx).coerceIn(left + 1, safeW)
            val bottom = (faceBottom + my).coerceIn(top + 1, safeH)
            return CropRect(left, top, right, bottom)
        }

        fun computeLaplacianSharpness(bitmap: Bitmap): Float {
            if (bitmap.isRecycled || bitmap.width < 16 || bitmap.height < 16) return 0.50f
            val sz = 64
            val scaled = if (bitmap.width == sz && bitmap.height == sz) {
                bitmap
            } else {
                Bitmap.createScaledBitmap(bitmap, sz, sz, true)
            }
            return try {
                val px = IntArray(sz * sz)
                scaled.getPixels(px, 0, sz, 0, 0, sz, sz)
                val lum = FloatArray(sz * sz)
                for (i in px.indices) {
                    val c = px[i]
                    val r = (c shr 16) and 0xFF
                    val g = (c shr 8) and 0xFF
                    val b = c and 0xFF
                    lum[i] = 0.299f * r + 0.587f * g + 0.114f * b
                }
                var sumSq = 0.0
                var count = 0
                for (y in 1 until (sz - 1)) {
                    val row = y * sz
                    for (x in 1 until (sz - 1)) {
                        val idx = row + x
                        val lap = (4.0f * lum[idx] - lum[idx - 1] - lum[idx + 1] - lum[idx - sz] - lum[idx + sz]).toDouble()
                        sumSq += lap * lap
                        count++
                    }
                }
                val varLap = if (count > 0) sumSq / count else 0.0
                (0.35f + (varLap / 360.0).toFloat() * 0.65f).coerceIn(0.35f, 1.0f)
            } catch (_: Throwable) {
                0.75f
            } finally {
                if (scaled !== bitmap && !scaled.isRecycled) {
                    scaled.recycle()
                }
            }
        }

        private fun probeFacesSync(
            detector: FaceDetector,
            bmp: Bitmap,
            isClosedCheck: () -> Boolean
        ): List<Face> {
            return synchronized(TfliteExecutionLock.faceLock) {
                if (isClosedCheck()) emptyList() else Tasks.await(detector.process(InputImage.fromBitmap(bmp, 0)))
            }
        }

        fun detectAndCropFaceWithDetector(
            detector: FaceDetector,
            source: Bitmap,
            targetSize: Int = INPUT_SIZE,
            forceTightCrop: Boolean = false,
            isClosedCheck: () -> Boolean = { false },
            onNoFace: (Bitmap) -> Bitmap
        ): Pair<Bitmap, Boolean> {
            if (isClosedCheck() || source.isRecycled || source.width <= 0 || source.height <= 0) {
                return Pair(Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888), false)
            }
            var normalizedSource: Bitmap? = null
            var upscaledProbe: Bitmap? = null
            var hiResProbe: Bitmap? = null
            var centerProbe: Bitmap? = null
            var paddedProbe: Bitmap? = null
            var croppedTemp: Bitmap? = null
            return try {
                val maxDim = maxOf(source.width, source.height)
                val target = if (maxDim > 800) {
                    val scale = 800f / maxDim
                    val scaled = Bitmap.createScaledBitmap(
                        source,
                        maxOf(1, (source.width * scale).toInt()),
                        maxOf(1, (source.height * scale).toInt()),
                        true
                    )
                    normalizedSource = scaled
                    scaled
                } else {
                    source
                }

                var cropRef: Bitmap = target
                var coordScaleBack = 1f
                var offsetX = 0f
                var offsetY = 0f
                var faces = probeFacesSync(detector, target, isClosedCheck)

                val minTargetDim = minOf(target.width, target.height)
                if (faces.isEmpty() && minTargetDim in 48 until 320) {
                    val mul = maxOf(2, (360 + minTargetDim - 1) / minTargetDim)
                    val probe = Bitmap.createScaledBitmap(target, target.width * mul, target.height * mul, true)
                    upscaledProbe = probe
                    faces = probeFacesSync(detector, probe, isClosedCheck)
                    if (faces.isNotEmpty()) {
                        coordScaleBack = 1.0f / mul.toFloat()
                    }
                }

                if (faces.isEmpty() && maxDim > 800) {
                    val hiScale = minOf(1.0f, 1280f / maxDim)
                    val hiW = maxOf(1, (source.width * hiScale).toInt())
                    val hiH = maxOf(1, (source.height * hiScale).toInt())
                    val hi = if (hiScale >= 0.99f) source else Bitmap.createScaledBitmap(source, hiW, hiH, true)
                    if (hi !== source) hiResProbe = hi
                    faces = probeFacesSync(detector, hi, isClosedCheck)
                    if (faces.isNotEmpty()) {
                        cropRef = hi
                        coordScaleBack = 1.0f
                    }
                }

                if (faces.isEmpty() && minOf(source.width, source.height) >= 200) {
                    val cw = maxOf(64, (source.width * 0.62f).toInt())
                    val ch = maxOf(64, (source.height * 0.62f).toInt())
                    val cx0 = ((source.width - cw) / 2).coerceIn(0, source.width - cw)
                    val cy0 = ((source.height - ch) / 3).coerceIn(0, source.height - ch)
                    val cropZoom = Bitmap.createBitmap(source, cx0, cy0, cw, ch)
                    val zoomScaled = if (maxOf(cw, ch) < 480) {
                        val s = 480f / maxOf(cw, ch)
                        val zs = Bitmap.createScaledBitmap(cropZoom, maxOf(1, (cw * s).toInt()), maxOf(1, (ch * s).toInt()), true)
                        if (zs !== cropZoom && !cropZoom.isRecycled) cropZoom.recycle()
                        zs
                    } else {
                        cropZoom
                    }
                    centerProbe = zoomScaled
                    faces = probeFacesSync(detector, zoomScaled, isClosedCheck)
                    if (faces.isNotEmpty()) {
                        cropRef = zoomScaled
                        coordScaleBack = 1.0f
                    }
                }

                if (faces.isEmpty() && minOf(target.width, target.height) >= 64) {
                    val padX = maxOf(16, (target.width * 0.18f).toInt())
                    val padY = maxOf(16, (target.height * 0.18f).toInt())
                    val padW = target.width + padX * 2
                    val padH = target.height + padY * 2
                    val canvasBmp = Bitmap.createBitmap(padW, padH, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(canvasBmp)
                    canvas.drawColor(Color.rgb(24, 24, 28))
                    canvas.drawBitmap(target, padX.toFloat(), padY.toFloat(), null)
                    paddedProbe = canvasBmp
                    faces = probeFacesSync(detector, canvasBmp, isClosedCheck)
                    if (faces.isNotEmpty()) {
                        cropRef = target
                        coordScaleBack = 1.0f
                        offsetX = -padX.toFloat()
                        offsetY = -padY.toFloat()
                    }
                }

                if (faces.isNotEmpty()) {
                    val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: faces[0]
                    val box = face.boundingBox
                    val fLeft = (box.left * coordScaleBack + offsetX).toInt().coerceIn(0, maxOf(0, cropRef.width - 2))
                    val fTop = (box.top * coordScaleBack + offsetY).toInt().coerceIn(0, maxOf(0, cropRef.height - 2))
                    val fRight = (box.right * coordScaleBack + offsetX).toInt().coerceIn(fLeft + 1, cropRef.width)
                    val fBottom = (box.bottom * coordScaleBack + offsetY).toInt().coerceIn(fTop + 1, cropRef.height)
                    val rect = computeAdaptiveFaceCropRect(
                        imgWidth = cropRef.width,
                        imgHeight = cropRef.height,
                        faceLeft = fLeft,
                        faceTop = fTop,
                        faceRight = fRight,
                        faceBottom = fBottom,
                        forceTightCrop = forceTightCrop
                    )

                    if (rect.width > 24 && rect.height > 24) {
                        val isFullFrame = rect.left == 0 && rect.top == 0 &&
                            rect.width == cropRef.width && rect.height == cropRef.height
                        val result = if (isFullFrame) {
                            Bitmap.createScaledBitmap(cropRef, targetSize, targetSize, true)
                        } else {
                            val cropped = Bitmap.createBitmap(cropRef, rect.left, rect.top, rect.width, rect.height)
                            croppedTemp = cropped
                            if (rect.width == targetSize && rect.height == targetSize) {
                                croppedTemp = null
                                cropped
                            } else {
                                Bitmap.createScaledBitmap(cropped, targetSize, targetSize, true)
                            }
                        }
                        Log.d(TAG, "BlazeFace ROI [x=${rect.left}, y=${rect.top}, dim=${rect.width}x${rect.height}, full=$isFullFrame, tight=$forceTightCrop].")
                        return Pair(result, true)
                    }
                }

                Log.i(TAG, "BlazeFace: No se detecto rostro humano en la imagen. Descarte automatico activo.")
                Pair(onNoFace(target), false)
            } catch (t: Throwable) {
                Log.w(TAG, "Deteccion facial BlazeFace con fallback seguro: ${t.message}")
                Pair(onNoFace(source), false)
            } finally {
                val up = upscaledProbe
                if (up != null && up !== source && up !== normalizedSource && !up.isRecycled) up.recycle()
                val hr = hiResProbe
                if (hr != null && hr !== source && hr !== normalizedSource && !hr.isRecycled) hr.recycle()
                val cp = centerProbe
                if (cp != null && cp !== source && cp !== normalizedSource && !cp.isRecycled) cp.recycle()
                val pp = paddedProbe
                if (pp != null && pp !== source && pp !== normalizedSource && !pp.isRecycled) pp.recycle()
                val ct = croppedTemp
                if (ct != null && ct !== source && ct !== normalizedSource && !ct.isRecycled) ct.recycle()
                val ns = normalizedSource
                if (ns != null && ns !== source && !ns.isRecycled) ns.recycle()
            }
        }

        fun extractBestFiveVideoKeyframes(
            context: Context,
            uri: Uri,
            detector: FaceDetector,
            forceTightCrop: Boolean = true,
            maxFrames: Int = 5,
            isClosedCheck: () -> Boolean = { false },
            onNoFace: (Bitmap) -> Bitmap
        ): VideoExtractionResult {
            val retriever = MediaMetadataRetriever()
            return try {
                if (uri.scheme == "file" && !uri.path.isNullOrBlank()) {
                    retriever.setDataSource(uri.path)
                } else {
                    retriever.setDataSource(context, uri)
                }
                val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                val durationUs = maxOf(0L, durationMs * 1000L)
                val fractions = listOf(0.10, 0.30, 0.50, 0.70, 0.90).take(maxFrames.coerceIn(1, 5))
                val baseTimesUs = if (durationMs > 500L) {
                    fractions.map { (durationUs.toDouble() * it).toLong() }
                } else {
                    listOf(0L)
                }

                val selectedCrops = mutableListOf<Bitmap>()
                val selectedQualities = mutableListOf<Float>()
                var fallbackThumb: Bitmap? = null
                val offsets = if (durationMs > 1800L) longArrayOf(0L, -400_000L, 400_000L) else longArrayOf(0L)

                for (baseTs in baseTimesUs) {
                    var bestCrop: Bitmap? = null
                    var bestSharpness = -1.0f

                    for (oi in offsets.indices) {
                        val tsUs = if (durationUs > 0L) (baseTs + offsets[oi]).coerceIn(0L, maxOf(0L, durationUs - 50_000L)) else 0L
                        var rawFrame: Bitmap? = null
                        try {
                            val opt = if (oi == 0) MediaMetadataRetriever.OPTION_CLOSEST_SYNC else MediaMetadataRetriever.OPTION_CLOSEST
                            rawFrame = retriever.getFrameAtTime(tsUs, opt) ?: retriever.getFrameAtTime(tsUs)
                            if (rawFrame != null && rawFrame.width > 20 && rawFrame.height > 20) {
                                val (crop, hasFace) = detectAndCropFaceWithDetector(
                                    detector = detector,
                                    source = rawFrame,
                                    targetSize = INPUT_SIZE,
                                    forceTightCrop = forceTightCrop,
                                    isClosedCheck = isClosedCheck,
                                    onNoFace = onNoFace
                                )
                                if (hasFace) {
                                    val sharp = computeLaplacianSharpness(crop)
                                    if (sharp > bestSharpness) {
                                        if (bestCrop != null && !bestCrop.isRecycled) bestCrop.recycle()
                                        bestCrop = crop
                                        bestSharpness = sharp
                                    } else if (!crop.isRecycled) {
                                        crop.recycle()
                                    }
                                    if (bestSharpness >= 0.72f) break
                                } else {
                                    if (fallbackThumb == null) {
                                        fallbackThumb = crop
                                    } else if (!crop.isRecycled) {
                                        crop.recycle()
                                    }
                                }
                            }
                        } catch (_: Throwable) {
                        } finally {
                            if (rawFrame != null && !rawFrame.isRecycled) rawFrame.recycle()
                        }
                    }

                    if (bestCrop != null) {
                        selectedCrops.add(bestCrop)
                        selectedQualities.add(bestSharpness.coerceIn(0.35f, 1.0f))
                    }
                }

                if (selectedCrops.isNotEmpty()) {
                    if (fallbackThumb != null && !fallbackThumb.isRecycled) fallbackThumb.recycle()
                    VideoExtractionResult(selectedCrops, selectedQualities, true)
                } else {
                    VideoExtractionResult(listOfNotNull(fallbackThumb), emptyList(), false)
                }
            } catch (_: Throwable) {
                VideoExtractionResult(emptyList(), emptyList(), false)
            } finally {
                try { retriever.release() } catch (_: Throwable) {}
            }
        }
    }

    fun extractBestFiveVideoKeyframes(
        uri: Uri,
        forceTightCrop: Boolean = true,
        maxFrames: Int = 5
    ): VideoExtractionResult {
        return extractBestFiveVideoKeyframes(
            context = context,
            uri = uri,
            detector = blazeFaceDetector,
            forceTightCrop = forceTightCrop,
            maxFrames = maxFrames,
            isClosedCheck = { isClosed },
            onNoFace = { Bitmap.createScaledBitmap(it, INPUT_SIZE, INPUT_SIZE, true) }
        )
    }

    private var interpreter: Interpreter? = null
    private var modelByteBuffer: ByteBuffer? = null
    private var isInitialized = false

    @Volatile
    private var isClosed = false

    private var gradCamWeights: FloatArray? = null
    private val gradCamLock = Any()
    private var lastGradCamBitmap: Bitmap? = null

    private val blazeFaceDetector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.05f)
            .build()
        FaceDetection.getClient(options)
    }

    init {
        initInterpreter()
        loadGradCamWeights()
    }

    fun getLastGradCamHeatmap(): Bitmap? = synchronized(gradCamLock) {
        lastGradCamBitmap?.takeIf { !it.isRecycled }
    }

    private fun setLastGradCamHeatmap(bitmap: Bitmap?) {
        synchronized(gradCamLock) {
            lastGradCamBitmap = bitmap
        }
    }

    private fun loadGradCamWeights() {
        try {
            context.assets.open("gradcam_weights.bin").use { stream ->
                val bytes = stream.readBytes()
                val floatBuf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val weights = FloatArray(floatBuf.remaining())
                floatBuf.get(weights)
                gradCamWeights = weights
                Log.i(TAG, "Pesos Grad-CAM cargados exitosamente: ${weights.size} canales.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo cargar gradcam_weights.bin: ${e.message}")
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
                    isInitialized = true

                    val inCount = created.inputTensorCount
                    val outCount = created.outputTensorCount
                    Log.i(TAG, "VisionClassifier TFLite inicializado en CPU (4 hilos, XNNPACK=true). Entradas=$inCount, Salidas=$outCount")
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
                    Log.e(TAG, "Archivo de modelo '$MODEL_FILE' no encontrado en assets. VisionClassifier no disponible.")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error al inicializar VisionClassifier TFLite: ${t.message}", t)
                interpreter = null
                modelByteBuffer = null
                isInitialized = false
            }
        }
    }

    data class FaceDetectionResult(
        val hasFace: Boolean,
        val processedBitmap: Bitmap,
        val confidenceScore: Float
    )

    fun checkFacePresence(bitmap: Bitmap): Boolean {
        if (isClosed || bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            return false
        }
        var scaled: Bitmap? = null
        return try {
            val maxDim = maxOf(bitmap.width, bitmap.height)
            val target = if (maxDim > 640) {
                val factor = 640f / maxDim
                val s = Bitmap.createScaledBitmap(
                    bitmap,
                    maxOf(1, (bitmap.width * factor).toInt()),
                    maxOf(1, (bitmap.height * factor).toInt()),
                    true
                )
                scaled = s
                s
            } else {
                bitmap
            }
            val inputImage = InputImage.fromBitmap(target, 0)
            val faces = synchronized(TfliteExecutionLock.faceLock) {
                if (isClosed) emptyList() else Tasks.await(blazeFaceDetector.process(inputImage))
            }
            faces.isNotEmpty()
        } catch (t: Throwable) {
            Log.w(TAG, "Error o excepcion durante verificacion facial BlazeFace: ${t.message}")
            false
        } finally {
            val s = scaled
            if (s != null && s !== bitmap && !s.isRecycled) {
                s.recycle()
            }
        }
    }

    fun detectAndCropFace(source: Bitmap, forceTightCrop: Boolean = false): Pair<Bitmap, Boolean> {
        return detectAndCropFaceWithDetector(
            detector = blazeFaceDetector,
            source = source,
            targetSize = INPUT_SIZE,
            forceTightCrop = forceTightCrop,
            isClosedCheck = { isClosed },
            onNoFace = { createCenterSquareThumbnail(it) }
        )
    }

    private fun createCenterSquareThumbnail(source: Bitmap): Bitmap {
        val minDim = maxOf(1, minOf(source.width, source.height))
        val cropX = ((source.width - minDim) / 2).coerceIn(0, maxOf(0, source.width - 1))
        val cropY = ((source.height - minDim) / 2).coerceIn(0, maxOf(0, source.height - 1))
        val centerCropped = Bitmap.createBitmap(source, cropX, cropY, minDim, minDim)
        if (minDim == INPUT_SIZE) {
            return centerCropped
        }
        val scaled = Bitmap.createScaledBitmap(centerCropped, INPUT_SIZE, INPUT_SIZE, true)
        if (scaled !== centerCropped && !centerCropped.isRecycled) {
            centerCropped.recycle()
        }
        return scaled
    }

    fun classifyFaceKeyframe(
        faceBitmap: Bitmap,
        @Suppress("UNUSED_PARAMETER") contextHint: String? = null,
        requireFaceDetection: Boolean = true
    ): Float {
        if (isClosed || faceBitmap.isRecycled || faceBitmap.width <= 0 || faceBitmap.height <= 0) {
            setLastGradCamHeatmap(null)
            return 0.0f
        }

        if (requireFaceDetection) {
            val hasFace = checkFacePresence(faceBitmap)
            if (!hasFace) {
                Log.i(TAG, "Descarte automatico activado: No se detecto rostro humano en la imagen con BlazeFace.")
                setLastGradCamHeatmap(null)
                return 0.0f
            }
        }

        if (isInitialized && !isClosed) {
            try {
                val inputBuffer = convertBitmapToByteBuffer(faceBitmap)
                synchronized(TfliteExecutionLock.nativeLock) {
                    val currentInterpreter = if (!isClosed) interpreter else null
                    if (currentInterpreter != null) {
                        val outTensorCount = currentInterpreter.outputTensorCount
                        val outputsMap = HashMap<Int, Any>(outTensorCount)
                        val outputBuffers = ArrayList<ByteBuffer>(outTensorCount)
                        var logitTensorIdx = -1
                        var featureTensorIdx = -1

                        for (i in 0 until outTensorCount) {
                            val tensor = currentInterpreter.getOutputTensor(i)
                            val buf = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
                            outputBuffers.add(buf)
                            outputsMap[i] = buf
                            val shape = tensor.shape()
                            if (shape.size == 4 && shape[1] > 1 && shape[2] > 1) {
                                featureTensorIdx = i
                            } else if (logitTensorIdx == -1 && tensor.numElements() in 1..8) {
                                logitTensorIdx = i
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
                                temperature = VISION_TEMPERATURE,
                                fakeIndex = VISION_FAKE_INDEX
                            )
                            Log.i(
                                TAG,
                                "Inferencia TFLite Vision ejecutada -> logits=${logits.contentToString()}, T=$VISION_TEMPERATURE, probFake=$probFake"
                            )

                            val gradCamBmp = if (featureTensorIdx != -1) {
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
                            } else {
                                generateAdaptiveGradCamBitmap(probFake)
                            }

                            setLastGradCamHeatmap(gradCamBmp)
                            return probFake
                        }
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error durante inferencia visual TFLite: ${t.message}", t)
            }
        }

        setLastGradCamHeatmap(null)
        return 0.0f
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

    private fun computeCalibratedProbFake(logits: FloatArray, temperature: Float, fakeIndex: Int): Float {
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
                Log.e(TAG, "Error cargando archivo de modelo TFLite '$modelName': ${e2.message}")
                null
            }
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
    ): Bitmap {
        if (gridH <= 0 || gridW <= 0 || channels <= 0) {
            return generateAdaptiveGradCamBitmap(riskProbability)
        }
        val weights = gradCamWeights
        val rawCam = Array(gridH) { FloatArray(gridW) }
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE

        buffer.rewind()
        val floatBuf = if (dataType == DataType.FLOAT32) buffer.asFloatBuffer() else null
        val totalElements = gridH * gridW * channels

        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                var sum = 0f
                val baseIdx = (y * gridW + x) * channels
                for (c in 0 until channels) {
                    val idx = baseIdx + c
                    if (idx >= totalElements) break
                    val fVal = when (dataType) {
                        DataType.FLOAT32 -> floatBuf?.get(idx) ?: 0f
                        DataType.INT8 -> (buffer.get(idx).toInt() - zeroPoint) * scale
                        DataType.UINT8 -> ((buffer.get(idx).toInt() and 0xFF) - zeroPoint) * scale
                        else -> 0f
                    }
                    val wVal = if (weights != null && c < weights.size) weights[c] else 0f
                    sum += fVal * wVal
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

        return renderHeatmapToBitmap(normCam, gridH, gridW, riskProbability)
    }

    private fun generateAdaptiveGradCamBitmap(
        riskProbability: Float
    ): Bitmap {
        val gridH = 7
        val gridW = 7
        val normCam = Array(gridH) { FloatArray(gridW) }
        val isHighRisk = riskProbability >= 0.5f

        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                val dx = (x - 3.0f) / 3.0f
                val dy = (y - 3.0f) / 3.0f
                val dist = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()

                val facialFeature = if (isHighRisk) {
                    val eyeZone = Math.exp(-((x - 3.0) * (x - 3.0) + (y - 2.5) * (y - 2.5)) / 1.5).toFloat()
                    val mouthZone = Math.exp(-((x - 3.0) * (x - 3.0) + (y - 4.8) * (y - 4.8)) / 1.2).toFloat()
                    (eyeZone * 0.9f + mouthZone * 0.75f).coerceIn(0f, 1f)
                } else {
                    (1.0f - dist.coerceIn(0f, 1f)) * 0.25f
                }

                normCam[y][x] = facialFeature
            }
        }

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

    fun close() {
        isClosed = true
        synchronized(TfliteExecutionLock.nativeLock) {
            try {
                interpreter?.close()
                interpreter = null
                modelByteBuffer = null
                isInitialized = false
                Log.d(TAG, "Recursos de VisionClassifier liberados.")
            } catch (e: Exception) {
                Log.e(TAG, "Error al cerrar VisionClassifier: ${e.message}")
            }
        }
        try {
            blazeFaceDetector.close()
        } catch (_: Exception) {
        }
    }
}
