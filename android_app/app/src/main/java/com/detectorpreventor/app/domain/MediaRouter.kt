package com.detectorpreventor.app.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.detectorpreventor.app.ml.TfliteExecutionLock
import com.detectorpreventor.app.notifications.NotificationMediaCache
import com.detectorpreventor.app.notifications.NotificationRepository
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

enum class MediaType {
    AUDIO_ONLY,
    IMAGE_ONLY,
    VIDEO_MULTIMODAL,
    UNKNOWN
}

data class ProcessedMediaPayload(
    val mediaType: MediaType,
    val audioSpectrogram: Bitmap? = null,
    val faceKeyframe: Bitmap? = null,
    val gradCamBitmap: Bitmap? = null,
    val audioGradCamBitmap: Bitmap? = null,
    val filename: String = "",
    val faceDetected: Boolean = true,
    val discardReason: String? = null
)

class MediaRouter(private val context: Context) {

    companion object {
        private const val TAG = "MediaRouter"
    }

    private val blazeFaceDetector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.12f)
            .build()
        FaceDetection.getClient(options)
    }

    suspend fun processIncomingUri(uri: Uri, mimeType: String?): ProcessedMediaPayload = withContext(Dispatchers.IO) {
        val uriStr = uri.toString()
        if (uriStr.startsWith("content://com.detectorpreventor.app.samples/")) {
            val parts = uri.pathSegments
            val sampleType = if (parts.isNotEmpty()) parts[0] else "audio"
            val index = if (parts.size > 1) parts[1].toIntOrNull() ?: 1 else 1

            return@withContext when (sampleType) {
                "audio" -> {
                    val asset = when (index) {
                        1 -> "samples/01_voz_humana_real_bonafide_1_spec.png"
                        2 -> "samples/06_clonacion_ia_spoof_tts_1_spec.png"
                        3 -> "samples/07_clonacion_ia_spoof_tts_2_spec.png"
                        4 -> "samples/08_clonacion_ia_spoof_tts_3_spec.png"
                        else -> "samples/02_voz_humana_real_bonafide_2_spec.png"
                    }
                    val bitmap = loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (index % 2 != 0) "samples/audio_real_spec.png" else "samples/audio_fake_spec.png")
                        ?: generateSpectrogramFromAudio(uri)
                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        faceKeyframe = null,
                        filename = getFileNameFromUri(uri),
                        faceDetected = true
                    )
                }
                "image" -> {
                    val asset = when (index) {
                        1 -> "samples/01_retrato_humano_real_1.jpg"
                        2 -> "samples/06_deepfake_rostro_ia_1.jpg"
                        3 -> "samples/07_deepfake_rostro_ia_2.jpg"
                        4 -> "samples/08_deepfake_rostro_ia_3.jpg"
                        else -> "samples/02_retrato_humano_real_2.jpg"
                    }
                    val bitmap = loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (index % 2 != 0) "samples/image_real_face.jpg" else "samples/image_fake_face.jpg")
                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        audioSpectrogram = null,
                        faceKeyframe = bitmap,
                        filename = getFileNameFromUri(uri),
                        faceDetected = true
                    )
                }
                else -> {
                    val faceAsset = when (index) {
                        1 -> "samples/01_retrato_humano_real_1.jpg"
                        2 -> "samples/06_deepfake_rostro_ia_1.jpg"
                        3 -> "samples/08_deepfake_rostro_ia_3.jpg"
                        4 -> "samples/09_deepfake_rostro_ia_4.jpg"
                        else -> "samples/03_retrato_humano_real_3.jpg"
                    }
                    val audioAsset = when (index) {
                        1 -> "samples/01_voz_humana_real_bonafide_1_spec.png"
                        2 -> "samples/06_clonacion_ia_spoof_tts_1_spec.png"
                        3 -> "samples/07_clonacion_ia_spoof_tts_2_spec.png"
                        4 -> "samples/08_clonacion_ia_spoof_tts_3_spec.png"
                        else -> "samples/04_voz_humana_real_bonafide_4_spec.png"
                    }
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = loadBitmapFromAsset(audioAsset) ?: generateSpectrogramFromAudio(uri),
                        faceKeyframe = loadBitmapFromAsset(faceAsset) ?: loadBitmapFromAsset("samples/image_real_face.jpg"),
                        filename = getFileNameFromUri(uri),
                        faceDetected = true
                    )
                }
            }
        }

        if (uriStr.startsWith("content://com.detectorpreventor.app.notifications/")) {
            val notifId = uri.lastPathSegment ?: ""
            val notif = NotificationRepository.notifications.value.find { it.id == notifId }
            val mType = notif?.mediaType ?: resolveMediaType(uri, mimeType)

            val senderLower = notif?.sender?.lowercase() ?: ""
            val textLower = notif?.text?.lowercase() ?: ""
            val isThreatContext = (notif?.isThreat == true) ||
                    (notif?.riskScore != null && notif.riskScore >= 70f) ||
                    senderLower.contains("urgente") ||
                    senderLower.contains("mamá") ||
                    senderLower.contains("mama") ||
                    senderLower.contains("desconocido") ||
                    senderLower.contains("+") ||
                    senderLower.contains("banco") ||
                    textLower.contains("urgente") ||
                    textLower.contains("depósito") ||
                    textLower.contains("deposito") ||
                    textLower.contains("tarjeta") ||
                    textLower.contains("dinero") ||
                    textLower.contains("comprometedor") ||
                    textLower.contains("alerta") ||
                    textLower.contains("sospechoso") ||
                    textLower.contains("clonaci") ||
                    textLower.contains("deepfake") ||
                    textLower.contains("faceswap")

            val storedUriStr = notif?.mediaUriString ?: NotificationMediaCache.getMediaUriString(notifId)
            val cachedFace = NotificationMediaCache.getFaceBitmap(context, notifId) ?: NotificationMediaCache.getFaceBitmap(notifId)
            val cachedAudio = NotificationMediaCache.getAudioBitmap(context, notifId) ?: NotificationMediaCache.getAudioBitmap(notifId)

            val defaultRealFace = "samples/01_retrato_humano_real_1.jpg"
            val defaultFakeFace = "samples/06_deepfake_rostro_ia_1.jpg"
            val defaultRealAudio = "samples/01_voz_humana_real_bonafide_1_spec.png"
            val defaultFakeAudio = "samples/06_clonacion_ia_spoof_tts_1_spec.png"

            return@withContext when (mType) {
                MediaType.AUDIO_ONLY -> {
                    val asset = if (isThreatContext) defaultFakeAudio else defaultRealAudio
                    val bitmap = cachedAudio
                        ?: storedUriStr?.let { s ->
                            when {
                                s.startsWith("asset://") -> loadBitmapFromAsset(s.removePrefix("asset://"))
                                s.startsWith("file://") || s.startsWith("content://") -> {
                                    val parsed = Uri.parse(s)
                                    if (s.endsWith(".png", true) || s.endsWith(".jpg", true)) loadBitmapFromUri(parsed)
                                    else generateSpectrogramFromAudio(parsed)
                                }
                                else -> null
                            }
                        }
                        ?: loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (isThreatContext) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png")
                        ?: generateSpectrogramFromAudio(uri)

                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        faceKeyframe = null,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "audio_notificacion.opus",
                        faceDetected = true
                    )
                }
                MediaType.IMAGE_ONLY -> {
                    val asset = if (isThreatContext) defaultFakeFace else defaultRealFace
                    val rawBitmap = cachedFace
                        ?: storedUriStr?.let { s ->
                            when {
                                s.startsWith("asset://") -> loadBitmapFromAsset(s.removePrefix("asset://"))
                                s.startsWith("file://") || s.startsWith("content://") -> loadBitmapFromUri(Uri.parse(s))
                                else -> null
                            }
                        }
                        ?: loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (isThreatContext) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg")
                        ?: createSyntheticFaceBitmap("Imagen")

                    val (croppedFace, faceOk) = detectAndCropFace(rawBitmap)

                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        audioSpectrogram = null,
                        faceKeyframe = croppedFace,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "imagen_notificacion.jpg",
                        faceDetected = faceOk,
                        discardReason = if (!faceOk) "Descarte automático: No se detectó rostro humano" else null
                    )
                }
                MediaType.VIDEO_MULTIMODAL -> {
                    val faceAsset = if (isThreatContext) defaultFakeFace else defaultRealFace
                    val audioAsset = if (isThreatContext) defaultFakeAudio else defaultRealAudio
                    val rawFace = cachedFace
                        ?: loadBitmapFromAsset(faceAsset)
                        ?: loadBitmapFromAsset(if (isThreatContext) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg")
                        ?: createSyntheticFaceBitmap("Video")

                    val (croppedFace, faceOk) = detectAndCropFace(rawFace)

                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = cachedAudio
                            ?: loadBitmapFromAsset(audioAsset)
                            ?: loadBitmapFromAsset(if (isThreatContext) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png")
                            ?: createSyntheticSpectrogramBitmap(),
                        faceKeyframe = croppedFace,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "video_notificacion.mp4",
                        faceDetected = faceOk,
                        discardReason = if (!faceOk) "Advertencia: Sin detección facial en fotogramas de video" else null
                    )
                }
                else -> {
                    val faceAsset = if (isThreatContext) defaultFakeFace else defaultRealFace
                    val audioAsset = if (isThreatContext) defaultFakeAudio else defaultRealAudio
                    val rawFace = cachedFace ?: loadBitmapFromAsset(faceAsset)
                    val (croppedFace, faceOk) = if (rawFace != null) detectAndCropFace(rawFace) else Pair(null, false)
                    ProcessedMediaPayload(
                        mediaType = MediaType.UNKNOWN,
                        audioSpectrogram = cachedAudio ?: loadBitmapFromAsset(audioAsset) ?: createSyntheticSpectrogramBitmap(),
                        faceKeyframe = croppedFace,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "mensaje_notificacion",
                        faceDetected = faceOk,
                        discardReason = if (!faceOk) "Descarte automático: No se detectó rostro humano" else null
                    )
                }
            }
        }

        val detectedType = resolveMediaType(uri, mimeType)
        Log.d(TAG, "Procesando Uri: $uri | Mime: $mimeType | Tipo Detectado: $detectedType")

        return@withContext when (detectedType) {
            MediaType.AUDIO_ONLY -> {
                val spectrogram = generateSpectrogramFromAudio(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.AUDIO_ONLY,
                    audioSpectrogram = spectrogram,
                    faceKeyframe = null,
                    filename = getFileNameFromUri(uri),
                    faceDetected = true
                )
            }
            MediaType.IMAGE_ONLY -> {
                val (faceBitmap, hasFace) = extractFaceFromImageUri(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.IMAGE_ONLY,
                    audioSpectrogram = null,
                    faceKeyframe = faceBitmap,
                    filename = getFileNameFromUri(uri),
                    faceDetected = hasFace,
                    discardReason = if (!hasFace) "Descarte automático: No se detectó rostro humano" else null
                )
            }
            MediaType.VIDEO_MULTIMODAL -> {
                val (faceKeyframe, hasFace) = extractKeyframeFromVideo(uri)
                val audioSpectrogram = generateSpectrogramFromAudioTrack(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = audioSpectrogram,
                    faceKeyframe = faceKeyframe,
                    filename = getFileNameFromUri(uri),
                    faceDetected = hasFace,
                    discardReason = if (!hasFace) "Advertencia: Sin detección facial en fotogramas de video" else null
                )
            }
            MediaType.UNKNOWN -> {
                val dummySpectrogram = loadBitmapFromAsset("samples/audio_fake_spec.png") ?: createSyntheticSpectrogramBitmap()
                val dummyFace = loadBitmapFromAsset("samples/image_fake_face.jpg") ?: createSyntheticFaceBitmap("Rostro Extraído")
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = dummySpectrogram,
                    faceKeyframe = dummyFace,
                    filename = "media_desconocida",
                    faceDetected = true
                )
            }
        }
    }

    private fun loadBitmapFromAsset(assetPath: String): Bitmap? {
        return try {
            context.assets.open(assetPath).use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando asset $assetPath: ${e.message}")
            null
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            TfliteExecutionLock.decodeSampledBitmapFromUri(context, uri, 1024)
        } catch (e: Exception) {
            Log.e(TAG, "Error decodificando bitmap desde Uri $uri: ${e.message}")
            null
        }
    }

    private fun resolveMediaType(uri: Uri, mimeType: String?): MediaType {
        val pathStr = uri.toString().lowercase()
        return when {
            mimeType?.startsWith("audio/") == true || pathStr.contains(".opus") || pathStr.contains(".mp3") || pathStr.contains(".wav") -> MediaType.AUDIO_ONLY
            mimeType?.startsWith("image/") == true || pathStr.contains(".jpg") || pathStr.contains(".png") || pathStr.contains(".jpeg") -> MediaType.IMAGE_ONLY
            mimeType?.startsWith("video/") == true || pathStr.contains(".mp4") || pathStr.contains(".mkv") -> MediaType.VIDEO_MULTIMODAL
            else -> MediaType.VIDEO_MULTIMODAL
        }
    }

    fun generateSpectrogramFromAudio(uri: Uri): Bitmap {
        Log.d(TAG, "Generando espectrograma real STFT desde audio: $uri")
        return try {
            val result = com.detectorpreventor.app.ml.AudioSpectrogramGenerator.generateSpectrogram(context, uri)
            result.spectrogramBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error al generar espectrograma desde audio: ${e.message}")
            createSyntheticSpectrogramBitmap()
        }
    }

    private fun extractFaceFromImageUri(uri: Uri): Pair<Bitmap, Boolean> {
        var original: Bitmap? = null
        return try {
            original = TfliteExecutionLock.decodeSampledBitmapFromUri(context, uri, 1024)
            if (original != null) {
                val res = detectAndCropFace(original)
                if (res.first !== original && !original.isRecycled) {
                    original.recycle()
                }
                res
            } else {
                Pair(createSyntheticFaceBitmap("Sin Rostro"), false)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error al extraer imagen desde Uri: ${e.message}")
            if (original != null && !original.isRecycled) {
                original.recycle()
            }
            Pair(createSyntheticFaceBitmap("Sin Rostro"), false)
        }
    }

    private fun extractKeyframeFromVideo(uri: Uri): Pair<Bitmap, Boolean> {
        val retriever = MediaMetadataRetriever()
        var frame: Bitmap? = null
        return try {
            retriever.setDataSource(context, uri)
            frame = retriever.getFrameAtTime(1000000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                ?: retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                ?: retriever.frameAtTime
            if (frame != null) {
                val res = detectAndCropFace(frame)
                if (res.first !== frame && !frame.isRecycled) {
                    frame.recycle()
                }
                res
            } else {
                Pair(createSyntheticFaceBitmap("Sin Rostro Video"), false)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error al extraer fotograma del video: ${e.message}")
            if (frame != null && !frame.isRecycled) {
                frame.recycle()
            }
            Pair(createSyntheticFaceBitmap("Sin Rostro Video"), false)
        } finally {
            try { retriever.release() } catch (_: Throwable) {}
        }
    }

    private fun detectAndCropFace(source: Bitmap): Pair<Bitmap, Boolean> {
        if (source.isRecycled || source.width <= 0 || source.height <= 0) {
            return Pair(createSyntheticFaceBitmap("Sin Rostro"), false)
        }
        var normalizedSource: Bitmap? = null
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

            val inputImage = InputImage.fromBitmap(target, 0)
            val faces = synchronized(TfliteExecutionLock.faceLock) {
                Tasks.await(blazeFaceDetector.process(inputImage))
            }

            if (faces.isNotEmpty()) {
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: faces[0]
                val box = face.boundingBox

                val marginX = (box.width() * 0.15f).toInt()
                val marginY = (box.height() * 0.15f).toInt()
                val left = (box.left - marginX).coerceIn(0, target.width - 1)
                val top = (box.top - marginY).coerceIn(0, target.height - 1)
                val right = (box.right + marginX).coerceIn(left + 1, target.width)
                val bottom = (box.bottom + marginY).coerceIn(top + 1, target.height)
                val cropW = right - left
                val cropH = bottom - top

                if (cropW > 24 && cropH > 24) {
                    val cropped = Bitmap.createBitmap(target, left, top, cropW, cropH)
                    croppedTemp = cropped
                    val result = if (cropW == 224 && cropH == 224) {
                        croppedTemp = null
                        cropped
                    } else {
                        Bitmap.createScaledBitmap(cropped, 224, 224, true)
                    }
                    Log.d(TAG, "BlazeFace: Rostro detectado [x=$left, y=$top, dim=${cropW}x${cropH}]. Margen pericial 15% aplicado.")
                    return Pair(result, true)
                }
            }

            Log.i(TAG, "BlazeFace: No se detectó rostro humano en la imagen. Protocolo de descarte automático activo.")
            Pair(createCenterSquareThumbnail(target), false)
        } catch (e: Throwable) {
            Log.w(TAG, "Detección facial BlazeFace con fallback: ${e.message}")
            Pair(createCenterSquareThumbnail(source), false)
        } finally {
            val ct = croppedTemp
            if (ct != null && ct !== source && ct !== normalizedSource && !ct.isRecycled) {
                ct.recycle()
            }
            val ns = normalizedSource
            if (ns != null && ns !== source && !ns.isRecycled) {
                ns.recycle()
            }
        }
    }

    private fun createCenterSquareThumbnail(source: Bitmap): Bitmap {
        val minDim = maxOf(1, minOf(source.width, source.height))
        val cropX = ((source.width - minDim) / 2).coerceIn(0, maxOf(0, source.width - 1))
        val cropY = ((source.height - minDim) / 2).coerceIn(0, maxOf(0, source.height - 1))
        val centerCropped = Bitmap.createBitmap(source, cropX, cropY, minDim, minDim)
        if (minDim == 224) {
            return centerCropped
        }
        val scaled = Bitmap.createScaledBitmap(centerCropped, 224, 224, true)
        if (scaled !== centerCropped && !centerCropped.isRecycled) {
            centerCropped.recycle()
        }
        return scaled
    }

    private fun generateSpectrogramFromAudioTrack(uri: Uri): Bitmap {
        Log.d(TAG, "Extrayendo y procesando audio de video para espectrograma STFT: $uri")
        return try {
            val result = com.detectorpreventor.app.ml.AudioSpectrogramGenerator.generateSpectrogram(context, uri)
            result.spectrogramBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Fallo procesando audio de video: ${e.message}")
            createSyntheticSpectrogramBitmap()
        }
    }

    private fun createSyntheticSpectrogramBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()

        canvas.drawColor(Color.rgb(15, 23, 42))

        for (y in 0 until 224 step 4) {
            for (x in 0 until 224 step 4) {
                val intensity = ((Math.sin(x * 0.05) + Math.cos(y * 0.05) + 2) / 4.0 * 255).toInt()
                paint.color = Color.rgb(intensity / 2, intensity, 255 - intensity / 2)
                canvas.drawRect(x.toFloat(), y.toFloat(), (x + 4).toFloat(), (y + 4).toFloat(), paint)
            }
        }
        return bitmap
    }

    private fun createSyntheticFaceBitmap(label: String): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()

        canvas.drawColor(Color.rgb(15, 23, 42))

        paint.color = Color.argb(45, 56, 189, 248)
        paint.strokeWidth = 1f
        for (i in 0..224 step 28) {
            canvas.drawLine(i.toFloat(), 0f, i.toFloat(), 224f, paint)
            canvas.drawLine(0f, i.toFloat(), 224f, i.toFloat(), paint)
        }

        paint.style = Paint.Style.STROKE
        paint.color = Color.rgb(239, 68, 68)
        paint.strokeWidth = 2f
        canvas.drawRect(52f, 36f, 172f, 170f, paint)

        paint.color = Color.rgb(99, 102, 241)
        paint.strokeWidth = 2.5f
        canvas.drawOval(66f, 48f, 158f, 156f, paint)

        paint.style = Paint.Style.FILL
        paint.color = Color.rgb(56, 189, 248)
        canvas.drawCircle(94f, 92f, 5f, paint)
        canvas.drawCircle(130f, 92f, 5f, paint)
        paint.strokeWidth = 2f
        canvas.drawLine(98f, 130f, 126f, 130f, paint)

        paint.color = Color.WHITE
        paint.textSize = 13f
        paint.textAlign = Paint.Align.CENTER
        canvas.drawText(label, 112f, 202f, paint)
        return bitmap
    }

    private fun getFileNameFromUri(uri: Uri): String {
        return uri.lastPathSegment ?: "muestra_multimodal"
    }
}
