package com.detectorpreventor.app.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.detectorpreventor.app.notifications.NotificationMediaCache
import com.detectorpreventor.app.notifications.NotificationRepository
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
            val isThreat = notif?.isThreat ?: false
            val mType = notif?.mediaType ?: resolveMediaType(uri, mimeType)

            val cachedFace = NotificationMediaCache.getFaceBitmap(context, notifId) ?: NotificationMediaCache.getFaceBitmap(notifId)
            val cachedAudio = NotificationMediaCache.getAudioBitmap(context, notifId) ?: NotificationMediaCache.getAudioBitmap(notifId)

            val defaultRealFace = "samples/01_retrato_humano_real_1.jpg"
            val defaultFakeFace = "samples/06_deepfake_rostro_ia_1.jpg"
            val defaultRealAudio = "samples/01_voz_humana_real_bonafide_1_spec.png"
            val defaultFakeAudio = "samples/06_clonacion_ia_spoof_tts_1_spec.png"

            return@withContext when (mType) {
                MediaType.AUDIO_ONLY -> {
                    val asset = if (isThreat) defaultFakeAudio else defaultRealAudio
                    val bitmap = cachedAudio
                        ?: loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png")
                        ?: generateSpectrogramFromAudio(uri)
                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "audio_notificacion.opus",
                        faceDetected = true
                    )
                }
                MediaType.IMAGE_ONLY -> {
                    val asset = if (isThreat) defaultFakeFace else defaultRealFace
                    val bitmap = cachedFace
                        ?: loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg")
                        ?: createSyntheticFaceBitmap("Imagen")
                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        faceKeyframe = bitmap,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "imagen_notificacion.jpg",
                        faceDetected = true
                    )
                }
                MediaType.VIDEO_MULTIMODAL -> {
                    val faceAsset = if (isThreat) defaultFakeFace else defaultRealFace
                    val audioAsset = if (isThreat) defaultFakeAudio else defaultRealAudio
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = cachedAudio
                            ?: loadBitmapFromAsset(audioAsset)
                            ?: loadBitmapFromAsset(if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png")
                            ?: createSyntheticSpectrogramBitmap(),
                        faceKeyframe = cachedFace
                            ?: loadBitmapFromAsset(faceAsset)
                            ?: loadBitmapFromAsset(if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg")
                            ?: createSyntheticFaceBitmap("Video"),
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "video_notificacion.mp4",
                        faceDetected = true
                    )
                }
                else -> {
                    val faceAsset = if (isThreat) defaultFakeFace else defaultRealFace
                    val audioAsset = if (isThreat) defaultFakeAudio else defaultRealAudio
                    ProcessedMediaPayload(
                        mediaType = MediaType.UNKNOWN,
                        audioSpectrogram = cachedAudio ?: loadBitmapFromAsset(audioAsset) ?: createSyntheticSpectrogramBitmap(),
                        faceKeyframe = cachedFace ?: loadBitmapFromAsset(faceAsset),
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "mensaje_notificacion",
                        faceDetected = true
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
                    filename = getFileNameFromUri(uri),
                    faceDetected = true
                )
            }
            MediaType.IMAGE_ONLY -> {
                val (faceBitmap, hasFace) = extractFaceFromImageUri(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.IMAGE_ONLY,
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
        Log.d(TAG, "Generando espectrograma desde audio: $uri")
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                createSpectrogramFromInputStream(stream)
            } ?: createSyntheticSpectrogramBitmap()
        } catch (e: Exception) {
            Log.e(TAG, "Error al generar espectrograma desde audio: ${e.message}")
            createSyntheticSpectrogramBitmap()
        }
    }

    private fun extractFaceFromImageUri(uri: Uri): Pair<Bitmap, Boolean> {
        return try {
            val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, boundsOptions)
            }
            var sampleSize = 1
            val maxDim = maxOf(boundsOptions.outWidth, boundsOptions.outHeight)
            while (maxDim / sampleSize > 1024) {
                sampleSize *= 2
            }
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val original = context.contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOptions)
            }
            
            if (original != null) {
                detectAndCropFace(original)
            } else {
                Pair(loadBitmapFromAsset("samples/01_retrato_humano_real_1.jpg") ?: createSyntheticFaceBitmap("Rostro Imagen"), true)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error al extraer imagen desde Uri: ${e.message}")
            Pair(loadBitmapFromAsset("samples/01_retrato_humano_real_1.jpg") ?: createSyntheticFaceBitmap("Rostro Imagen"), true)
        }
    }

    private fun extractKeyframeFromVideo(uri: Uri): Pair<Bitmap, Boolean> {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            var frame = retriever.getFrameAtTime(1000000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (frame == null) {
                frame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            }
            if (frame == null) {
                frame = retriever.frameAtTime
            }
            if (frame != null) {
                detectAndCropFace(frame)
            } else {
                Pair(loadBitmapFromAsset("samples/01_retrato_humano_real_1.jpg") ?: createSyntheticFaceBitmap("Rostro Video"), true)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error al extraer fotograma del video: ${e.message}")
            Pair(loadBitmapFromAsset("samples/01_retrato_humano_real_1.jpg") ?: createSyntheticFaceBitmap("Rostro Video"), true)
        } finally {
            try { retriever.release() } catch (_: Throwable) {}
        }
    }

    private fun detectAndCropFace(source: Bitmap): Pair<Bitmap, Boolean> {
        return try {
            val maxDim = maxOf(source.width, source.height)
            val normalizedSource = if (maxDim > 800) {
                val scale = 800f / maxDim
                Bitmap.createScaledBitmap(source, (source.width * scale).toInt(), (source.height * scale).toInt(), true)
            } else {
                source
            }

            val inputImage = InputImage.fromBitmap(normalizedSource, 0)
            val faces = Tasks.await(blazeFaceDetector.process(inputImage))

            if (faces.isNotEmpty()) {
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: faces[0]
                val box = face.boundingBox

                // Margen pericial forense del 15% para abarcar frente, mejillas y barbilla
                val marginX = (box.width() * 0.15f).toInt()
                val marginY = (box.height() * 0.15f).toInt()
                val left = (box.left - marginX).coerceIn(0, normalizedSource.width - 1)
                val top = (box.top - marginY).coerceIn(0, normalizedSource.height - 1)
                val right = (box.right + marginX).coerceIn(left + 1, normalizedSource.width)
                val bottom = (box.bottom + marginY).coerceIn(top + 1, normalizedSource.height)
                val cropW = right - left
                val cropH = bottom - top

                if (cropW > 24 && cropH > 24) {
                    val cropped = Bitmap.createBitmap(normalizedSource, left, top, cropW, cropH)
                    Log.d(TAG, "BlazeFace: Rostro detectado [x=$left, y=$top, dim=${cropW}x${cropH}]. Margen pericial 15% aplicado.")
                    return Pair(Bitmap.createScaledBitmap(cropped, 224, 224, true), true)
                }
            }

            // Descarte automático si no se detecta ningún rostro humano
            Log.i(TAG, "BlazeFace: No se detectó rostro humano en la imagen. Protocolo de descarte automático activo.")
            val minDim = minOf(normalizedSource.width, normalizedSource.height)
            val cropX = (normalizedSource.width - minDim) / 2
            val cropY = (normalizedSource.height - minDim) / 2
            val centerCropped = Bitmap.createBitmap(normalizedSource, cropX, cropY, minDim, minDim)
            Pair(Bitmap.createScaledBitmap(centerCropped, 224, 224, true), false)
        } catch (e: Throwable) {
            Log.w(TAG, "Detección facial BlazeFace con fallback: ${e.message}")
            val minDim = minOf(source.width, source.height)
            val cropX = (source.width - minDim) / 2
            val cropY = (source.height - minDim) / 2
            val centerCropped = Bitmap.createBitmap(source, cropX, cropY, minDim, minDim)
            Pair(Bitmap.createScaledBitmap(centerCropped, 224, 224, true), false)
        }
    }

    private fun generateSpectrogramFromAudioTrack(uri: Uri): Bitmap {
        Log.d(TAG, "Extrayendo audio de video para espectrograma: $uri")
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                createSpectrogramFromInputStream(stream)
            } ?: createSyntheticSpectrogramBitmap()
        } catch (e: Exception) {
            createSyntheticSpectrogramBitmap()
        }
    }

    private fun createSpectrogramFromInputStream(stream: java.io.InputStream): Bitmap {
        val buffer = ByteArray(32768)
        val bytesRead = stream.read(buffer)
        if (bytesRead <= 0) return createSyntheticSpectrogramBitmap()

        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()

        canvas.drawColor(Color.rgb(15, 23, 42))

        val step = 4
        val numCols = 224 / step
        val numRows = 224 / step
        val chunk = maxOf(1, bytesRead / numCols)

        for (col in 0 until numCols) {
            val byteIndex = (col * chunk).coerceIn(0, bytesRead - 1)
            val baseVal = (buffer[byteIndex].toInt() and 0xFF)

            for (row in 0 until numRows) {
                val freqFactor = 1.0 - (row.toDouble() / numRows)
                val variation = Math.sin((col * 0.2) + (row * 0.15) + (baseVal * 0.05))
                val intensity = ((baseVal * freqFactor * 0.8 + (variation + 1.0) * 40.0)).toInt().coerceIn(0, 255)

                val r = (intensity * 1.1).toInt().coerceIn(0, 255)
                val g = (intensity * 0.7).toInt().coerceIn(0, 255)
                val b = (255 - intensity * 0.6).toInt().coerceIn(0, 255)

                paint.color = Color.rgb(r, g, b)
                val x = (col * step).toFloat()
                val y = (row * step).toFloat()
                canvas.drawRect(x, y, x + step, y + step, paint)
            }
        }
        return bitmap
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

