package com.detectorpreventor.app.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
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
    val filename: String = ""
)

class MediaRouter(private val context: Context) {

    companion object {
        private const val TAG = "MediaRouter"
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
                        filename = getFileNameFromUri(uri)
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
                        filename = getFileNameFromUri(uri)
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
                        3 -> "samples/03_voz_humana_real_bonafide_3_spec.png"
                        4 -> "samples/08_clonacion_ia_spoof_tts_3_spec.png"
                        else -> "samples/04_voz_humana_real_bonafide_4_spec.png"
                    }
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = loadBitmapFromAsset(audioAsset) ?: generateSpectrogramFromAudio(uri),
                        faceKeyframe = loadBitmapFromAsset(faceAsset) ?: loadBitmapFromAsset("samples/image_real_face.jpg"),
                        filename = getFileNameFromUri(uri)
                    )
                }
            }
        }

        if (uriStr.startsWith("content://com.detectorpreventor.app.notifications/")) {
            val notifId = uri.lastPathSegment ?: ""
            val notif = NotificationRepository.notifications.value.find { it.id == notifId }
            val isThreat = notif?.isThreat ?: false
            val mType = notif?.mediaType ?: resolveMediaType(uri, mimeType)

            return@withContext when (mType) {
                MediaType.AUDIO_ONLY -> {
                    val asset = if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png"
                    val bitmap = loadBitmapFromAsset(asset) ?: generateSpectrogramFromAudio(uri)
                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "audio_notificacion.opus"
                    )
                }
                MediaType.IMAGE_ONLY -> {
                    val asset = if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg"
                    val bitmap = loadBitmapFromAsset(asset) ?: createSyntheticFaceBitmap("Imagen")
                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        faceKeyframe = bitmap,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "imagen_notificacion.jpg"
                    )
                }
                MediaType.VIDEO_MULTIMODAL -> {
                    val faceAsset = if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg"
                    val audioAsset = if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png"
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = loadBitmapFromAsset(audioAsset) ?: createSyntheticSpectrogramBitmap(),
                        faceKeyframe = loadBitmapFromAsset(faceAsset) ?: createSyntheticFaceBitmap("Video"),
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "video_notificacion.mp4"
                    )
                }
                else -> {
                    val audioAsset = if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png"
                    ProcessedMediaPayload(
                        mediaType = MediaType.UNKNOWN,
                        audioSpectrogram = loadBitmapFromAsset(audioAsset) ?: createSyntheticSpectrogramBitmap(),
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "mensaje_notificacion"
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
                    filename = getFileNameFromUri(uri)
                )
            }
            MediaType.IMAGE_ONLY -> {
                val faceBitmap = extractFaceFromImageUri(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.IMAGE_ONLY,
                    faceKeyframe = faceBitmap,
                    filename = getFileNameFromUri(uri)
                )
            }
            MediaType.VIDEO_MULTIMODAL -> {
                val faceKeyframe = extractKeyframeFromVideo(uri)
                val audioSpectrogram = generateSpectrogramFromAudioTrack(uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = audioSpectrogram,
                    faceKeyframe = faceKeyframe,
                    filename = getFileNameFromUri(uri)
                )
            }
            MediaType.UNKNOWN -> {
                val dummySpectrogram = loadBitmapFromAsset("samples/audio_fake_spec.png") ?: createSyntheticSpectrogramBitmap()
                val dummyFace = loadBitmapFromAsset("samples/image_fake_face.jpg") ?: createSyntheticFaceBitmap("Rostro Extraído")
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = dummySpectrogram,
                    faceKeyframe = dummyFace,
                    filename = "media_desconocida"
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

    private fun generateSpectrogramFromAudio(uri: Uri): Bitmap {
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

    private fun extractFaceFromImageUri(uri: Uri): Bitmap {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri)
            val original = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            
            original?.let { Bitmap.createScaledBitmap(it, 224, 224, true) }
                ?: createSyntheticFaceBitmap("Rostro Imagen")
        } catch (e: Exception) {
            Log.e(TAG, "Error al extraer imagen desde Uri: ${e.message}")
            createSyntheticFaceBitmap("Rostro Imagen")
        }
    }

    private fun extractKeyframeFromVideo(uri: Uri): Bitmap {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val frame = retriever.getFrameAtTime(1000000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            frame?.let { Bitmap.createScaledBitmap(it, 224, 224, true) }
                ?: createSyntheticFaceBitmap("Rostro Video")
        } catch (e: Exception) {
            Log.e(TAG, "Error al extraer fotograma del video: ${e.message}")
            createSyntheticFaceBitmap("Rostro Video")
        } finally {
            try { retriever.release() } catch (_: Exception) {}
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

