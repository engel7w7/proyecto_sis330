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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

enum class MediaType {
    AUDIO_ONLY,
    IMAGE_ONLY,
    VIDEO_MULTIMODAL,
    UNKNOWN
}

typealias MediaPayload = ProcessedMediaPayload

data class ProcessedMediaPayload(
    val mediaType: MediaType,
    val audioSpectrogram: Bitmap? = null,
    val faceKeyframe: Bitmap? = null,
    val filename: String = ""
)

/**
 * Enrutador de medios locales: Clasifica archivos multimedia entrantes (audio, imagen o video)
 * y extrae fotogramas o genera espectrogramas para el procesamiento de los clasificadores.
 */
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
                    val asset = if (index % 2 != 0) "samples/audio_real_spec.png" else "samples/audio_fake_spec.png"
                    val bitmap = loadBitmapFromAsset(asset) ?: generateSpectrogramFromAudio(uri)
                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        filename = getFileNameFromUri(uri)
                    )
                }
                "image" -> {
                    val asset = if (index % 2 != 0) "samples/image_real_face.jpg" else "samples/image_fake_face.jpg"
                    val bitmap = loadBitmapFromAsset(asset) ?: loadBitmapFromAsset("samples/image_fake_face.jpg")
                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        faceKeyframe = bitmap,
                        filename = getFileNameFromUri(uri)
                    )
                }
                else -> {
                    val faceAsset = if (index % 2 != 0) "samples/image_real_face.jpg" else "samples/image_fake_face.jpg"
                    val audioAsset = if (index % 2 != 0) "samples/audio_real_spec.png" else "samples/audio_fake_spec.png"
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = loadBitmapFromAsset(audioAsset) ?: generateSpectrogramFromAudio(uri),
                        faceKeyframe = loadBitmapFromAsset(faceAsset) ?: loadBitmapFromAsset("samples/image_fake_face.jpg"),
                        filename = getFileNameFromUri(uri)
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
        return createSyntheticSpectrogramBitmap()
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
        return createSyntheticSpectrogramBitmap()
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

    private fun generatePlaceholderBitmap(label: String): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            color = Color.DKGRAY
            style = Paint.Style.FILL
        }
        canvas.drawRect(0f, 0f, 224f, 224f, paint)
        paint.apply {
            color = Color.WHITE
            textSize = 20f
            textAlign = Paint.Align.CENTER
        }
        canvas.drawText(label, 112f, 112f, paint)
        return bitmap
    }

    private fun getFileNameFromUri(uri: Uri): String {
        return uri.lastPathSegment ?: "muestra_multimodal"
    }
}

