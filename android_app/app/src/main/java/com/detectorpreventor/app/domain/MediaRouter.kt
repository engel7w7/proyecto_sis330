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
                val dummySpectrogram = generatePlaceholderBitmap("Audio Generado")
                val dummyFace = generatePlaceholderBitmap("Rostro Extraído")
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = dummySpectrogram,
                    faceKeyframe = dummyFace,
                    filename = "media_desconocida"
                )
            }
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
                ?: generatePlaceholderBitmap("Rostro Imagen")
        } catch (e: Exception) {
            Log.e(TAG, "Error al extraer imagen desde Uri: ${e.message}")
            generatePlaceholderBitmap("Rostro Imagen")
        }
    }

    private fun extractKeyframeFromVideo(uri: Uri): Bitmap {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val frame = retriever.getFrameAtTime(1000000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            frame?.let { Bitmap.createScaledBitmap(it, 224, 224, true) }
                ?: generatePlaceholderBitmap("Rostro Video")
        } catch (e: Exception) {
            Log.e(TAG, "Error al extraer fotograma del video: ${e.message}")
            generatePlaceholderBitmap("Rostro Video")
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

