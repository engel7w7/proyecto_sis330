package com.detectorpreventor.app.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import com.detectorpreventor.app.ml.AudioSpectrogramGenerator
import com.detectorpreventor.app.ml.TfliteExecutionLock
import com.detectorpreventor.app.ml.VisionClassifier
import com.detectorpreventor.app.notifications.NotificationMediaCache
import com.detectorpreventor.app.notifications.NotificationRepository
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
    val discardReason: String? = null,
    val faceKeyframes: List<Bitmap> = emptyList(),
    val audioPcm: FloatArray? = null,
    val faceKeyframesQualities: List<Float> = emptyList(),
    val audioConfidence: Float = 1.0f
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
            .setMinFaceSize(0.05f)
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
                    val wavAsset = wavAssetFromSampleSpec(asset)
                    val audioPcm = loadWavPcm48kFromAsset(wavAsset)
                        ?: AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, uri)?.pcm48k
                    val bitmap = loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (index % 2 != 0) "samples/audio_real_spec.png" else "samples/audio_fake_spec.png")
                        ?: audioPcm?.let { AudioSpectrogramGenerator.analyzePcm(it).spectrogramBitmap }
                        ?: generateSpectrogramFromAudio(uri)
                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        faceKeyframe = null,
                        filename = getFileNameFromUri(uri),
                        faceDetected = true,
                        audioPcm = audioPcm
                    )
                }
                "image" -> {
                    val asset = when (index) {
                        1 -> "samples/01_retrato_humano_real_1.jpg"
                        2 -> "samples/10_deepfake_rostro_ia_5.jpg"
                        3 -> "samples/06_deepfake_rostro_ia_1.jpg"
                        4 -> "samples/07_deepfake_rostro_ia_2.jpg"
                        else -> "samples/02_retrato_humano_real_2.jpg"
                    }
                    val bitmap = loadBitmapFromAsset(asset)
                        ?: loadBitmapFromAsset(if (index % 2 != 0) "samples/image_real_face.jpg" else "samples/image_fake_face.jpg")
                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        audioSpectrogram = null,
                        faceKeyframe = bitmap,
                        faceKeyframes = if (bitmap != null) listOf(bitmap) else emptyList(),
                        filename = getFileNameFromUri(uri),
                        faceDetected = bitmap != null
                    )
                }
                else -> {
                    val faceAsset = when (index) {
                        1 -> "samples/01_retrato_humano_real_1.jpg"
                        2 -> "samples/10_deepfake_rostro_ia_5.jpg"
                        3 -> "samples/06_deepfake_rostro_ia_1.jpg"
                        4 -> "samples/07_deepfake_rostro_ia_2.jpg"
                        else -> "samples/03_retrato_humano_real_3.jpg"
                    }
                    val audioAsset = when (index) {
                        1 -> "samples/01_voz_humana_real_bonafide_1_spec.png"
                        2 -> "samples/06_clonacion_ia_spoof_tts_1_spec.png"
                        3 -> "samples/07_clonacion_ia_spoof_tts_2_spec.png"
                        4 -> "samples/08_clonacion_ia_spoof_tts_3_spec.png"
                        else -> "samples/04_voz_humana_real_bonafide_4_spec.png"
                    }
                    val audioWavAsset = wavAssetFromSampleSpec(audioAsset)
                    val audioPcm = loadWavPcm48kFromAsset(audioWavAsset)
                    val audioSpec = loadBitmapFromAsset(audioAsset)
                        ?: audioPcm?.let { AudioSpectrogramGenerator.analyzePcm(it).spectrogramBitmap }
                        ?: generateSpectrogramFromAudio(uri)
                    val faceBmp = loadBitmapFromAsset(faceAsset) ?: loadBitmapFromAsset("samples/image_real_face.jpg")
                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = audioSpec,
                        faceKeyframe = faceBmp,
                        faceKeyframes = if (faceBmp != null) listOf(faceBmp) else emptyList(),
                        filename = getFileNameFromUri(uri),
                        faceDetected = faceBmp != null,
                        audioPcm = audioPcm
                    )
                }
            }
        }

        if (uriStr.startsWith("content://com.detectorpreventor.app.notifications/")) {
            val notifId = uri.lastPathSegment ?: ""
            val notif = NotificationRepository.notifications.value.find { it.id == notifId }
            val mType = notif?.mediaType ?: resolveMediaType(uri, mimeType)

            val storedUriStr = notif?.mediaUriString ?: NotificationMediaCache.getMediaUriString(notifId)
            val isAssetSimulation = storedUriStr?.startsWith("asset://") == true ||
                storedUriStr?.startsWith("synthetic://") == true
            val cachedFace = NotificationMediaCache.getFaceBitmap(context, notifId) ?: NotificationMediaCache.getFaceBitmap(notifId)
            val cachedAudio = NotificationMediaCache.getAudioBitmap(context, notifId) ?: NotificationMediaCache.getAudioBitmap(notifId)
            val cachedAudioPcm = NotificationMediaCache.getAudioPcm(notifId)

            return@withContext when (mType) {
                MediaType.AUDIO_ONLY -> {
                    var bitmap: Bitmap? = cachedAudio
                    var audioPcm: FloatArray? = cachedAudioPcm
                    if (storedUriStr != null && (bitmap == null || audioPcm == null)) {
                        when {
                            storedUriStr.startsWith("asset://") -> {
                                val assetRef = storedUriStr.removePrefix("asset://")
                                val wavAsset = wavAssetFromSampleSpec(assetRef)
                                val specAsset = specAssetFromAudioAsset(assetRef)
                                if (audioPcm == null) {
                                    audioPcm = loadWavPcm48kFromAsset(wavAsset)
                                }
                                if (bitmap == null) {
                                    bitmap = loadBitmapFromAsset(specAsset)
                                        ?: audioPcm?.let { AudioSpectrogramGenerator.analyzePcm(it).spectrogramBitmap }
                                }
                            }
                            storedUriStr.startsWith("file://") || storedUriStr.startsWith("content://") -> {
                                val parsed = Uri.parse(storedUriStr)
                                if (storedUriStr.endsWith(".png", true) || storedUriStr.endsWith(".jpg", true)) {
                                    if (bitmap == null) bitmap = loadBitmapFromUri(parsed)
                                } else {
                                    val audioRes = AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, parsed)
                                    if (bitmap == null) bitmap = audioRes?.spectrogramBitmap
                                    if (audioPcm == null) audioPcm = audioRes?.pcm48k
                                }
                            }
                        }
                    }

                    ProcessedMediaPayload(
                        mediaType = MediaType.AUDIO_ONLY,
                        audioSpectrogram = bitmap,
                        faceKeyframe = null,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "audio_notificacion.opus",
                        faceDetected = true,
                        audioPcm = audioPcm
                    )
                }
                MediaType.IMAGE_ONLY -> {
                    val rawBitmap = cachedFace
                        ?: storedUriStr?.let { s ->
                            when {
                                s.startsWith("asset://") -> loadBitmapFromAsset(s.removePrefix("asset://"))
                                s.startsWith("synthetic://") -> createSyntheticLandscapeBitmap()
                                s.startsWith("file://") || s.startsWith("content://") -> loadBitmapFromUri(Uri.parse(s))
                                else -> null
                            }
                        }

                    val (croppedFace, faceOk) = when {
                        notif?.isFaceDiscarded() == true && cachedFace != null -> Pair(cachedFace, false)
                        cachedFace != null && notif?.fusionResult != null && !notif.isFaceDiscarded() -> Pair(cachedFace, true)
                        rawBitmap != null -> detectAndCropFace(rawBitmap)
                        else -> Pair(null, false)
                    }

                    ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        audioSpectrogram = null,
                        faceKeyframe = croppedFace,
                        faceKeyframes = if (faceOk && croppedFace != null) listOf(croppedFace) else emptyList(),
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "imagen_notificacion.jpg",
                        faceDetected = faceOk,
                        discardReason = if (!faceOk) "Descarte automatico: No se detecto rostro humano" else null
                    )
                }
                MediaType.VIDEO_MULTIMODAL -> {
                    val parsedStoredUri = storedUriStr?.takeIf { it.startsWith("file://") || it.startsWith("content://") }?.let { Uri.parse(it) }
                    val extractedFrames = if (cachedFace == null && parsedStoredUri != null && !storedUriStr.endsWith(".jpg", true) && !storedUriStr.endsWith(".png", true)) {
                        extractKeyframesFromVideo(parsedStoredUri, maxFrames = 5)
                    } else null

                    val rawFace = cachedFace
                        ?: extractedFrames?.first?.firstOrNull()
                        ?: storedUriStr?.takeIf { it.startsWith("asset://") }?.let { loadBitmapFromAsset(it.removePrefix("asset://")) }

                    val (croppedFace, faceOk, keyframesList) = when {
                        extractedFrames != null -> Triple(extractedFrames.first.firstOrNull(), extractedFrames.second, extractedFrames.first)
                        notif?.isFaceDiscarded() == true && cachedFace != null -> Triple(cachedFace, false, emptyList())
                        cachedFace != null && notif?.fusionResult != null && !notif.isFaceDiscarded() -> Triple(cachedFace, true, listOf(cachedFace))
                        rawFace != null -> {
                            val (c, ok) = detectAndCropFace(rawFace, forceTightCrop = false)
                            Triple(c, ok, if (ok) listOf(c) else emptyList())
                        }
                        else -> Triple(null, false, emptyList())
                    }

                    val audioRes = if ((cachedAudio == null || cachedAudioPcm == null) && parsedStoredUri != null) {
                        AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, parsedStoredUri)
                    } else null

                    val pairedAudio = if (isAssetSimulation && cachedAudio == null) {
                        pairedAudioAssetFromVideoAsset(storedUriStr)?.let { loadBitmapFromAsset(it) }
                    } else null

                    val pairedAudioPcm = if (isAssetSimulation && cachedAudioPcm == null) {
                        pairedAudioWavFromVideoAsset(storedUriStr)?.let { loadWavPcm48kFromAsset(it) }
                    } else null

                    val audioSpec = cachedAudio ?: audioRes?.spectrogramBitmap ?: pairedAudio
                    val audioPcm = cachedAudioPcm ?: audioRes?.pcm48k ?: pairedAudioPcm

                    ProcessedMediaPayload(
                        mediaType = MediaType.VIDEO_MULTIMODAL,
                        audioSpectrogram = audioSpec,
                        faceKeyframe = croppedFace,
                        faceKeyframes = keyframesList,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "video_notificacion.mp4",
                        faceDetected = faceOk,
                        discardReason = if (!faceOk) "Advertencia: Sin deteccion facial en fotogramas de video" else null,
                        audioPcm = audioPcm
                    )
                }
                else -> {
                    ProcessedMediaPayload(
                        mediaType = MediaType.UNKNOWN,
                        audioSpectrogram = cachedAudio,
                        faceKeyframe = cachedFace,
                        filename = notif?.let { "${it.appName}: ${it.sender}" } ?: "mensaje_notificacion",
                        faceDetected = cachedFace != null,
                        discardReason = if (cachedFace == null) "Sin contenido multimedia adjunto" else null,
                        audioPcm = cachedAudioPcm
                    )
                }
            }
        }

        val detectedType = resolveMediaType(uri, mimeType)
        Log.d(TAG, "Procesando Uri: $uri | Mime: $mimeType | Tipo Detectado: $detectedType")

        return@withContext when (detectedType) {
            MediaType.AUDIO_ONLY -> {
                val audioRes = AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.AUDIO_ONLY,
                    audioSpectrogram = audioRes?.spectrogramBitmap,
                    faceKeyframe = null,
                    filename = getFileNameFromUri(uri),
                    faceDetected = true,
                    audioPcm = audioRes?.pcmFull16k ?: audioRes?.pcm48k,
                    audioConfidence = audioRes?.audioConfidence ?: 1.0f
                )
            }
            MediaType.IMAGE_ONLY -> {
                val (faceBitmap, hasFace) = extractFaceFromImageUri(uri)
                val qList = if (hasFace) listOf(VisionClassifier.computeLaplacianSharpness(faceBitmap)) else emptyList()
                ProcessedMediaPayload(
                    mediaType = MediaType.IMAGE_ONLY,
                    audioSpectrogram = null,
                    faceKeyframe = faceBitmap,
                    faceKeyframes = if (hasFace) listOf(faceBitmap) else emptyList(),
                    filename = getFileNameFromUri(uri),
                    faceDetected = hasFace,
                    discardReason = if (!hasFace) "Descarte automatico: No se detecto rostro humano" else null,
                    faceKeyframesQualities = qList
                )
            }
            MediaType.VIDEO_MULTIMODAL -> {
                val extRes = VisionClassifier.extractBestFiveVideoKeyframes(
                    context = context,
                    uri = uri,
                    detector = blazeFaceDetector,
                    forceTightCrop = false,
                    maxFrames = 5,
                    onNoFace = { createCenterSquareThumbnail(it) }
                )
                val keyframes = if (extRes.crops.isNotEmpty()) extRes.crops else listOf(createSyntheticFaceBitmap("Sin Rostro Video"))
                val hasFace = extRes.hasFace
                val audioRes = AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, uri)
                ProcessedMediaPayload(
                    mediaType = MediaType.VIDEO_MULTIMODAL,
                    audioSpectrogram = audioRes?.spectrogramBitmap,
                    faceKeyframe = keyframes.firstOrNull(),
                    faceKeyframes = keyframes,
                    filename = getFileNameFromUri(uri),
                    faceDetected = hasFace,
                    discardReason = if (!hasFace) "Advertencia: Sin deteccion facial en fotogramas de video" else null,
                    audioPcm = audioRes?.pcmFull16k ?: audioRes?.pcm48k,
                    faceKeyframesQualities = extRes.qualities,
                    audioConfidence = audioRes?.audioConfidence ?: 1.0f
                )
            }
            MediaType.UNKNOWN -> {
                ProcessedMediaPayload(
                    mediaType = MediaType.UNKNOWN,
                    audioSpectrogram = null,
                    faceKeyframe = null,
                    faceKeyframes = emptyList(),
                    filename = getFileNameFromUri(uri),
                    faceDetected = false,
                    discardReason = "Formato de archivo no compatible o sin contenido audiovisual"
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
        val resolvedMime = (mimeType ?: try { context.contentResolver.getType(uri) } catch (_: Exception) { null })?.lowercase()
        val displayName = getFileNameFromUri(uri).lowercase()
        val ext = displayName.substringAfterLast('.', "").substringBefore('?').trim()
        val pathLower = uri.toString().lowercase()

        val audioExts = setOf("opus", "ogg", "m4a", "aac", "amr", "wav", "flac", "3ga", "mp3", "wma")
        val imageExts = setOf("jpg", "jpeg", "png", "webp", "heic", "heif", "bmp")
        val videoExts = setOf("mp4", "mkv", "3gp", "webm", "avi", "mov", "m4v")

        return when {
            resolvedMime?.startsWith("audio/") == true ||
                resolvedMime == "application/ogg" ||
                ext in audioExts ||
                listOf(".opus", ".ogg", ".m4a", ".aac", ".amr", ".wav", ".flac", ".3ga", ".mp3").any { pathLower.contains(it) } -> MediaType.AUDIO_ONLY
            resolvedMime?.startsWith("image/") == true ||
                ext in imageExts ||
                listOf(".jpg", ".jpeg", ".png", ".webp", ".heic").any { pathLower.contains(it) } -> MediaType.IMAGE_ONLY
            resolvedMime?.startsWith("video/") == true ||
                ext in videoExts ||
                listOf(".mp4", ".mkv", ".3gp", ".webm", ".avi", ".mov").any { pathLower.contains(it) } -> MediaType.VIDEO_MULTIMODAL
            else -> MediaType.UNKNOWN
        }
    }

    fun generateSpectrogramFromAudio(uri: Uri): Bitmap? {
        Log.d(TAG, "Generando espectrograma real STFT desde audio: $uri")
        return try {
            AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, uri)?.spectrogramBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error al generar espectrograma desde audio: ${e.message}")
            null
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

    private fun extractKeyframesFromVideo(uri: Uri, maxFrames: Int = 5): Pair<List<Bitmap>, Boolean> {
        val extRes = VisionClassifier.extractBestFiveVideoKeyframes(
            context = context,
            uri = uri,
            detector = blazeFaceDetector,
            forceTightCrop = false,
            maxFrames = maxFrames,
            onNoFace = { createCenterSquareThumbnail(it) }
        )
        return if (extRes.crops.isNotEmpty()) {
            Pair(extRes.crops, extRes.hasFace)
        } else {
            Pair(listOf(createSyntheticFaceBitmap("Sin Rostro Video")), false)
        }
    }

    private fun extractKeyframeFromVideo(uri: Uri): Pair<Bitmap, Boolean> {
        val (frames, hasFace) = extractKeyframesFromVideo(uri, maxFrames = 1)
        return Pair(frames.firstOrNull() ?: createSyntheticFaceBitmap("Sin Rostro Video"), hasFace)
    }

    private fun detectAndCropFace(source: Bitmap, forceTightCrop: Boolean = false): Pair<Bitmap, Boolean> {
        if (source.isRecycled || source.width <= 0 || source.height <= 0) {
            return Pair(createSyntheticFaceBitmap("Sin Rostro"), false)
        }
        return VisionClassifier.detectAndCropFaceWithDetector(
            detector = blazeFaceDetector,
            source = source,
            targetSize = 224,
            forceTightCrop = forceTightCrop,
            onNoFace = { createCenterSquareThumbnail(it) }
        )
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

    private fun generateSpectrogramFromAudioTrack(uri: Uri): Bitmap? {
        Log.d(TAG, "Extrayendo y procesando audio de video para espectrograma STFT: $uri")
        return try {
            AudioSpectrogramGenerator.generateSpectrogramIfAudioPresent(context, uri)?.spectrogramBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Fallo procesando audio de video: ${e.message}")
            null
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

    private fun loadWavPcm48kFromAsset(assetPath: String): FloatArray? {
        return try {
            context.assets.open(assetPath).use { inputStream ->
                AudioSpectrogramGenerator.buildPcm48kFromWavBytes(inputStream.readBytes())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando audio WAV asset $assetPath: ${e.message}")
            null
        }
    }

    private fun wavAssetFromSampleSpec(assetPath: String): String {
        return when {
            assetPath.endsWith(".wav", ignoreCase = true) -> assetPath
            assetPath == "samples/audio_real_spec.png" -> "samples/01_voz_humana_real_bonafide_1.wav"
            assetPath == "samples/audio_fake_spec.png" -> "samples/06_clonacion_ia_spoof_tts_1.wav"
            assetPath.endsWith("_spec.png") -> assetPath.removeSuffix("_spec.png") + ".wav"
            else -> "samples/01_voz_humana_real_bonafide_1.wav"
        }
    }

    private fun specAssetFromAudioAsset(assetPath: String): String {
        return if (assetPath.endsWith(".wav", ignoreCase = true)) {
            assetPath.removeSuffix(".wav") + "_spec.png"
        } else {
            assetPath
        }
    }

    private fun pairedAudioAssetFromVideoAsset(storedUriStr: String?): String? {
        if (storedUriStr.isNullOrBlank()) return null
        return if (storedUriStr.contains("06_deepfake")) {
            "samples/06_clonacion_ia_spoof_tts_1_spec.png"
        } else if (storedUriStr.contains("01_retrato")) {
            "samples/01_voz_humana_real_bonafide_1_spec.png"
        } else {
            null
        }
    }

    private fun pairedAudioWavFromVideoAsset(storedUriStr: String?): String? {
        if (storedUriStr.isNullOrBlank()) return null
        return if (storedUriStr.contains("06_deepfake")) {
            "samples/06_clonacion_ia_spoof_tts_1.wav"
        } else if (storedUriStr.contains("01_retrato")) {
            "samples/01_voz_humana_real_bonafide_1.wav"
        } else {
            null
        }
    }

    private fun createSyntheticLandscapeBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()
        canvas.drawColor(Color.rgb(12, 74, 110))
        paint.color = Color.rgb(3, 105, 161)
        canvas.drawRect(0f, 70f, 224f, 140f, paint)
        val mountainPath = android.graphics.Path().apply {
            moveTo(0f, 175f)
            lineTo(55f, 95f)
            lineTo(118f, 155f)
            lineTo(175f, 88f)
            lineTo(224f, 175f)
            close()
        }
        paint.color = Color.rgb(30, 41, 59)
        canvas.drawPath(mountainPath, paint)
        paint.color = Color.rgb(20, 83, 45)
        canvas.drawRect(0f, 175f, 224f, 224f, paint)
        return bitmap
    }

    private fun createSyntheticFaceBitmap(label: String): Bitmap {
        val bitmap = createSyntheticLandscapeBitmap()
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 13f
            textAlign = Paint.Align.CENTER
        }
        canvas.drawText(label, 112f, 206f, paint)
        return bitmap
    }

    private fun getFileNameFromUri(uri: Uri): String {
        var result: String? = null
        if (uri.scheme == "content") {
            try {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (idx >= 0) {
                            result = cursor.getString(idx)
                        }
                    }
                }
            } catch (_: Exception) {
            }
        }
        return result?.takeIf { it.isNotBlank() } ?: uri.lastPathSegment ?: "muestra_multimodal"
    }
}
