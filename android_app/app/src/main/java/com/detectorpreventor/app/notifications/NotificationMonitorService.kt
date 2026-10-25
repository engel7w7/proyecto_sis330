package com.detectorpreventor.app.notifications

import android.app.Notification
import android.content.ComponentName
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaRouter
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

class NotificationMonitorService : NotificationListenerService() {

    companion object {
        private const val TAG = "NotificationMonitor"

        fun ensureServiceBound(context: Context) {
            if (!NotificationRepository.isPermissionGranted(context)) {
                NotificationRepository.setServiceConnected(false)
                return
            }

            try {
                val intent = Intent(context, NotificationMonitorService::class.java)
                context.startService(intent)
            } catch (e: Exception) {
                Log.w(TAG, "No se pudo invocar startService: ${e.message}")
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                try {
                    val componentName = ComponentName(context, NotificationMonitorService::class.java)
                    requestRebind(componentName)
                    Log.i(TAG, "Revinculacion solicitada correctamente via requestRebind para NotificationMonitorService.")
                } catch (e: Exception) {
                    Log.w(TAG, "Fallo al solicitar requestRebind: ${e.message}")
                }
            }
        }
    }

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.Default + serviceJob)

    private var visionClassifier: VisionClassifier? = null
    private var audioClassifier: AudioClassifier? = null

    override fun onCreate() {
        super.onCreate()
        NotificationRepository.init(applicationContext)
        try {
            visionClassifier = VisionClassifier(applicationContext)
            audioClassifier = AudioClassifier(applicationContext)
            Log.i(TAG, "Modelos TFLite inicializados para inferencia en NotificationMonitorService.")
        } catch (e: Exception) {
            Log.e(TAG, "Error inicializando clasificadores TFLite: ${e.message}")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                requestRebind(ComponentName(this, NotificationMonitorService::class.java))
            } catch (e: Exception) {
            }
        }
        return START_STICKY
    }

    override fun onListenerConnected() {
        super.onListenerConnected()
        Log.i(TAG, "NotificationListenerService CONECTADO y vinculado por el SO.")
        NotificationRepository.setServiceConnected(true)
    }

    override fun onListenerDisconnected() {
        super.onListenerDisconnected()
        Log.w(TAG, "NotificationListenerService DESCONECTADO por el SO. Programando re-vinculacion...")
        NotificationRepository.setServiceConnected(false)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            serviceScope.launch {
                delay(1500)
                try {
                    requestRebind(ComponentName(applicationContext, NotificationMonitorService::class.java))
                    Log.i(TAG, "Auto-revinculacion completada tras desconexion.")
                } catch (e: Exception) {
                    Log.e(TAG, "Fallo en auto-revinculacion: ${e.message}")
                }
            }
        }
    }

    override fun onDestroy() {
        serviceJob.cancel()
        try {
            visionClassifier?.close()
            audioClassifier?.close()
            visionClassifier = null
            audioClassifier = null
            Log.i(TAG, "Clasificadores TFLite cerrados en NotificationMonitorService.")
        } catch (e: Exception) {
            Log.e(TAG, "Error liberando clasificadores: ${e.message}")
        }
        NotificationRepository.setServiceConnected(false)
        super.onDestroy()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        super.onNotificationPosted(sbn)
        if (sbn == null || !NotificationRepository.isMonitoringActive.value) return

        serviceScope.launch {
            try {
                processNotificationSafely(sbn)
            } catch (t: Throwable) {
                Log.e(TAG, "Excepcion no controlada en processNotificationSafely: ${t.message}", t)
            }
        }
    }

    private fun processNotificationSafely(sbn: StatusBarNotification) {
        val pkgName = sbn.packageName ?: return
        val isWhatsApp = pkgName.contains("whatsapp", ignoreCase = true)
        val isTelegram = pkgName.contains("telegram", ignoreCase = true)
        val isMessaging = pkgName.contains("messaging", ignoreCase = true) ||
                pkgName.contains("orca", ignoreCase = true)

        if (!isWhatsApp && !isTelegram && !isMessaging) return

        val notification = sbn.notification ?: return
        val extras = notification.extras ?: return

        val rawTitle = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim()
        val bigTitle = extras.getCharSequence(Notification.EXTRA_TITLE_BIG)?.toString()?.trim()
        val conversationTitle = extras.getCharSequence("android.conversationTitle")?.toString()?.trim()
        val subText = extras.getCharSequence(Notification.EXTRA_SUB_TEXT)?.toString()?.trim()
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim() ?: ""
        val ticker = notification.tickerText?.toString()?.trim() ?: ""
        val textLines = (extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES) ?: emptyArray())
            .mapNotNull { it?.toString()?.trim() }
            .filter { it.isNotBlank() }

        if (isWhatsApp && (rawTitle.isNullOrBlank() || rawTitle.equals("WhatsApp", ignoreCase = true)) &&
            (text.contains("mensajes de", ignoreCase = true) || text.contains("messages from", ignoreCase = true))) {
            Log.d(TAG, "Notificacion ignorada por ser resumen multi-chat generico de WhatsApp ($text)")
            return
        }

        if ((notification.flags and Notification.FLAG_GROUP_SUMMARY) != 0 && (rawTitle.isNullOrBlank() || rawTitle.equals("WhatsApp", ignoreCase = true))) {
            Log.d(TAG, "Notificacion ignorada por FLAG_GROUP_SUMMARY generico ($pkgName)")
            return
        }

        val styleMessages = mutableListOf<String>()
        var messageSender: String? = null
        var isMediaFromMessagingStyle = false
        var messagingStyleMediaType = MediaType.UNKNOWN

        val messagesBundleArray = extras.getParcelableArray("android.messages")
            ?: extras.getParcelableArray(Notification.EXTRA_MESSAGES)

        if (messagesBundleArray != null) {
            for (p in messagesBundleArray) {
                if (p is android.os.Bundle) {
                    val mText = p.getCharSequence("text")?.toString()?.trim()
                    if (!mText.isNullOrBlank()) {
                        styleMessages.add(mText)
                    }
                    val senderPerson = p.getCharSequence("sender")?.toString()?.trim()
                    if (!senderPerson.isNullOrBlank()) {
                        messageSender = senderPerson
                    }
                    val mimeType = p.getString("type") ?: p.getString("data_type")
                    if (!mimeType.isNullOrBlank()) {
                        if (mimeType.startsWith("audio", ignoreCase = true)) {
                            isMediaFromMessagingStyle = true
                            messagingStyleMediaType = MediaType.AUDIO_ONLY
                            styleMessages.add("Nota de voz adjunta ($mimeType)")
                        } else if (mimeType.startsWith("video", ignoreCase = true)) {
                            isMediaFromMessagingStyle = true
                            messagingStyleMediaType = MediaType.VIDEO_MULTIMODAL
                            styleMessages.add("Video adjunto ($mimeType)")
                        } else if (mimeType.startsWith("image", ignoreCase = true)) {
                            if (mimeType.contains("webp", ignoreCase = true)) {
                                styleMessages.add("sticker")
                            } else {
                                isMediaFromMessagingStyle = true
                                messagingStyleMediaType = MediaType.IMAGE_ONLY
                                styleMessages.add("Imagen adjunta ($mimeType)")
                            }
                        }
                    }
                }
            }
        }

        val allContentList = mutableListOf<String>()
        if (text.isNotBlank()) allContentList.add(text)
        if (bigText.isNotBlank()) allContentList.add(bigText)
        allContentList.addAll(textLines)
        allContentList.addAll(styleMessages)
        if (ticker.isNotBlank()) allContentList.add(ticker)

        val combinedContent = allContentList.distinct().joinToString(" ")

        val isSystemStatusNotification = sbn.isOngoing && (
                combinedContent.contains("WhatsApp Web", ignoreCase = true) ||
                combinedContent.contains("copia de seguridad", ignoreCase = true) ||
                combinedContent.contains("backup", ignoreCase = true) ||
                combinedContent.contains("buscando mensajes", ignoreCase = true) ||
                combinedContent.contains("llamada en curso", ignoreCase = true)
        )
        if (isSystemStatusNotification) return

        if (combinedContent.isBlank() && rawTitle.isNullOrBlank()) return

        val isSticker = combinedContent.contains("sticker", ignoreCase = true) ||
                styleMessages.any { it.contains("sticker", ignoreCase = true) }

        if (isSticker) {
            Log.d(TAG, "Notificacion ignorada: Sticker detectado.")
            return
        }

        val hasPictureExtra = extras.get(Notification.EXTRA_PICTURE) != null ||
                extras.get("android.pictureIcon") != null

        val urlPattern = Regex("""(?i)\b(?:https?://|www\.)\S+""")
        val hasUrl = urlPattern.containsMatchIn(combinedContent) ||
                combinedContent.contains("http://", ignoreCase = true) ||
                combinedContent.contains("https://", ignoreCase = true)

        val isVideoRelated = (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.VIDEO_MULTIMODAL) ||
                (!hasUrl && (
                    combinedContent.contains("videollamada", ignoreCase = true) ||
                    combinedContent.contains("videonota", ignoreCase = true) ||
                    Regex("""(?i)\b(?:video|vídeo)\b""").containsMatchIn(combinedContent)
                ))

        val isDurationOnly = Regex("""^\s*\d{1,2}:\d{2}\s*$""").matches(text.trim())
        val isVoiceNoteExplicit = Regex("""(?i)\b(?:mensaje de voz|nota de voz|voice message|voice note)\b""").containsMatchIn(combinedContent)
        val isWhatsAppAudioHeader = !hasUrl && (
            text.equals("Audio", ignoreCase = true) ||
            text.startsWith("Audio (", ignoreCase = true) ||
            Regex("""(?i)\b(?:audio)\s*\(\d{1,2}:\d{2}\)""").containsMatchIn(combinedContent)
        )

        val isAudioRelated = !isVideoRelated && !hasUrl && (
                (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.AUDIO_ONLY) ||
                isDurationOnly ||
                isVoiceNoteExplicit ||
                isWhatsAppAudioHeader
        )

        val isPhotoExplicit = !hasUrl && (
            text.equals("Foto", ignoreCase = true) ||
            text.equals("Imagen", ignoreCase = true) ||
            text.startsWith("Foto (", ignoreCase = true) ||
            text.startsWith("Imagen (", ignoreCase = true) ||
            combinedContent.contains("fotografía", ignoreCase = true)
        )
        val isImageRelated = !isVideoRelated && !isAudioRelated && (
                (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.IMAGE_ONLY) ||
                (hasPictureExtra && !hasUrl) ||
                isPhotoExplicit
        )

        if (!isVideoRelated && !isAudioRelated && !isImageRelated) {
            Log.d(TAG, "Notificacion descartada: Mensaje de texto o enlace sin contenido biometrico audiovisual.")
            return
        }

        val candidateSender = when {
            !messageSender.isNullOrBlank() && !messageSender.equals("WhatsApp", ignoreCase = true) -> messageSender
            !conversationTitle.isNullOrBlank() && !conversationTitle.equals("WhatsApp", ignoreCase = true) -> conversationTitle
            !rawTitle.isNullOrBlank() && !rawTitle.equals("WhatsApp", ignoreCase = true) -> rawTitle
            !bigTitle.isNullOrBlank() && !bigTitle.equals("WhatsApp", ignoreCase = true) -> bigTitle
            !subText.isNullOrBlank() && !subText.equals("WhatsApp", ignoreCase = true) -> subText
            else -> "Contacto Desconocido"
        }

        val phonePattern = Regex("""^(\+?\d[\d\s\-()]{6,}\d)""")
        val isPhone = phonePattern.containsMatchIn(candidateSender) || candidateSender.startsWith("+")
        val isUnregistered = isPhone ||
                candidateSender.startsWith("~") ||
                candidateSender.contains("desconocido", ignoreCase = true) ||
                (candidateSender.count { it.isDigit() } >= 6)

        val sender = if (isUnregistered && !candidateSender.startsWith("[No Registrado]")) {
            "[No Registrado] $candidateSender"
        } else {
            candidateSender
        }

        val appName = when {
            isWhatsApp -> "WhatsApp"
            isTelegram -> "Telegram"
            else -> "Mensajeria"
        }

        val vClassifier = visionClassifier ?: VisionClassifier(applicationContext).also { visionClassifier = it }
        val aClassifier = audioClassifier ?: AudioClassifier(applicationContext).also { audioClassifier = it }

        val mediaType: MediaType
        val audioProb: Float?
        val visionProb: Float?
        val displayText: String
        var cachedFace: Bitmap? = null
        var cachedAudio: Bitmap? = null
        var computedFusionResult: FusionResult? = null

        var isFaceDetected = true

        when {
            isImageRelated -> {
                mediaType = MediaType.IMAGE_ONLY
                var realPicture = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = false)
                if (realPicture == null && isWhatsApp) {
                    try {
                        Thread.sleep(300)
                        realPicture = findRecentWhatsAppImage()
                    } catch (e: Exception) {
                    }
                }

                if (realPicture != null) {
                    cachedFace = realPicture
                    isFaceDetected = vClassifier.checkFacePresence(realPicture)
                    visionProb = if (isFaceDetected) {
                        Log.i(TAG, "Ejecutando inferencia TFLite real sobre imagen capturada de la notificacion...")
                        vClassifier.classifyFaceKeyframe(realPicture, null, requireFaceDetection = true)
                    } else {
                        Log.i(TAG, "Descarte automatico: Notificacion de imagen sin rostro humano detectado.")
                        0.0f
                    }
                    audioProb = null
                    val isThreatScore = (visionProb ?: 0f) >= 0.5f
                    displayText = if (!isFaceDetected) {
                        "Fotografía recibida (Descarte automático: Sin rostro humano)"
                    } else if (isThreatScore) {
                        "Fotografía sospechosa (Alerta: Posible FaceSwap / Deepfake)"
                    } else {
                        "Fotografía entrante analizada (Rostro humano auténtico)"
                    }
                } else {
                    cachedFace = null
                    audioProb = null
                    visionProb = null
                    displayText = "Fotografía entrante (Pendiente de descarga)"
                    computedFusionResult = FusionResult(
                        globalRiskPercentage = 0.0f,
                        audioFraudProb = 0.0f,
                        visionFraudProb = 0.0f,
                        riskLevel = RiskLevel.BAJO,
                        weightAudio = 0.0f,
                        weightVision = 1.0f,
                        diagnosticSummary = "Fotografía recibida. En espera de apertura para inspección facial con EfficientNet-B0."
                    )
                }
            }
            isAudioRelated -> {
                mediaType = MediaType.AUDIO_ONLY
                val realAudioUri = if (isWhatsApp) findRecentWhatsAppAudioUri() else null
                val realSpecBitmap = realAudioUri?.let { uri ->
                    try {
                        val router = MediaRouter(applicationContext)
                        router.generateSpectrogramFromAudio(uri)
                    } catch (e: Exception) {
                        Log.w(TAG, "Error generando espectrograma de audio real: ${e.message}")
                        null
                    }
                }

                if (realSpecBitmap != null) {
                    cachedAudio = realSpecBitmap
                    Log.i(TAG, "Ejecutando inferencia TFLite real sobre espectrograma de audio recibido...")
                    audioProb = aClassifier.classifySpectrogram(realSpecBitmap, null)
                    visionProb = null
                    val isThreatScore = (audioProb ?: 0f) >= 0.5f
                    displayText = if (isThreatScore) {
                        "Nota de voz sospechosa (Alerta: Posible clonación sintética)"
                    } else {
                        "Nota de voz entrante analizada (Voz humana auténtica)"
                    }
                } else {
                    cachedAudio = null
                    audioProb = null
                    visionProb = null
                    displayText = "Nota de voz entrante (Pendiente de descarga)"
                    computedFusionResult = FusionResult(
                        globalRiskPercentage = 0.0f,
                        audioFraudProb = 0.0f,
                        visionFraudProb = 0.0f,
                        riskLevel = RiskLevel.BAJO,
                        weightAudio = 1.0f,
                        weightVision = 0.0f,
                        diagnosticSummary = "Nota de voz recibida. En espera de descarga para inferencia espectral con MobileNetV3-Small."
                    )
                }
            }
            isVideoRelated -> {
                mediaType = MediaType.VIDEO_MULTIMODAL
                var realVideoFrame = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = true)
                if (realVideoFrame == null && isWhatsApp) {
                    try {
                        Thread.sleep(300)
                        realVideoFrame = findRecentWhatsAppVideoFrame()
                    } catch (e: Exception) {
                    }
                }
                val realVideoAudioUri = if (isWhatsApp) findRecentWhatsAppVideoAudioUri() else null
                val realSpecBitmap = realVideoAudioUri?.let { uri ->
                    try {
                        val router = MediaRouter(applicationContext)
                        router.generateSpectrogramFromAudio(uri)
                    } catch (e: Exception) {
                        null
                    }
                }

                if (realVideoFrame != null || realSpecBitmap != null) {
                    cachedFace = realVideoFrame
                    cachedAudio = realSpecBitmap

                    if (realVideoFrame != null) {
                        isFaceDetected = vClassifier.checkFacePresence(realVideoFrame)
                    }

                    visionProb = if (realVideoFrame != null) {
                        Log.i(TAG, "Ejecutando inferencia TFLite real sobre keyframe de video capturado...")
                        if (isFaceDetected) vClassifier.classifyFaceKeyframe(realVideoFrame, null, requireFaceDetection = true) else 0.0f
                    } else null

                    audioProb = if (realSpecBitmap != null) {
                        Log.i(TAG, "Ejecutando inferencia TFLite real sobre pista de audio de video capturado...")
                        aClassifier.classifySpectrogram(realSpecBitmap, null)
                    } else null

                    val isThreatScore = (visionProb ?: 0f) >= 0.5f || (audioProb ?: 0f) >= 0.5f
                    displayText = if (isThreatScore) {
                        "Video sospechoso con posible alteración audiovisual"
                    } else {
                        "Video entrante analizado (Contenido audiovisual auténtico)"
                    }
                } else {
                    cachedFace = null
                    cachedAudio = null
                    audioProb = null
                    visionProb = null
                    displayText = "Video entrante (Pendiente de descarga)"
                    computedFusionResult = FusionResult(
                        globalRiskPercentage = 0.0f,
                        audioFraudProb = 0.0f,
                        visionFraudProb = 0.0f,
                        riskLevel = RiskLevel.BAJO,
                        weightAudio = 0.55f,
                        weightVision = 0.45f,
                        diagnosticSummary = "Video recibido. En espera de apertura para análisis multimodal y regla de 3/5."
                    )
                }
            }
            else -> return
        }

        val fusionResult = computedFusionResult ?: if (mediaType == MediaType.VIDEO_MULTIMODAL) {
            RiskScorer.calculateVideoRisk(
                audioProb = audioProb,
                frameProbs = listOfNotNull(visionProb),
                isFaceDetected = isFaceDetected
            )
        } else {
            RiskScorer.calculateGlobalRisk(
                audioProb = audioProb,
                visionProb = visionProb,
                customWeightAudio = 0.55f,
                customWeightVision = 0.45f,
                isFaceDetected = isFaceDetected
            )
        }

        val item = InterceptedNotification(
            id = UUID.randomUUID().toString(),
            appName = appName,
            packageName = pkgName,
            sender = sender,
            text = displayText,
            timestamp = System.currentTimeMillis(),
            mediaType = mediaType,
            riskScore = fusionResult.globalRiskPercentage,
            isThreat = fusionResult.globalRiskPercentage >= 70f,
            fusionResult = fusionResult
        )

        NotificationMediaCache.storeMedia(
            context = applicationContext,
            notifId = item.id,
            faceBitmap = cachedFace,
            audioBitmap = cachedAudio
        )

        NotificationRepository.addNotification(item)
        Log.i(TAG, "Notificacion procesada exitosamente: [$appName] $sender: $displayText (Riesgo evaluado: ${fusionResult.globalRiskPercentage}%)")
    }

    private fun extractPictureFromNotification(
        notification: Notification,
        extras: android.os.Bundle,
        isWhatsApp: Boolean,
        isVideo: Boolean = false
    ): Bitmap? {
        try {
            // 1. EXTRA_PICTURE (Bitmap o Icon en BigPictureStyle) - Representa la previsualizacion real de la foto/video enviado
            val pictureObj = extras.get(Notification.EXTRA_PICTURE)
            when (pictureObj) {
                is Bitmap -> {
                    Log.i(TAG, "Imagen real extraida directamente de EXTRA_PICTURE (${pictureObj.width}x${pictureObj.height})")
                    return pictureObj
                }
                is android.graphics.drawable.Icon -> {
                    iconToBitmap(pictureObj)?.let {
                        Log.i(TAG, "Imagen real extraida de EXTRA_PICTURE (Icon -> Bitmap ${it.width}x${it.height})")
                        return it
                    }
                }
            }

            // 2. EXTRA_PICTURE_ICON o android.pictureIcon
            val pictureIconObj = extras.get("android.pictureIcon")
            if (pictureIconObj is android.graphics.drawable.Icon) {
                iconToBitmap(pictureIconObj)?.let {
                    Log.i(TAG, "Imagen real extraida de EXTRA_PICTURE_ICON (${it.width}x${it.height})")
                    return it
                }
            } else if (pictureIconObj is Bitmap) {
                Log.i(TAG, "Imagen real extraida de EXTRA_PICTURE_ICON (${pictureIconObj.width}x${pictureIconObj.height})")
                return pictureIconObj
            }

            // 3. Si es WhatsApp: Obtener el archivo real recibido en el almacenamiento / MediaStore
            // (NUNCA usamos EXTRA_LARGE_ICON porque WhatsApp asigna alli el avatar de perfil del usuario o grupo)
            if (isWhatsApp) {
                if (isVideo) {
                    val videoFrame = findRecentWhatsAppVideoFrame()
                    if (videoFrame != null) {
                        return videoFrame
                    }
                } else {
                    val realImage = findRecentWhatsAppImage()
                    if (realImage != null) {
                        return realImage
                    }
                }
            }

            // 4. MessagingStyle (android.messages) buscando URIs de imagen adjunta
            val messagesBundleArray = extras.getParcelableArray("android.messages")
                ?: extras.getParcelableArray(Notification.EXTRA_MESSAGES)
            if (messagesBundleArray != null) {
                for (i in messagesBundleArray.indices.reversed()) {
                    val p = messagesBundleArray[i]
                    if (p is android.os.Bundle) {
                        val uriObj = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            p.getParcelable("uri", android.net.Uri::class.java)
                                ?: p.getParcelable("data_uri", android.net.Uri::class.java)
                        } else {
                            @Suppress("DEPRECATION")
                            (p.getParcelable("uri") as? android.net.Uri)
                                ?: (p.getParcelable("data_uri") as? android.net.Uri)
                        } ?: (p.getString("uri") ?: p.getString("data_uri"))?.let {
                            try { android.net.Uri.parse(it) } catch (e: Exception) { null }
                        }

                        if (uriObj != null) {
                            loadBitmapFromUri(uriObj)?.let {
                                Log.i(TAG, "Imagen extraida desde URI MessagingStyle: $uriObj (${it.width}x${it.height})")
                                return it
                            }
                        }
                    }
                }
            }

            // 5. EXTRA_LARGE_ICON_BIG solo si es un mapa de bits grande (> 300px), descartando avatares pequeños
            val bigLargeObj = extras.get("android.largeIcon.big") ?: extras.get(Notification.EXTRA_LARGE_ICON_BIG)
            when (bigLargeObj) {
                is Bitmap -> {
                    if (bigLargeObj.width > 300 && bigLargeObj.height > 300) {
                        Log.i(TAG, "Imagen grande extraida de EXTRA_LARGE_ICON_BIG (${bigLargeObj.width}x${bigLargeObj.height})")
                        return bigLargeObj
                    }
                }
                is android.graphics.drawable.Icon -> {
                    iconToBitmap(bigLargeObj)?.let {
                        if (it.width > 300 && it.height > 300) {
                            Log.i(TAG, "Imagen grande extraida de EXTRA_LARGE_ICON_BIG (${it.width}x${it.height})")
                            return it
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error durante extraccion de imagen: ${e.message}")
        }
        return null
    }

    private fun findRecentWhatsAppImage(): Bitmap? {
        // 1. Consulta en MediaStore de Android
        try {
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.Images.Media.DISPLAY_NAME,
                MediaStore.Images.Media.DATE_ADDED
            )
            val selection = "${MediaStore.Images.Media.DISPLAY_NAME} LIKE 'IMG-%-WA%.%'"
            val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idCol = cursor.getColumnIndex(MediaStore.Images.Media._ID)
                    val nameCol = cursor.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME)
                    val dateCol = cursor.getColumnIndex(MediaStore.Images.Media.DATE_ADDED)
                    if (idCol >= 0) {
                        val id = cursor.getLong(idCol)
                        val name = if (nameCol >= 0) cursor.getString(nameCol) else "IMG_WA"
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val ageMs = System.currentTimeMillis() - (dateSec * 1000L)
                        if (ageMs < 2 * 3600 * 1000L || dateSec == 0L) {
                            val contentUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                            contentResolver.openInputStream(contentUri)?.use { stream ->
                                val bmp = BitmapFactory.decodeStream(stream)
                                if (bmp != null) {
                                    Log.i(TAG, "Foto real recuperada de MediaStore: $name (${bmp.width}x${bmp.height})")
                                    return bmp
                                }
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "MediaStore images no accesible aun: ${e.message}")
        }

        // 2. Lectura directa de carpetas compartidas de WhatsApp
        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Images/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Images/"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "WhatsApp")
        )

        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile && f.name.startsWith("IMG-", ignoreCase = true) && f.name.contains("-WA", ignoreCase = true)
                    }
                    val latest = files?.maxByOrNull { it.lastModified() }
                    if (latest != null) {
                        val ageMs = System.currentTimeMillis() - latest.lastModified()
                        if (ageMs < 2 * 3600 * 1000L) {
                            val bmp = BitmapFactory.decodeFile(latest.absolutePath)
                            if (bmp != null) {
                                Log.i(TAG, "Foto real recuperada directamente de archivo: ${latest.name} (${bmp.width}x${bmp.height})")
                                return bmp
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "Directorio WhatsApp ${dir.path} no accesible: ${e.message}")
            }
        }

        return null
    }

    private fun findRecentWhatsAppVideoFrame(): Bitmap? {
        // 1. Consulta en MediaStore de video
        try {
            val projection = arrayOf(
                MediaStore.Video.Media._ID,
                MediaStore.Video.Media.DISPLAY_NAME,
                MediaStore.Video.Media.DATE_ADDED
            )
            val selection = "${MediaStore.Video.Media.DISPLAY_NAME} LIKE 'VID-%-WA%.%'"
            val sortOrder = "${MediaStore.Video.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idCol = cursor.getColumnIndex(MediaStore.Video.Media._ID)
                    val nameCol = cursor.getColumnIndex(MediaStore.Video.Media.DISPLAY_NAME)
                    val dateCol = cursor.getColumnIndex(MediaStore.Video.Media.DATE_ADDED)
                    if (idCol >= 0) {
                        val id = cursor.getLong(idCol)
                        val name = if (nameCol >= 0) cursor.getString(nameCol) else "VID_WA"
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val ageMs = System.currentTimeMillis() - (dateSec * 1000L)
                        if (ageMs < 2 * 3600 * 1000L || dateSec == 0L) {
                            val contentUri = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id)
                            val retriever = MediaMetadataRetriever()
                            try {
                                retriever.setDataSource(applicationContext, contentUri)
                                val frame = retriever.getFrameAtTime(1000000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                                    ?: retriever.getFrameAtTime(0L)
                                if (frame != null) {
                                    Log.i(TAG, "Keyframe de video real obtenido de MediaStore: $name (${frame.width}x${frame.height})")
                                    return frame
                                }
                            } finally {
                                retriever.release()
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "MediaStore video no accesible aun: ${e.message}")
        }

        // 2. Lectura directa de carpetas de video de WhatsApp
        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "WhatsApp")
        )

        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile && f.name.startsWith("VID-", ignoreCase = true) && f.name.contains("-WA", ignoreCase = true)
                    }
                    val latest = files?.maxByOrNull { it.lastModified() }
                    if (latest != null) {
                        val ageMs = System.currentTimeMillis() - latest.lastModified()
                        if (ageMs < 2 * 3600 * 1000L) {
                            val retriever = MediaMetadataRetriever()
                            try {
                                retriever.setDataSource(latest.absolutePath)
                                val frame = retriever.getFrameAtTime(1000000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                                    ?: retriever.getFrameAtTime(0L)
                                if (frame != null) {
                                    Log.i(TAG, "Keyframe de video real leido directamente: ${latest.name} (${frame.width}x${frame.height})")
                                    return frame
                                }
                            } finally {
                                retriever.release()
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "Directorio WhatsApp Video ${dir.path} no accesible: ${e.message}")
            }
        }

        return null
    }

    private fun findRecentWhatsAppAudioUri(): Uri? {
        // 1. Consulta en MediaStore de Audio
        try {
            val projection = arrayOf(
                MediaStore.Audio.Media._ID,
                MediaStore.Audio.Media.DISPLAY_NAME,
                MediaStore.Audio.Media.DATE_ADDED
            )
            val selection = "${MediaStore.Audio.Media.DISPLAY_NAME} LIKE 'PTT-%-WA%.%' OR ${MediaStore.Audio.Media.DISPLAY_NAME} LIKE 'AUD-%-WA%.%'"
            val sortOrder = "${MediaStore.Audio.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idCol = cursor.getColumnIndex(MediaStore.Audio.Media._ID)
                    val dateCol = cursor.getColumnIndex(MediaStore.Audio.Media.DATE_ADDED)
                    if (idCol >= 0) {
                        val id = cursor.getLong(idCol)
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val ageMs = System.currentTimeMillis() - (dateSec * 1000L)
                        if (ageMs < 2 * 3600 * 1000L || dateSec == 0L) {
                            return ContentUris.withAppendedId(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, id)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "MediaStore audio no accesible: ${e.message}")
        }

        // 2. Busqueda en directorios de notas de voz de WhatsApp
        val candidateDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/"),
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Voice Notes/")
        )

        for (baseDir in candidateDirs) {
            try {
                if (baseDir.exists() && baseDir.canRead()) {
                    val allAudioFiles = mutableListOf<File>()
                    baseDir.walkTopDown().maxDepth(3).forEach { f ->
                        if (f.isFile && (f.name.endsWith(".opus", ignoreCase = true) || f.name.endsWith(".m4a", ignoreCase = true) || f.name.endsWith(".mp3", ignoreCase = true))) {
                            allAudioFiles.add(f)
                        }
                    }
                    val latest = allAudioFiles.maxByOrNull { it.lastModified() }
                    if (latest != null && (System.currentTimeMillis() - latest.lastModified()) < 2 * 3600 * 1000L) {
                        Log.i(TAG, "Audio real de WhatsApp localizado: ${latest.name}")
                        return Uri.fromFile(latest)
                    }
                }
            } catch (e: Exception) {
            }
        }
        return null
    }

    private fun findRecentWhatsAppVideoAudioUri(): Uri? {
        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/")
        )
        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile && f.name.startsWith("VID-", ignoreCase = true) && f.name.contains("-WA", ignoreCase = true)
                    }
                    val latest = files?.maxByOrNull { it.lastModified() }
                    if (latest != null && (System.currentTimeMillis() - latest.lastModified()) < 2 * 3600 * 1000L) {
                        return Uri.fromFile(latest)
                    }
                }
            } catch (e: Exception) {
            }
        }
        return null
    }

    private fun iconToBitmap(icon: android.graphics.drawable.Icon): Bitmap? {
        return try {
            val drawable = icon.loadDrawable(applicationContext) ?: return null
            if (drawable is android.graphics.drawable.BitmapDrawable && drawable.bitmap != null) {
                return drawable.bitmap
            }
            val w = if (drawable.intrinsicWidth > 0) drawable.intrinsicWidth else 256
            val h = if (drawable.intrinsicHeight > 0) drawable.intrinsicHeight else 256
            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(bitmap)
            drawable.setBounds(0, 0, canvas.width, canvas.height)
            drawable.draw(canvas)
            bitmap
        } catch (e: Exception) {
            Log.w(TAG, "Error convirtiendo Icon a Bitmap: ${e.message}")
            null
        }
    }

    private fun loadBitmapFromUri(uri: android.net.Uri): Bitmap? {
        return try {
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo decodificar stream desde URI ($uri): ${e.message}")
            null
        }
    }

    private fun loadAssetBitmap(path: String): Bitmap? {
        return try {
            applicationContext.assets.open(path).use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando asset $path: ${e.message}")
            null
        }
    }
}
