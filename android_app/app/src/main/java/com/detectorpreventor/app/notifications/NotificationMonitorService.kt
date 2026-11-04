package com.detectorpreventor.app.notifications

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ComponentName
import android.content.ContentUris
import android.content.Context
import android.content.Intent
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
import androidx.core.app.NotificationCompat
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaRouter
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
import com.detectorpreventor.app.ui.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.util.UUID

class NotificationMonitorService : NotificationListenerService() {

    companion object {
        private const val TAG = "NotificationMonitor"
        private const val CHANNEL_FRAUD_ALERTS = "detector_fraud_alerts"

        const val MIN_IMAGE_BYTES = 4096L
        const val MIN_AUDIO_BYTES = 2048L
        const val MIN_VIDEO_BYTES = 20480L
        const val MAX_MEDIA_AGE_MS = 120_000L
        const val MAX_DOWNLOAD_POLL_ATTEMPTS = 12

        fun isExcludedDownloadOrTempFile(name: String): Boolean {
            val lower = name.lowercase()
            return lower.endsWith(".tmp") ||
                    lower.endsWith(".download") ||
                    lower.endsWith(".crdownload") ||
                    lower.endsWith(".part") ||
                    lower.endsWith(".enc") ||
                    lower.endsWith(".pending") ||
                    lower.endsWith(".nomedia") ||
                    lower.startsWith(".") ||
                    lower.contains(".tmp.") ||
                    lower.contains(".download.") ||
                    lower.contains(".part.")
        }

        fun isStableAndReadableFile(file: File, minBytes: Long): Boolean {
            if (!file.exists() || !file.isFile || !file.canRead()) return false
            val len = file.length()
            if (len < minBytes) return false
            return try {
                FileInputStream(file).use { input ->
                    val buf = ByteArray(minOf(len.toInt(), 512))
                    input.read(buf) > 0
                }
            } catch (e: Exception) {
                false
            }
        }

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
        createAlertNotificationChannel()
        try {
            visionClassifier = VisionClassifier(applicationContext)
            audioClassifier = AudioClassifier(applicationContext)
            Log.i(TAG, "Modelos TFLite inicializados para inferencia en NotificationMonitorService.")
        } catch (e: Exception) {
            Log.e(TAG, "Error inicializando clasificadores TFLite: ${e.message}")
        }
    }

    private fun createAlertNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
                val channel = NotificationChannel(
                    CHANNEL_FRAUD_ALERTS,
                    "Alertas de Estafas Digitales (Edge AI)",
                    NotificationManager.IMPORTANCE_HIGH
                ).apply {
                    description = "Alertas de alta prioridad ante deteccion de audios clonados o deepfakes"
                    enableVibration(true)
                    enableLights(true)
                }
                notificationManager?.createNotificationChannel(channel)
            } catch (e: Exception) {
                Log.w(TAG, "Error creando canal de notificaciones: ${e.message}")
            }
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
                    combinedContent.contains("🎥") ||
                    combinedContent.contains("📹") ||
                    combinedContent.contains("🎬") ||
                    text.equals("Video", ignoreCase = true) ||
                    text.equals("Vídeo", ignoreCase = true) ||
                    text.startsWith("Video (", ignoreCase = true) ||
                    text.startsWith("Vídeo (", ignoreCase = true) ||
                    combinedContent.contains("videonota", ignoreCase = true) ||
                    combinedContent.contains("videollamada", ignoreCase = true) ||
                    combinedContent.contains(".mp4", ignoreCase = true) ||
                    combinedContent.contains(".3gp", ignoreCase = true) ||
                    combinedContent.contains(".mkv", ignoreCase = true) ||
                    combinedContent.contains("VID-", ignoreCase = true)
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
                isWhatsAppAudioHeader ||
                combinedContent.contains("🎤") ||
                combinedContent.contains("🎙") ||
                combinedContent.contains(".opus", ignoreCase = true) ||
                combinedContent.contains(".mp3", ignoreCase = true) ||
                combinedContent.contains(".wav", ignoreCase = true) ||
                combinedContent.contains(".m4a", ignoreCase = true) ||
                combinedContent.contains(".ogg", ignoreCase = true) ||
                combinedContent.contains("PTT-", ignoreCase = true)
        )

        val isPhotoExplicit = !hasUrl && (
            combinedContent.contains("📷") ||
            combinedContent.contains("📸") ||
            combinedContent.contains("🖼") ||
            text.equals("Foto", ignoreCase = true) ||
            text.equals("Imagen", ignoreCase = true) ||
            text.startsWith("Foto (", ignoreCase = true) ||
            text.startsWith("Imagen (", ignoreCase = true) ||
            combinedContent.contains(".jpg", ignoreCase = true) ||
            combinedContent.contains(".jpeg", ignoreCase = true) ||
            combinedContent.contains(".png", ignoreCase = true) ||
            combinedContent.contains("IMG-", ignoreCase = true) ||
            hasPictureExtra
        )

        val isImageRelated = !isVideoRelated && !isAudioRelated && (
                (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.IMAGE_ONLY) ||
                isPhotoExplicit
        )

        if (!isVideoRelated && !isAudioRelated && !isImageRelated) {
            Log.d(TAG, "Notificacion descartada: Mensaje de texto o enlace sin contenido biometrico audiovisual ($text)")
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

        val notifPostTime = if (sbn.postTime > 0L) sbn.postTime else System.currentTimeMillis()

        when {
            isImageRelated -> {
                val mediaType = MediaType.IMAGE_ONLY
                var realPicture = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = false)
                if (realPicture == null && isWhatsApp) {
                    realPicture = findRecentWhatsAppImage(notifPostTime)
                }

                if (realPicture != null) {
                    processCompletedMedia(
                        appName = appName,
                        pkgName = pkgName,
                        sender = sender,
                        mediaType = mediaType,
                        timestamp = notifPostTime,
                        faceBitmap = realPicture,
                        audioBitmap = null,
                        vClassifier = vClassifier,
                        aClassifier = aClassifier
                    )
                } else {
                    val pendingItem = InterceptedNotification(
                        id = UUID.randomUUID().toString(),
                        appName = appName,
                        packageName = pkgName,
                        sender = sender,
                        text = "Fotografía entrante (Descargando en WhatsApp...)",
                        timestamp = notifPostTime,
                        mediaType = mediaType,
                        riskScore = 0.0f,
                        isThreat = false,
                        fusionResult = FusionResult(
                            globalRiskPercentage = 0.0f,
                            audioFraudProb = 0.0f,
                            visionFraudProb = 0.0f,
                            riskLevel = RiskLevel.BAJO,
                            weightAudio = 0.0f,
                            weightVision = 1.0f,
                            diagnosticSummary = "Descargando imagen en WhatsApp. Se verificará con EfficientNet-B0 en cuanto finalice."
                        )
                    )
                    NotificationRepository.addNotification(pendingItem)
                    Log.i(TAG, "Imagen no disponible de inmediato. Iniciando observador de descarga en segundo plano para $sender...")
                    observeAndResolveDownload(pendingItem, mediaType, vClassifier, aClassifier)
                }
            }
            isAudioRelated -> {
                val mediaType = MediaType.AUDIO_ONLY
                val realAudioUri = if (isWhatsApp) findRecentWhatsAppAudioUri(notifPostTime) else null
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
                    processCompletedMedia(
                        appName = appName,
                        pkgName = pkgName,
                        sender = sender,
                        mediaType = mediaType,
                        timestamp = notifPostTime,
                        faceBitmap = null,
                        audioBitmap = realSpecBitmap,
                        vClassifier = vClassifier,
                        aClassifier = aClassifier,
                        mediaUriString = realAudioUri?.toString()
                    )
                } else {
                    val pendingItem = InterceptedNotification(
                        id = UUID.randomUUID().toString(),
                        appName = appName,
                        packageName = pkgName,
                        sender = sender,
                        text = "Nota de voz entrante (Descargando en WhatsApp...)",
                        timestamp = notifPostTime,
                        mediaType = mediaType,
                        riskScore = 0.0f,
                        isThreat = false,
                        fusionResult = FusionResult(
                            globalRiskPercentage = 0.0f,
                            audioFraudProb = 0.0f,
                            visionFraudProb = 0.0f,
                            riskLevel = RiskLevel.BAJO,
                            weightAudio = 1.0f,
                            weightVision = 0.0f,
                            diagnosticSummary = "Descargando nota de voz en WhatsApp. Se verificará con MobileNetV3-Small en cuanto finalice."
                        )
                    )
                    NotificationRepository.addNotification(pendingItem)
                    Log.i(TAG, "Audio no disponible de inmediato. Iniciando observador de descarga en segundo plano para $sender...")
                    observeAndResolveDownload(pendingItem, mediaType, vClassifier, aClassifier)
                }
            }
            isVideoRelated -> {
                val mediaType = MediaType.VIDEO_MULTIMODAL
                var realVideoFrame = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = true)
                if (realVideoFrame == null && isWhatsApp) {
                    realVideoFrame = findRecentWhatsAppVideoFrame(notifPostTime)
                }
                val realVideoAudioUri = if (isWhatsApp) findRecentWhatsAppVideoAudioUri(notifPostTime) else null
                val realSpecBitmap = realVideoAudioUri?.let { uri ->
                    try {
                        val router = MediaRouter(applicationContext)
                        router.generateSpectrogramFromAudio(uri)
                    } catch (e: Exception) {
                        null
                    }
                }

                if (realVideoFrame != null || realSpecBitmap != null) {
                    processCompletedMedia(
                        appName = appName,
                        pkgName = pkgName,
                        sender = sender,
                        mediaType = mediaType,
                        timestamp = notifPostTime,
                        faceBitmap = realVideoFrame,
                        audioBitmap = realSpecBitmap,
                        vClassifier = vClassifier,
                        aClassifier = aClassifier,
                        mediaUriString = realVideoAudioUri?.toString()
                    )
                } else {
                    val pendingItem = InterceptedNotification(
                        id = UUID.randomUUID().toString(),
                        appName = appName,
                        packageName = pkgName,
                        sender = sender,
                        text = "Video entrante (Descargando en WhatsApp...)",
                        timestamp = notifPostTime,
                        mediaType = mediaType,
                        riskScore = 0.0f,
                        isThreat = false,
                        fusionResult = FusionResult(
                            globalRiskPercentage = 0.0f,
                            audioFraudProb = 0.0f,
                            visionFraudProb = 0.0f,
                            riskLevel = RiskLevel.BAJO,
                            weightAudio = 0.55f,
                            weightVision = 0.45f,
                            diagnosticSummary = "Descargando video en WhatsApp. Se analizará multimodalmente en cuanto finalice."
                        )
                    )
                    NotificationRepository.addNotification(pendingItem)
                    Log.i(TAG, "Video no disponible de inmediato. Iniciando observador de descarga en segundo plano para $sender...")
                    observeAndResolveDownload(pendingItem, mediaType, vClassifier, aClassifier)
                }
            }
        }
    }

    private fun processCompletedMedia(
        appName: String,
        pkgName: String,
        sender: String,
        mediaType: MediaType,
        timestamp: Long,
        faceBitmap: Bitmap?,
        audioBitmap: Bitmap?,
        vClassifier: VisionClassifier,
        aClassifier: AudioClassifier,
        mediaUriString: String? = null
    ) {
        var isFaceDetected = true
        val visionProb: Float? = if (faceBitmap != null) {
            isFaceDetected = vClassifier.checkFacePresence(faceBitmap)
            if (isFaceDetected) {
                Log.i(TAG, "Ejecutando inferencia TFLite real sobre imagen facial...")
                vClassifier.classifyFaceKeyframe(faceBitmap, null, requireFaceDetection = true)
            } else {
                Log.i(TAG, "Descarte automatico: Notificacion de imagen sin rostro humano.")
                0.0f
            }
        } else null

        val audioProb: Float? = if (audioBitmap != null) {
            Log.i(TAG, "Ejecutando inferencia TFLite real sobre espectrograma de audio...")
            aClassifier.classifySpectrogram(audioBitmap, null)
        } else null

        val fusionResult = if (mediaType == MediaType.VIDEO_MULTIMODAL) {
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

        val isSenderBlocked = NotificationRepository.isSenderBlocked(sender)
        val isThreatScore = fusionResult.globalRiskPercentage >= 70f || isSenderBlocked
        val baseDisplayText = when (mediaType) {
            MediaType.IMAGE_ONLY -> when {
                !isFaceDetected -> "Fotografía recibida (Descarte automático: Sin rostro humano)"
                isThreatScore -> "Fotografía sospechosa (Alerta: Posible FaceSwap / Deepfake)"
                else -> "Fotografía entrante analizada (Rostro humano auténtico)"
            }
            MediaType.AUDIO_ONLY -> when {
                isThreatScore -> "Nota de voz sospechosa (Alerta: Posible clonación sintética)"
                else -> "Nota de voz entrante analizada (Voz humana auténtica)"
            }
            MediaType.VIDEO_MULTIMODAL -> when {
                isThreatScore -> "Video sospechoso con posible alteración audiovisual"
                else -> "Video entrante analizado (Contenido audiovisual auténtico)"
            }
            else -> "Archivo analizado"
        }
        val displayText = if (isSenderBlocked) "[REMITENTE BLOQUEADO] $baseDisplayText" else baseDisplayText

        val finalFusion = if (isSenderBlocked) {
            fusionResult.copy(
                diagnosticSummary = fusionResult.diagnosticSummary + "\n• Remitente en Lista Negra: Contacto aislado y bloqueado localmente por Edge AI."
            )
        } else {
            fusionResult
        }

        val item = InterceptedNotification(
            id = UUID.randomUUID().toString(),
            appName = appName,
            packageName = pkgName,
            sender = sender,
            text = displayText,
            timestamp = timestamp,
            mediaType = mediaType,
            riskScore = if (isSenderBlocked) maxOf(finalFusion.globalRiskPercentage, 85.0f) else finalFusion.globalRiskPercentage,
            isThreat = isThreatScore,
            fusionResult = finalFusion,
            mediaUriString = mediaUriString
        )

        NotificationMediaCache.storeMedia(
            context = applicationContext,
            notifId = item.id,
            faceBitmap = faceBitmap,
            audioBitmap = audioBitmap,
            mediaUriString = mediaUriString
        )

        NotificationRepository.addNotification(item)
        Log.i(TAG, "Notificacion procesada exitosamente: [$appName] $sender: $displayText (Riesgo evaluado: ${fusionResult.globalRiskPercentage}%)")

        if (isThreatScore) {
            showFraudAlertNotification(sender, displayText, fusionResult.globalRiskPercentage, item.id)
        }
    }

    private fun observeAndResolveDownload(
        pendingItem: InterceptedNotification,
        mediaType: MediaType,
        vClassifier: VisionClassifier,
        aClassifier: AudioClassifier
    ) {
        serviceScope.launch(Dispatchers.IO) {
            val notifId = pendingItem.id
            val notifTime = pendingItem.timestamp
            var resolved = false

            for (attempt in 1..MAX_DOWNLOAD_POLL_ATTEMPTS) {
                delay(if (attempt <= 6) 600L else 1000L)

                try {
                    when (mediaType) {
                        MediaType.IMAGE_ONLY -> {
                            val realPicture = findRecentWhatsAppImage(notifTime)
                            if (realPicture != null) {
                                val isFaceDetected = vClassifier.checkFacePresence(realPicture)
                                val visionProb = if (isFaceDetected) {
                                    vClassifier.classifyFaceKeyframe(realPicture, null, requireFaceDetection = true)
                                } else 0.0f

                                val fusion = RiskScorer.calculateGlobalRisk(
                                    audioProb = null,
                                    visionProb = visionProb,
                                    isFaceDetected = isFaceDetected
                                )
                                val isThreat = fusion.globalRiskPercentage >= 70f
                                val displayText = when {
                                    !isFaceDetected -> "Fotografía recibida (Descarte automático: Sin rostro humano)"
                                    isThreat -> "Fotografía sospechosa (Alerta: Posible FaceSwap / Deepfake)"
                                    else -> "Fotografía entrante analizada (Rostro humano auténtico)"
                                }

                                val updated = pendingItem.copy(
                                    text = displayText,
                                    riskScore = fusion.globalRiskPercentage,
                                    isThreat = isThreat,
                                    fusionResult = fusion
                                )
                                NotificationMediaCache.storeMedia(applicationContext, notifId, faceBitmap = realPicture)
                                NotificationRepository.updateNotification(updated)
                                if (isThreat) {
                                    showFraudAlertNotification(pendingItem.sender, displayText, fusion.globalRiskPercentage, notifId)
                                }
                                Log.i(TAG, "Descarga de imagen resuelta en intento $attempt: $displayText (${fusion.globalRiskPercentage}%)")
                                resolved = true
                                break
                            }
                        }
                        MediaType.AUDIO_ONLY -> {
                            val realAudioUri = findRecentWhatsAppAudioUri(notifTime)
                            if (realAudioUri != null) {
                                val spec = try {
                                    MediaRouter(applicationContext).generateSpectrogramFromAudio(realAudioUri)
                                } catch (e: Exception) { null }

                                if (spec != null) {
                                    val audioProb = aClassifier.classifySpectrogram(spec, null)
                                    val fusion = RiskScorer.calculateGlobalRisk(
                                        audioProb = audioProb,
                                        visionProb = null,
                                        isFaceDetected = true
                                    )
                                    val isThreat = fusion.globalRiskPercentage >= 70f
                                    val displayText = if (isThreat) {
                                        "Nota de voz sospechosa (Alerta: Posible clonación sintética)"
                                    } else {
                                        "Nota de voz entrante analizada (Voz humana auténtica)"
                                    }

                                    val updated = pendingItem.copy(
                                        text = displayText,
                                        riskScore = fusion.globalRiskPercentage,
                                        isThreat = isThreat,
                                        fusionResult = fusion,
                                        mediaUriString = realAudioUri.toString()
                                    )
                                    NotificationMediaCache.storeMedia(
                                        context = applicationContext,
                                        notifId = notifId,
                                        audioBitmap = spec,
                                        mediaUriString = realAudioUri.toString()
                                    )
                                    NotificationRepository.updateNotification(updated)
                                    if (isThreat) {
                                        showFraudAlertNotification(pendingItem.sender, displayText, fusion.globalRiskPercentage, notifId)
                                    }
                                    Log.i(TAG, "Descarga de audio resuelta en intento $attempt: $displayText (${fusion.globalRiskPercentage}%)")
                                    resolved = true
                                    break
                                }
                            }
                        }
                        MediaType.VIDEO_MULTIMODAL -> {
                            val realVideoFrame = findRecentWhatsAppVideoFrame(notifTime)
                            val realAudioUri = findRecentWhatsAppVideoAudioUri(notifTime)
                            val realSpec = realAudioUri?.let {
                                try { MediaRouter(applicationContext).generateSpectrogramFromAudio(it) } catch (e: Exception) { null }
                            }

                            if (realVideoFrame != null || realSpec != null) {
                                val isFaceDetected = realVideoFrame?.let { vClassifier.checkFacePresence(it) } ?: true
                                val visionProb = if (realVideoFrame != null && isFaceDetected) {
                                    vClassifier.classifyFaceKeyframe(realVideoFrame, null, requireFaceDetection = true)
                                } else 0.0f
                                val audioProb = realSpec?.let { aClassifier.classifySpectrogram(it, null) }

                                val fusion = RiskScorer.calculateVideoRisk(
                                    audioProb = audioProb,
                                    frameProbs = listOfNotNull(visionProb),
                                    isFaceDetected = isFaceDetected
                                )
                                val isThreat = fusion.globalRiskPercentage >= 70f
                                val displayText = if (isThreat) {
                                    "Video sospechoso con posible alteración audiovisual"
                                } else {
                                    "Video entrante analizado (Contenido audiovisual auténtico)"
                                }

                                val updated = pendingItem.copy(
                                    text = displayText,
                                    riskScore = fusion.globalRiskPercentage,
                                    isThreat = isThreat,
                                    fusionResult = fusion,
                                    mediaUriString = realAudioUri?.toString()
                                )
                                NotificationMediaCache.storeMedia(
                                    context = applicationContext,
                                    notifId = notifId,
                                    faceBitmap = realVideoFrame,
                                    audioBitmap = realSpec,
                                    mediaUriString = realAudioUri?.toString()
                                )
                                NotificationRepository.updateNotification(updated)
                                if (isThreat) {
                                    showFraudAlertNotification(pendingItem.sender, displayText, fusion.globalRiskPercentage, notifId)
                                }
                                Log.i(TAG, "Descarga de video resuelta en intento $attempt: $displayText (${fusion.globalRiskPercentage}%)")
                                resolved = true
                                break
                            }
                        }
                        else -> break
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Error durante intento $attempt de resolución de descarga: ${e.message}")
                }
            }

            if (!resolved) {
                Log.i(TAG, "Descarga de red WhatsApp no completada tras $MAX_DOWNLOAD_POLL_ATTEMPTS intentos para $notifId. Auto-resolviendo mediante resolveSinglePendingNotification...")
                try {
                    val resolvedItem = NotificationRepository.resolveSinglePendingNotification(applicationContext, pendingItem)
                    NotificationRepository.updateNotification(resolvedItem)
                    if (resolvedItem.isThreat) {
                        showFraudAlertNotification(pendingItem.sender, resolvedItem.text, resolvedItem.riskScore, notifId)
                    }
                    Log.i(TAG, "Descarga auto-resuelta con éxito: ${resolvedItem.text} (${resolvedItem.riskScore}%)")
                } catch (e: Exception) {
                    Log.e(TAG, "Error en auto-resolución de descarga: ${e.message}")
                }
            }
        }
    }

    private fun showFraudAlertNotification(sender: String, displayText: String, riskScore: Float, notifId: String) {
        try {
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager ?: return

            val tapIntent = Intent(applicationContext, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra("screen", "notifications")
                putExtra("notification_id", notifId)
            }
            val pendingIntent = PendingIntent.getActivity(
                applicationContext,
                notifId.hashCode(),
                tapIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val alertNotif = NotificationCompat.Builder(applicationContext, CHANNEL_FRAUD_ALERTS)
                .setSmallIcon(android.R.drawable.stat_notify_error)
                .setContentTitle("ALERTA DE ESTAFA DETECTADA (${riskScore.toInt()}%)")
                .setContentText("$sender: $displayText")
                .setStyle(NotificationCompat.BigTextStyle().bigText(
                    "Remitente: $sender\n$displayText\nRiesgo Estimado: ${riskScore.toInt()}%\nNo transfiera dinero ni comparta información confidencial."
                ))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setDefaults(Notification.DEFAULT_ALL)
                .setAutoCancel(true)
                .setColor(0xFFDC2626.toInt())
                .setContentIntent(pendingIntent)
                .build()

            notificationManager.notify(notifId.hashCode(), alertNotif)
            Log.i(TAG, "Notificación de alerta del sistema disparada para $sender (Riesgo: ${riskScore.toInt()}%)")
        } catch (e: Exception) {
            Log.e(TAG, "Error mostrando notificacion de alerta: ${e.message}")
        }
    }

    private fun extractPictureFromNotification(
        notification: Notification,
        extras: android.os.Bundle,
        isWhatsApp: Boolean,
        isVideo: Boolean = false
    ): Bitmap? {
        try {
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
            Log.w(TAG, "Error durante extraccion de imagen de extras: ${e.message}")
        }
        return null
    }

    fun findRecentWhatsAppImage(notifTimestamp: Long = System.currentTimeMillis()): Bitmap? {
        val minTimestampMs = notifTimestamp - 60_000L
        val maxAgeMs = 120_000L

        try {
            val projection = mutableListOf(
                MediaStore.Images.Media._ID,
                MediaStore.Images.Media.DISPLAY_NAME,
                MediaStore.Images.Media.DATE_ADDED,
                MediaStore.Images.Media.SIZE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                projection.add(MediaStore.MediaColumns.IS_PENDING)
            }

            val selection = "${MediaStore.Images.Media.DISPLAY_NAME} LIKE 'IMG-%-WA%.%' AND ${MediaStore.Images.Media.SIZE} >= $MIN_IMAGE_BYTES"
            val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection.toTypedArray(),
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                while (cursor.moveToNext()) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        val pendingCol = cursor.getColumnIndex(MediaStore.MediaColumns.IS_PENDING)
                        if (pendingCol >= 0 && cursor.getInt(pendingCol) != 0) {
                            continue
                        }
                    }

                    val idCol = cursor.getColumnIndex(MediaStore.Images.Media._ID)
                    val nameCol = cursor.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME)
                    val dateCol = cursor.getColumnIndex(MediaStore.Images.Media.DATE_ADDED)

                    if (idCol >= 0) {
                        val name = if (nameCol >= 0) cursor.getString(nameCol) ?: "" else ""
                        if (isExcludedDownloadOrTempFile(name)) continue
                        val lower = name.lowercase()
                        if (!lower.endsWith(".jpg") && !lower.endsWith(".jpeg") && !lower.endsWith(".png")) continue

                        val id = cursor.getLong(idCol)
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val dateMs = dateSec * 1000L
                        val ageMs = System.currentTimeMillis() - dateMs

                        if ((dateMs >= minTimestampMs && ageMs <= maxAgeMs) || (notifTimestamp <= 0L && ageMs <= 60000L)) {
                            val contentUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                            val uriStr = contentUri.toString()
                            if (NotificationMediaCache.isFileBound(uriStr) || NotificationMediaCache.isFileBound(name)) continue

                            try {
                                contentResolver.openInputStream(contentUri)?.use { stream ->
                                    val bmp = BitmapFactory.decodeStream(stream)
                                    if (bmp != null && bmp.width > 20 && bmp.height > 20) {
                                        NotificationMediaCache.markFileBound(uriStr)
                                        NotificationMediaCache.markFileBound(name)
                                        Log.i(TAG, "Foto real recuperada de MediaStore: $name (${bmp.width}x${bmp.height})")
                                        return bmp
                                    }
                                }
                            } catch (e: Exception) {
                                Log.d(TAG, "Error decodificando imagen de MediaStore: ${e.message}")
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "MediaStore images no accesible aun: ${e.message}")
        }

        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/"),
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/Private/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Images/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Images/Private/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/Private/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Images/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Images/Private/"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "WhatsApp"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "WhatsApp/Private"),
            File("/storage/emulated/0/Pictures/WhatsApp/"),
            File("/storage/emulated/0/Pictures/WhatsApp/Private/")
        )

        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile &&
                        f.name.startsWith("IMG-", ignoreCase = true) &&
                        f.name.contains("-WA", ignoreCase = true) &&
                        !isExcludedDownloadOrTempFile(f.name) &&
                        (f.name.endsWith(".jpg", ignoreCase = true) || f.name.endsWith(".jpeg", ignoreCase = true) || f.name.endsWith(".png", ignoreCase = true)) &&
                        f.length() >= MIN_IMAGE_BYTES
                    }

                    val candidate = files?.filter { f ->
                        val mod = f.lastModified()
                        mod >= minTimestampMs && (System.currentTimeMillis() - mod) <= maxAgeMs && !NotificationMediaCache.isFileBound(f.absolutePath)
                    }?.maxByOrNull { it.lastModified() }

                    if (candidate != null && isStableAndReadableFile(candidate, MIN_IMAGE_BYTES)) {
                        val bmp = BitmapFactory.decodeFile(candidate.absolutePath)
                        if (bmp != null && bmp.width > 20 && bmp.height > 20) {
                            NotificationMediaCache.markFileBound(candidate.absolutePath)
                            Log.i(TAG, "Foto real recuperada directamente de archivo: ${candidate.name} (${bmp.width}x${bmp.height})")
                            return bmp
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "Directorio WhatsApp ${dir.path} no accesible: ${e.message}")
            }
        }

        return null
    }

    fun findRecentWhatsAppVideoFrame(notifTimestamp: Long = System.currentTimeMillis()): Bitmap? {
        val minTimestampMs = notifTimestamp - 60_000L
        val maxAgeMs = 120_000L

        try {
            val projection = mutableListOf(
                MediaStore.Video.Media._ID,
                MediaStore.Video.Media.DISPLAY_NAME,
                MediaStore.Video.Media.DATE_ADDED,
                MediaStore.Video.Media.SIZE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                projection.add(MediaStore.MediaColumns.IS_PENDING)
            }

            val selection = "${MediaStore.Video.Media.DISPLAY_NAME} LIKE 'VID-%-WA%.%' AND ${MediaStore.Video.Media.SIZE} >= $MIN_VIDEO_BYTES"
            val sortOrder = "${MediaStore.Video.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                projection.toTypedArray(),
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                while (cursor.moveToNext()) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        val pendingCol = cursor.getColumnIndex(MediaStore.MediaColumns.IS_PENDING)
                        if (pendingCol >= 0 && cursor.getInt(pendingCol) != 0) {
                            continue
                        }
                    }

                    val idCol = cursor.getColumnIndex(MediaStore.Video.Media._ID)
                    val nameCol = cursor.getColumnIndex(MediaStore.Video.Media.DISPLAY_NAME)
                    val dateCol = cursor.getColumnIndex(MediaStore.Video.Media.DATE_ADDED)

                    if (idCol >= 0) {
                        val name = if (nameCol >= 0) cursor.getString(nameCol) ?: "" else ""
                        if (isExcludedDownloadOrTempFile(name)) continue
                        val lower = name.lowercase()
                        if (!lower.endsWith(".mp4") && !lower.endsWith(".3gp") && !lower.endsWith(".mkv")) continue

                        val id = cursor.getLong(idCol)
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val dateMs = dateSec * 1000L
                        val ageMs = System.currentTimeMillis() - dateMs

                        if ((dateMs >= minTimestampMs && ageMs <= maxAgeMs) || (notifTimestamp <= 0L && ageMs <= 60000L)) {
                            val contentUri = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id)
                            val uriStr = contentUri.toString()
                            if (NotificationMediaCache.isFileBound(uriStr) || NotificationMediaCache.isFileBound(name)) continue

                            val retriever = MediaMetadataRetriever()
                            try {
                                retriever.setDataSource(applicationContext, contentUri)
                                val frame = retriever.getFrameAtTime(1000000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                                    ?: retriever.getFrameAtTime(0L)
                                if (frame != null) {
                                    NotificationMediaCache.markFileBound(uriStr)
                                    NotificationMediaCache.markFileBound(name)
                                    Log.i(TAG, "Keyframe de video real obtenido de MediaStore: $name (${frame.width}x${frame.height})")
                                    return frame
                                }
                            } catch (e: Exception) {
                                Log.d(TAG, "Error extrayendo frame de video de MediaStore: ${e.message}")
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

        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/Private/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Video/"),
            File("/storage/emulated/0/WhatsApp/Media/WhatsApp Video/Private/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/Private/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Video/Private/"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "WhatsApp"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "WhatsApp/Private"),
            File("/storage/emulated/0/Movies/WhatsApp/"),
            File("/storage/emulated/0/Movies/WhatsApp/Private/")
        )

        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile &&
                        f.name.startsWith("VID-", ignoreCase = true) &&
                        f.name.contains("-WA", ignoreCase = true) &&
                        !isExcludedDownloadOrTempFile(f.name) &&
                        (f.name.endsWith(".mp4", ignoreCase = true) || f.name.endsWith(".3gp", ignoreCase = true) || f.name.endsWith(".mkv", ignoreCase = true)) &&
                        f.length() >= MIN_VIDEO_BYTES
                    }

                    val candidate = files?.filter { f ->
                        val mod = f.lastModified()
                        mod >= minTimestampMs && (System.currentTimeMillis() - mod) <= maxAgeMs && !NotificationMediaCache.isFileBound(f.absolutePath)
                    }?.maxByOrNull { it.lastModified() }

                    if (candidate != null && isStableAndReadableFile(candidate, MIN_VIDEO_BYTES)) {
                        val retriever = MediaMetadataRetriever()
                        try {
                            retriever.setDataSource(candidate.absolutePath)
                            val frame = retriever.getFrameAtTime(1000000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                                ?: retriever.getFrameAtTime(0L)
                            if (frame != null) {
                                NotificationMediaCache.markFileBound(candidate.absolutePath)
                                Log.i(TAG, "Keyframe de video real leido directamente: ${candidate.name} (${frame.width}x${frame.height})")
                                return frame
                            }
                        } catch (e: Exception) {
                            Log.d(TAG, "Error extrayendo frame de video de archivo: ${e.message}")
                        } finally {
                            retriever.release()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "Directorio WhatsApp Video ${dir.path} no accesible: ${e.message}")
            }
        }

        return null
    }

    fun findRecentWhatsAppAudioUri(notifTimestamp: Long = System.currentTimeMillis()): Uri? {
        val minTimestampMs = notifTimestamp - 60_000L
        val maxAgeMs = 120_000L

        try {
            val projection = mutableListOf(
                MediaStore.Audio.Media._ID,
                MediaStore.Audio.Media.DISPLAY_NAME,
                MediaStore.Audio.Media.DATE_ADDED,
                MediaStore.Audio.Media.SIZE
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                projection.add(MediaStore.MediaColumns.IS_PENDING)
            }
            val selection = "(${MediaStore.Audio.Media.DISPLAY_NAME} LIKE 'PTT-%-WA%.%' OR ${MediaStore.Audio.Media.DISPLAY_NAME} LIKE 'AUD-%-WA%.%') AND ${MediaStore.Audio.Media.SIZE} >= $MIN_AUDIO_BYTES"
            val sortOrder = "${MediaStore.Audio.Media.DATE_ADDED} DESC"
            contentResolver.query(
                MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                projection.toTypedArray(),
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                while (cursor.moveToNext()) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        val pendingCol = cursor.getColumnIndex(MediaStore.MediaColumns.IS_PENDING)
                        if (pendingCol >= 0 && cursor.getInt(pendingCol) != 0) {
                            continue
                        }
                    }
                    val idCol = cursor.getColumnIndex(MediaStore.Audio.Media._ID)
                    val nameCol = cursor.getColumnIndex(MediaStore.Audio.Media.DISPLAY_NAME)
                    val dateCol = cursor.getColumnIndex(MediaStore.Audio.Media.DATE_ADDED)
                    if (idCol >= 0) {
                        val name = if (nameCol >= 0) cursor.getString(nameCol) ?: "" else ""
                        if (isExcludedDownloadOrTempFile(name)) continue
                        val lower = name.lowercase()
                        if (!lower.endsWith(".opus") && !lower.endsWith(".m4a") && !lower.endsWith(".mp3") && !lower.endsWith(".ogg") && !lower.endsWith(".wav")) continue

                        val id = cursor.getLong(idCol)
                        val dateSec = if (dateCol >= 0) cursor.getLong(dateCol) else 0L
                        val dateMs = dateSec * 1000L
                        val ageMs = System.currentTimeMillis() - dateMs
                        if ((dateMs >= minTimestampMs && ageMs <= maxAgeMs) || (notifTimestamp <= 0L && ageMs <= 60000L)) {
                            val uri = ContentUris.withAppendedId(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, id)
                            val uriStr = uri.toString()
                            if (NotificationMediaCache.isFileBound(uriStr) || NotificationMediaCache.isFileBound(name)) continue
                            NotificationMediaCache.markFileBound(uriStr)
                            NotificationMediaCache.markFileBound(name)
                            Log.i(TAG, "Audio real de WhatsApp localizado en MediaStore: $name")
                            return uri
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "MediaStore audio no accesible: ${e.message}")
        }

        val candidateDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/"),
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/"),
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/Private/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/"),
            File(Environment.getExternalStorageDirectory(), "WhatsApp/Media/WhatsApp Voice Notes/")
        )

        for (baseDir in candidateDirs) {
            try {
                if (baseDir.exists() && baseDir.canRead()) {
                    val allAudioFiles = mutableListOf<File>()
                    baseDir.walkTopDown().maxDepth(3).forEach { f ->
                        if (f.isFile &&
                            !isExcludedDownloadOrTempFile(f.name) &&
                            !NotificationMediaCache.isFileBound(f.absolutePath) &&
                            (f.name.endsWith(".opus", ignoreCase = true) || f.name.endsWith(".m4a", ignoreCase = true) || f.name.endsWith(".mp3", ignoreCase = true) || f.name.endsWith(".ogg", ignoreCase = true)) &&
                            f.length() >= MIN_AUDIO_BYTES
                        ) {
                            val mod = f.lastModified()
                            if (mod >= minTimestampMs && (System.currentTimeMillis() - mod) <= maxAgeMs) {
                                allAudioFiles.add(f)
                            }
                        }
                    }
                    val latest = allAudioFiles.maxByOrNull { it.lastModified() }
                    if (latest != null && isStableAndReadableFile(latest, MIN_AUDIO_BYTES)) {
                        NotificationMediaCache.markFileBound(latest.absolutePath)
                        Log.i(TAG, "Audio real de WhatsApp localizado en archivo: ${latest.name} (${latest.length()} bytes)")
                        return Uri.fromFile(latest)
                    }
                }
            } catch (e: Exception) {
            }
        }
        return null
    }

    fun findRecentWhatsAppVideoAudioUri(notifTimestamp: Long = System.currentTimeMillis()): Uri? {
        val minTimestampMs = notifTimestamp - 20_000L
        val maxAgeMs = MAX_MEDIA_AGE_MS

        val waDirs = listOf(
            File("/storage/emulated/0/Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/"),
            File(Environment.getExternalStorageDirectory(), "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/")
        )
        for (dir in waDirs) {
            try {
                if (dir.exists() && dir.canRead()) {
                    val files = dir.listFiles { f ->
                        f.isFile &&
                        f.name.startsWith("VID-", ignoreCase = true) &&
                        f.name.contains("-WA", ignoreCase = true) &&
                        !isExcludedDownloadOrTempFile(f.name) &&
                        (f.name.endsWith(".mp4", ignoreCase = true) || f.name.endsWith(".3gp", ignoreCase = true) || f.name.endsWith(".mkv", ignoreCase = true)) &&
                        f.length() >= MIN_VIDEO_BYTES
                    }

                    val candidate = files?.filter { f ->
                        val mod = f.lastModified()
                        mod >= minTimestampMs && (System.currentTimeMillis() - mod) <= maxAgeMs && !NotificationMediaCache.isFileBound(f.absolutePath)
                    }?.maxByOrNull { it.lastModified() }

                    if (candidate != null && isStableAndReadableFile(candidate, MIN_VIDEO_BYTES)) {
                        NotificationMediaCache.markFileBound(candidate.absolutePath)
                        return Uri.fromFile(candidate)
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
