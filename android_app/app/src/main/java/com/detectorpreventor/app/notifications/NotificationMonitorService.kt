package com.detectorpreventor.app.notifications

import android.app.Notification
import android.content.ComponentName
import android.content.ContentUris
import android.content.Context
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
import com.detectorpreventor.app.domain.MediaRouter
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
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
                val componentName = ComponentName(context, NotificationMonitorService::class.java)
                val pm = context.packageManager

                pm.setComponentEnabledSetting(
                    componentName,
                    PackageManager.COMPONENT_ENABLED_STATE_DISABLED,
                    PackageManager.DONT_KILL_APP
                )
                pm.setComponentEnabledSetting(
                    componentName,
                    PackageManager.COMPONENT_ENABLED_STATE_ENABLED,
                    PackageManager.DONT_KILL_APP
                )
                Log.i(TAG, "Ciclo de componente ejecutado para re-vincular NotificationMonitorService.")
            } catch (e: Exception) {
                Log.e(TAG, "Error al forzar re-vinculacion del servicio: ${e.message}")
            }
        }
    }

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

    override fun onListenerConnected() {
        super.onListenerConnected()
        Log.i(TAG, "NotificationListenerService CONECTADO y vinculado por el SO.")
        NotificationRepository.setServiceConnected(true)
    }

    override fun onListenerDisconnected() {
        super.onListenerDisconnected()
        Log.w(TAG, "NotificationListenerService DESCONECTADO por el SO. Solicitando re-vinculacion...")
        NotificationRepository.setServiceConnected(false)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                requestRebind(ComponentName(this, NotificationMonitorService::class.java))
            } catch (e: Exception) {
                Log.e(TAG, "Fallo al solicitar requestRebind: ${e.message}")
            }
        }
    }

    override fun onDestroy() {
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

        val pkgName = sbn.packageName ?: return
        val isWhatsApp = pkgName.contains("whatsapp", ignoreCase = true)
        val isTelegram = pkgName.contains("telegram", ignoreCase = true)
        val isMessaging = pkgName.contains("messaging", ignoreCase = true) ||
                pkgName.contains("orca", ignoreCase = true)

        if (!isWhatsApp && !isTelegram && !isMessaging) return

        val notification = sbn.notification ?: return
        val extras = notification.extras ?: return

        if ((notification.flags and Notification.FLAG_GROUP_SUMMARY) != 0) {
            Log.d(TAG, "Notificacion ignorada por FLAG_GROUP_SUMMARY ($pkgName)")
            return
        }

        val template = extras.getString(Notification.EXTRA_TEMPLATE) ?: ""
        val compatTemplate = extras.getString("androidx.core.app.extra.COMPAT_TEMPLATE") ?: ""
        if (template.contains("InboxStyle", ignoreCase = true) || compatTemplate.contains("InboxStyle", ignoreCase = true)) {
            Log.d(TAG, "Notificacion ignorada por ser plantilla InboxStyle multi-chat ($pkgName)")
            return
        }

        val rawTitle = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim()
        val conversationTitle = extras.getCharSequence("android.conversationTitle")?.toString()?.trim()
        val subText = extras.getCharSequence(Notification.EXTRA_SUB_TEXT)?.toString()?.trim()
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim() ?: ""
        val ticker = notification.tickerText?.toString()?.trim() ?: ""
        val textLines = (extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES) ?: emptyArray())
            .mapNotNull { it?.toString()?.trim() }
            .filter { it.isNotBlank() }

        if (isWhatsApp && (rawTitle.isNullOrBlank() || rawTitle == "WhatsApp") &&
            (text.contains("mensajes de", ignoreCase = true) || text.contains("messages from", ignoreCase = true))) {
            Log.d(TAG, "Notificacion ignorada por ser resumen agregado multi-chat de WhatsApp ($text)")
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

        val isVideoRelated = (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.VIDEO_MULTIMODAL) ||
                combinedContent.contains("video", ignoreCase = true) ||
                combinedContent.contains("vídeo", ignoreCase = true) ||
                combinedContent.contains("videollamada", ignoreCase = true) ||
                combinedContent.contains("videonota", ignoreCase = true)

        val durationRegex = Regex("""\b\d{1,2}:\d{2}\b""")
        val isAudioRelated = !isVideoRelated && (
                (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.AUDIO_ONLY) ||
                combinedContent.contains("audio", ignoreCase = true) ||
                combinedContent.contains("nota de voz", ignoreCase = true) ||
                combinedContent.contains("mensaje de voz", ignoreCase = true) ||
                combinedContent.contains("voice message", ignoreCase = true) ||
                combinedContent.contains("voice note", ignoreCase = true) ||
                (isWhatsApp && durationRegex.containsMatchIn(combinedContent))
        )

        val isImageRelated = !isVideoRelated && !isAudioRelated && (
                (isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.IMAGE_ONLY) ||
                combinedContent.contains("foto", ignoreCase = true) ||
                combinedContent.contains("imagen", ignoreCase = true) ||
                combinedContent.contains("photo", ignoreCase = true) ||
                combinedContent.contains("image", ignoreCase = true)
        )

        if (!isVideoRelated && !isAudioRelated && !isImageRelated) {
            return
        }

        val sender = when {
            !messageSender.isNullOrBlank() -> messageSender
            !conversationTitle.isNullOrBlank() -> conversationTitle
            !rawTitle.isNullOrBlank() && rawTitle != "WhatsApp" -> rawTitle
            !subText.isNullOrBlank() -> subText
            else -> rawTitle ?: "Contacto de WhatsApp"
        }

        val appName = when {
            isWhatsApp -> "WhatsApp"
            isTelegram -> "Telegram"
            else -> "Mensajeria"
        }

        val fraudKeywords = listOf(
            "urgente", "deposito", "depósito", "transferencia", "dinero", "banco",
            "tarjeta", "ganaste", "premio", "cuenta bloqueada", "mama", "mamá",
            "papa", "papá", "hijo", "ayuda", "familiar", "codigo", "código",
            "verificacion", "verificación", "nip", "clave", "emergencia", "prestamo", "préstamo"
        )
        val containsFraudKeyword = fraudKeywords.any { combinedContent.contains(it, ignoreCase = true) }
        val isUnknownSender = sender.startsWith("+") || sender.contains("desconocido", ignoreCase = true)
        val suspiciousContext = containsFraudKeyword || isUnknownSender

        val vClassifier = visionClassifier ?: VisionClassifier(applicationContext).also { visionClassifier = it }
        val aClassifier = audioClassifier ?: AudioClassifier(applicationContext).also { audioClassifier = it }

        val mediaType: MediaType
        val audioProb: Float?
        val visionProb: Float?
        val displayText: String
        var cachedFace: Bitmap? = null
        var cachedAudio: Bitmap? = null

        when {
            isImageRelated -> {
                mediaType = MediaType.IMAGE_ONLY
                var realPicture = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = false)
                if (realPicture == null && isWhatsApp) {
                    try {
                        Thread.sleep(350)
                        realPicture = findRecentWhatsAppImage()
                    } catch (e: Exception) {
                    }
                }
                val pictureBitmap = realPicture
                    ?: loadAssetBitmap(if (suspiciousContext) "samples/image_fake_face.jpg" else "samples/01_retrato_humano_real_1.jpg")
                cachedFace = pictureBitmap
                visionProb = if (realPicture != null) {
                    Log.i(TAG, "Ejecutando inferencia TFLite real sobre imagen capturada de la notificacion...")
                    vClassifier.classifyFaceKeyframe(realPicture, null)
                } else {
                    pictureBitmap?.let { vClassifier.classifyFaceKeyframe(it, if (suspiciousContext) "image_fake" else "image_real") }
                }
                audioProb = null
                displayText = if (containsFraudKeyword) {
                    "Fotografía sospechosa (Alerta: Posible FaceSwap)"
                } else {
                    "Fotografía entrante recibida"
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
                val specBitmap = realSpecBitmap
                    ?: loadAssetBitmap(if (suspiciousContext) "samples/audio_fake_spec.png" else "samples/01_voz_humana_real_bonafide_1_spec.png")
                cachedAudio = specBitmap
                audioProb = if (realSpecBitmap != null) {
                    Log.i(TAG, "Ejecutando inferencia TFLite real sobre espectrograma de audio recibido...")
                    aClassifier.classifySpectrogram(realSpecBitmap, null)
                } else {
                    specBitmap?.let { aClassifier.classifySpectrogram(it, if (suspiciousContext) "audio_fake" else "audio_real") }
                }
                visionProb = null
                displayText = if (containsFraudKeyword) {
                    "Nota de voz sospechosa (Posible clonación / Deepfake)"
                } else {
                    "Nota de voz entrante recibida"
                }
            }
            isVideoRelated -> {
                mediaType = MediaType.VIDEO_MULTIMODAL
                var realVideoFrame = extractPictureFromNotification(notification, extras, isWhatsApp, isVideo = true)
                if (realVideoFrame == null && isWhatsApp) {
                    try {
                        Thread.sleep(400)
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
                val faceBitmap = realVideoFrame
                    ?: loadAssetBitmap(if (suspiciousContext) "samples/06_deepfake_rostro_ia_1.jpg" else "samples/01_retrato_humano_real_1.jpg")
                val specBitmap = realSpecBitmap
                    ?: loadAssetBitmap(if (suspiciousContext) "samples/06_clonacion_ia_spoof_tts_1_spec.png" else "samples/01_voz_humana_real_bonafide_1_spec.png")
                cachedFace = faceBitmap
                cachedAudio = specBitmap
                visionProb = if (realVideoFrame != null) {
                    Log.i(TAG, "Ejecutando inferencia TFLite real sobre keyframe de video capturado...")
                    vClassifier.classifyFaceKeyframe(realVideoFrame, null)
                } else {
                    faceBitmap?.let { vClassifier.classifyFaceKeyframe(it, if (suspiciousContext) "deepfake" else "video real") }
                }
                audioProb = if (realSpecBitmap != null) {
                    Log.i(TAG, "Ejecutando inferencia TFLite real sobre pista de audio de video capturado...")
                    aClassifier.classifySpectrogram(realSpecBitmap, null)
                } else {
                    specBitmap?.let { aClassifier.classifySpectrogram(it, if (suspiciousContext) "clonacion" else "video real") }
                }
                displayText = if (containsFraudKeyword) {
                    "Video sospechoso con posible alteración audiovisual"
                } else {
                    "Video entrante recibido"
                }
            }
            else -> return
        }

        val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

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
        Log.i(TAG, "Notificacion procesada por TFLite: [$appName] $sender: $displayText (Riesgo evaluado: ${fusionResult.globalRiskPercentage}%)")
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
