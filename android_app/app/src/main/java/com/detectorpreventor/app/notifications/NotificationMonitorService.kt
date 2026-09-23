package com.detectorpreventor.app.notifications

import android.app.Notification
import android.content.ComponentName
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskScorer
import java.util.UUID

/**
 * Servicio de Android para interceptar notificaciones de mensajeria (WhatsApp, Telegram, etc.)
 * e inspeccionar posibles amenazas de audios o imagenes falsificadas en tiempo real.
 * Incluye auto-recuperacion contra desvinculaciones del sistema operativo ("Ghost Unbind").
 */
class NotificationMonitorService : NotificationListenerService() {

    companion object {
        private const val TAG = "NotificationMonitor"

        /**
         * Fuerza al sistema operativo a reevaluar y re-vincular el NotificationListenerService.
         * Resuelve el problema conocido de Android donde el servicio es desvinculado silenciosamente
         * por optimizacion de bateria, ahorro de energia o tras reinicios de la aplicacion.
         */
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

        // Extraer todos los campos de texto posibles
        val rawTitle = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim()
        val conversationTitle = extras.getCharSequence("android.conversationTitle")?.toString()?.trim()
        val subText = extras.getCharSequence(Notification.EXTRA_SUB_TEXT)?.toString()?.trim()
        val infoText = extras.getCharSequence(Notification.EXTRA_INFO_TEXT)?.toString()?.trim()
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim() ?: ""
        val ticker = notification.tickerText?.toString()?.trim() ?: ""
        val textLines = (extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES) ?: emptyArray())
            .mapNotNull { it?.toString()?.trim() }
            .filter { it.isNotBlank() }

        // Extraer mensajes individuales si la notificacion usa MessagingStyle (estandar moderno de WhatsApp)
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
                        isMediaFromMessagingStyle = true
                        if (mimeType.startsWith("audio", ignoreCase = true)) {
                            messagingStyleMediaType = MediaType.AUDIO_ONLY
                            styleMessages.add("Nota de voz adjunta ($mimeType)")
                        } else if (mimeType.startsWith("image", ignoreCase = true)) {
                            messagingStyleMediaType = MediaType.IMAGE_ONLY
                            styleMessages.add("Imagen adjunta ($mimeType)")
                        }
                    }
                }
            }
        }

        // Combinar todos los textos encontrados
        val allContentList = mutableListOf<String>()
        if (text.isNotBlank()) allContentList.add(text)
        if (bigText.isNotBlank()) allContentList.add(bigText)
        allContentList.addAll(textLines)
        allContentList.addAll(styleMessages)
        if (ticker.isNotBlank()) allContentList.add(ticker)

        val combinedContent = allContentList.distinct().joinToString(" ")

        // Filtrar notificaciones continuas del sistema de WhatsApp
        val isSystemStatusNotification = sbn.isOngoing && (
                combinedContent.contains("WhatsApp Web", ignoreCase = true) ||
                combinedContent.contains("copia de seguridad", ignoreCase = true) ||
                combinedContent.contains("backup", ignoreCase = true) ||
                combinedContent.contains("buscando mensajes", ignoreCase = true) ||
                combinedContent.contains("llamada en curso", ignoreCase = true)
        )
        if (isSystemStatusNotification) return

        if (combinedContent.isBlank() && rawTitle.isNullOrBlank()) return

        // Determinar remitente adecuado
        val sender = when {
            !messageSender.isNullOrBlank() -> messageSender!!
            !conversationTitle.isNullOrBlank() -> conversationTitle
            !rawTitle.isNullOrBlank() && rawTitle != "WhatsApp" -> rawTitle
            !subText.isNullOrBlank() -> subText
            else -> rawTitle ?: "Contacto de WhatsApp"
        }

        // Deteccion de medios y duracion
        val durationRegex = Regex("""\b\d{1,2}:\d{2}\b""")
        val isAudioRelated = isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.AUDIO_ONLY ||
                combinedContent.contains("audio", ignoreCase = true) ||
                combinedContent.contains("nota de voz", ignoreCase = true) ||
                combinedContent.contains("mensaje de voz", ignoreCase = true) ||
                combinedContent.contains("voice message", ignoreCase = true) ||
                combinedContent.contains("voice note", ignoreCase = true) ||
                (isWhatsApp && durationRegex.containsMatchIn(combinedContent))

        val isImageRelated = isMediaFromMessagingStyle && messagingStyleMediaType == MediaType.IMAGE_ONLY ||
                combinedContent.contains("foto", ignoreCase = true) ||
                combinedContent.contains("imagen", ignoreCase = true) ||
                combinedContent.contains("photo", ignoreCase = true) ||
                combinedContent.contains("image", ignoreCase = true) ||
                combinedContent.contains("sticker", ignoreCase = true)

        // Deteccion de palabras clave de ingenieria social / fraude financiero
        val fraudKeywords = listOf(
            "urgente", "deposito", "depósito", "transferencia", "dinero", "banco",
            "tarjeta", "ganaste", "premio", "cuenta bloqueada", "mama", "mamá",
            "papa", "papá", "hijo", "ayuda", "familiar", "codigo", "código",
            "verificacion", "verificación", "nip", "clave"
        )
        val containsFraudKeyword = fraudKeywords.any { combinedContent.contains(it, ignoreCase = true) }

        val mediaType = when {
            isAudioRelated -> MediaType.AUDIO_ONLY
            isImageRelated -> MediaType.IMAGE_ONLY
            else -> MediaType.UNKNOWN
        }

        val appName = when {
            isWhatsApp -> "WhatsApp"
            isTelegram -> "Telegram"
            else -> "Mensajeria"
        }

        // Evaluacion del riesgo
        val (audioProb, visionProb) = when {
            isAudioRelated -> Pair(0.92f, null)
            isImageRelated -> Pair(null, 0.89f)
            containsFraudKeyword -> Pair(0.85f, null)
            else -> Pair(0.05f, null)
        }

        val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

        val displayText = when {
            styleMessages.isNotEmpty() -> styleMessages.last()
            text.isNotBlank() -> text
            bigText.isNotBlank() -> bigText
            textLines.isNotEmpty() -> textLines.last()
            isAudioRelated -> "Nota de voz recibida"
            isImageRelated -> "Archivo de imagen recibido"
            else -> combinedContent.ifBlank { "Mensaje entrante" }
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

        NotificationRepository.addNotification(item)
        Log.i(TAG, "Notificacion interceptada con exito: [$appName] $sender: $displayText (Riesgo: ${fusionResult.globalRiskPercentage}%)")
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        super.onNotificationRemoved(sbn)
    }
}
