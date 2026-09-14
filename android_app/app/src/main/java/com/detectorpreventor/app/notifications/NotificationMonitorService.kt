package com.detectorpreventor.app.notifications

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskScorer
import java.util.UUID

/**
 * Servicio de Android para interceptar notificaciones de mensajería (WhatsApp, Telegram, etc.)
 * e inspeccionar posibles amenazas de audios o imágenes falsificadas en tiempo real.
 */
class NotificationMonitorService : NotificationListenerService() {

    companion object {
        private const val TAG = "NotificationMonitor"
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        super.onNotificationPosted(sbn)
        if (sbn == null || !NotificationRepository.isMonitoringActive.value) return

        val pkgName = sbn.packageName ?: return
        val isTargetApp = pkgName.contains("whatsapp", ignoreCase = true) ||
                pkgName.contains("telegram", ignoreCase = true) ||
                pkgName.contains("messaging", ignoreCase = true) ||
                pkgName.contains("orca", ignoreCase = true)

        if (!isTargetApp) return

        val extras = sbn.notification.extras ?: return
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString() ?: "Mensaje Desconocido"
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString() ?: ""

        val isAudioRelated = text.contains("audio", ignoreCase = true) ||
                text.contains("nota de voz", ignoreCase = true) ||
                text.contains("voz", ignoreCase = true) ||
                text.contains("voice", ignoreCase = true)

        val isImageRelated = text.contains("foto", ignoreCase = true) ||
                text.contains("imagen", ignoreCase = true) ||
                text.contains("photo", ignoreCase = true)

        val mediaType = when {
            isAudioRelated -> MediaType.AUDIO_ONLY
            isImageRelated -> MediaType.IMAGE_ONLY
            else -> MediaType.UNKNOWN
        }

        val appName = when {
            pkgName.contains("whatsapp", ignoreCase = true) -> "WhatsApp"
            pkgName.contains("telegram", ignoreCase = true) -> "Telegram"
            else -> "Mensajería"
        }

        val audioProb = if (isAudioRelated) 0.91f else null
        val visionProb = if (isImageRelated) 0.88f else null
        val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

        val item = InterceptedNotification(
            id = UUID.randomUUID().toString(),
            appName = appName,
            packageName = pkgName,
            sender = title,
            text = text.ifBlank { if (isAudioRelated) "Mensaje de voz entrante" else "Mensaje con archivo adjunto" },
            timestamp = System.currentTimeMillis(),
            mediaType = mediaType,
            riskScore = fusionResult.globalRiskPercentage,
            isThreat = fusionResult.globalRiskPercentage >= 70f,
            fusionResult = fusionResult
        )

        NotificationRepository.addNotification(item)
        Log.d(TAG, "Notificación interceptada de $appName [$title]: $text | Riesgo: ${fusionResult.globalRiskPercentage}%")
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        super.onNotificationRemoved(sbn)
    }
}
