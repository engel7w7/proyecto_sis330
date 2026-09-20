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

        // Filtrar notificaciones continuas del sistema (WhatsApp Web activo, sincronizacion, llamadas activas)
        if (sbn.isOngoing) return

        val pkgName = sbn.packageName ?: return
        val isWhatsApp = pkgName.equals("com.whatsapp", ignoreCase = true) ||
                pkgName.equals("com.whatsapp.w4b", ignoreCase = true)
        val isTelegram = pkgName.contains("telegram", ignoreCase = true)
        val isMessaging = pkgName.contains("messaging", ignoreCase = true) ||
                pkgName.contains("orca", ignoreCase = true)

        if (!isWhatsApp && !isTelegram && !isMessaging) return

        val notification = sbn.notification ?: return
        val extras = notification.extras ?: return

        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()?.trim() ?: "Mensaje Desconocido"
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()?.trim() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()?.trim() ?: ""
        val ticker = notification.tickerText?.toString()?.trim() ?: ""
        val textLines = (extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES) ?: emptyArray())
            .mapNotNull { it?.toString()?.trim() }
            .filter { it.isNotBlank() }
            .joinToString(" | ")

        val combinedContent = listOf(text, bigText, textLines, ticker)
            .filter { it.isNotBlank() }
            .joinToString(" ")

        if (combinedContent.isBlank()) return

        // Deteccion robusta de notas de voz (espanol e ingles y patrones de duracion de WhatsApp como '0:15')
        val durationRegex = Regex("""\b\d{1,2}:\d{2}\b""")
        val isAudioRelated = combinedContent.contains("audio", ignoreCase = true) ||
                combinedContent.contains("nota de voz", ignoreCase = true) ||
                combinedContent.contains("mensaje de voz", ignoreCase = true) ||
                combinedContent.contains("voice message", ignoreCase = true) ||
                combinedContent.contains("voice note", ignoreCase = true) ||
                (isWhatsApp && durationRegex.containsMatchIn(combinedContent))

        val isImageRelated = combinedContent.contains("foto", ignoreCase = true) ||
                combinedContent.contains("imagen", ignoreCase = true) ||
                combinedContent.contains("photo", ignoreCase = true) ||
                combinedContent.contains("image", ignoreCase = true) ||
                combinedContent.contains("sticker", ignoreCase = true)

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

        // Si es multimedia sospechoso (voz o imagen en mensajeria), evaluar con scoring preventivo
        val audioProb = if (isAudioRelated) 0.91f else null
        val visionProb = if (isImageRelated) 0.88f else null
        val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

        val displayText = when {
            text.isNotBlank() -> text
            bigText.isNotBlank() -> bigText
            textLines.isNotBlank() -> textLines
            isAudioRelated -> "Nota de voz entrante"
            isImageRelated -> "Archivo de imagen recibido"
            else -> combinedContent
        }

        val item = InterceptedNotification(
            id = UUID.randomUUID().toString(),
            appName = appName,
            packageName = pkgName,
            sender = title,
            text = displayText,
            timestamp = System.currentTimeMillis(),
            mediaType = mediaType,
            riskScore = fusionResult.globalRiskPercentage,
            isThreat = fusionResult.globalRiskPercentage >= 70f,
            fusionResult = fusionResult
        )

        NotificationRepository.addNotification(item)
        Log.d(TAG, "Notificacion interceptada de $appName [$title]: $displayText | Riesgo: ${fusionResult.globalRiskPercentage}%")
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        super.onNotificationRemoved(sbn)
    }
}
