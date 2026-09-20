package com.detectorpreventor.app.notifications

import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.util.UUID

class NotificationRepositoryTest {

    @BeforeEach
    fun setUp() {
        NotificationRepository.clearNotifications()
        NotificationRepository.setMonitoringActive(true)
        NotificationRepository.setServiceConnected(false)
    }

    @Test
    fun testNotificationJsonSerializationAndDeserialization() {
        val fusion = FusionResult(
            globalRiskPercentage = 94.0f,
            audioFraudProb = 0.94f,
            visionFraudProb = 0.0f,
            riskLevel = RiskLevel.ALTO,
            weightAudio = 1.0f,
            weightVision = 0.0f,
            diagnosticSummary = "Alerta de fraude por clonacion de voz"
        )

        val original = InterceptedNotification(
            id = "test-notif-123",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Mama (Urgente)",
            text = "Hijo hazme un deposito rapido (Nota de voz 0:14)",
            timestamp = 1715000000000L,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 94.0f,
            isThreat = true,
            fusionResult = fusion
        )

        val json = original.toJson()
        assertNotNull(json)
        assertEquals("test-notif-123", json.getString("id"))
        assertEquals("WhatsApp", json.getString("appName"))
        assertEquals("com.whatsapp", json.getString("packageName"))
        assertEquals("Mama (Urgente)", json.getString("sender"))
        assertEquals("AUDIO_ONLY", json.getString("mediaType"))
        assertTrue(json.getBoolean("isThreat"))

        val restored = InterceptedNotification.fromJson(json)
        assertEquals(original.id, restored.id)
        assertEquals(original.appName, restored.appName)
        assertEquals(original.packageName, restored.packageName)
        assertEquals(original.sender, restored.sender)
        assertEquals(original.text, restored.text)
        assertEquals(original.timestamp, restored.timestamp)
        assertEquals(original.mediaType, restored.mediaType)
        assertEquals(original.riskScore, restored.riskScore, 0.01f)
        assertEquals(original.isThreat, restored.isThreat)
        assertNotNull(restored.fusionResult)
        assertEquals(94.0f, restored.fusionResult!!.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.ALTO, restored.fusionResult!!.riskLevel)
        assertEquals("Alerta de fraude por clonacion de voz", restored.fusionResult!!.diagnosticSummary)
    }

    @Test
    fun testAddNotificationAndDeduplication() {
        val now = System.currentTimeMillis()
        val notif1 = InterceptedNotification(
            id = UUID.randomUUID().toString(),
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Carlos",
            text = "Audio 0:15",
            timestamp = now,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 85f,
            isThreat = true
        )

        NotificationRepository.addNotification(notif1)
        assertEquals(1, NotificationRepository.notifications.value.size)

        // Intento de duplicado inmediato (mismo sender y text dentro de 2000ms)
        val duplicate = notif1.copy(id = UUID.randomUUID().toString(), timestamp = now + 500L)
        NotificationRepository.addNotification(duplicate)
        assertEquals(1, NotificationRepository.notifications.value.size, "El duplicado inmediato debio ser descartado")

        // Mensaje diferente debe agregarse correctamente
        val notif2 = notif1.copy(
            id = UUID.randomUUID().toString(),
            text = "Foto recibida",
            mediaType = MediaType.IMAGE_ONLY,
            timestamp = now + 600L
        )
        NotificationRepository.addNotification(notif2)
        assertEquals(2, NotificationRepository.notifications.value.size)
    }

    @Test
    fun testServiceConnectionAndMonitoringToggle() {
        assertFalse(NotificationRepository.isServiceConnected.value)
        assertTrue(NotificationRepository.isMonitoringActive.value)

        NotificationRepository.setServiceConnected(true)
        assertTrue(NotificationRepository.isServiceConnected.value)

        NotificationRepository.setMonitoringActive(false)
        assertFalse(NotificationRepository.isMonitoringActive.value)

        NotificationRepository.setServiceConnected(false)
        assertFalse(NotificationRepository.isServiceConnected.value)
    }

    @Test
    fun testClearNotifications() {
        val notif = InterceptedNotification(
            id = "notif-abc",
            appName = "Telegram",
            packageName = "org.telegram.messenger",
            sender = "Contacto",
            text = "Mensaje",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.UNKNOWN,
            riskScore = 10f,
            isThreat = false
        )

        NotificationRepository.addNotification(notif)
        assertEquals(1, NotificationRepository.notifications.value.size)

        NotificationRepository.clearNotifications()
        assertTrue(NotificationRepository.notifications.value.isEmpty())
    }
}
