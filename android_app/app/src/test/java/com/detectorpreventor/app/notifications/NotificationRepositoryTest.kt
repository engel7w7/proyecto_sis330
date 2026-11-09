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
        NotificationRepository.clearBlockedSenders()
        NotificationRepository.setMonitoringActive(true)
        NotificationRepository.setServiceConnected(false)
        NotificationMediaCache.clear()
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
            fusionResult = fusion,
            mediaUriString = "asset://samples/06_clonacion_ia_spoof_tts_1_spec.png"
        )

        val json = original.toJson()
        assertNotNull(json)
        assertEquals("test-notif-123", json.getString("id"))
        assertEquals("WhatsApp", json.getString("appName"))
        assertEquals("com.whatsapp", json.getString("packageName"))
        assertEquals("Mama (Urgente)", json.getString("sender"))
        assertEquals("AUDIO_ONLY", json.getString("mediaType"))
        assertTrue(json.getBoolean("isThreat"))
        assertEquals("asset://samples/06_clonacion_ia_spoof_tts_1_spec.png", json.getString("mediaUriString"))

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
        assertEquals(original.mediaUriString, restored.mediaUriString)
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

        val duplicate = notif1.copy(id = UUID.randomUUID().toString(), timestamp = now + 500L)
        NotificationRepository.addNotification(duplicate)
        assertEquals(1, NotificationRepository.notifications.value.size, "El duplicado inmediato debio ser descartado")

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

    @Test
    fun testUpdateExistingNotification() {
        val initial = InterceptedNotification(
            id = "notif-pending-1",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Banco Union",
            text = "Descarga de audio pendiente...",
            timestamp = 1000L,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 0f,
            isThreat = false
        )

        NotificationRepository.addNotification(initial)
        assertEquals(1, NotificationRepository.notifications.value.size)
        assertEquals("Descarga de audio pendiente...", NotificationRepository.notifications.value[0].text)

        val updated = initial.copy(
            text = "Nota de voz analizada (0:15)",
            riskScore = 91.5f,
            isThreat = true
        )
        NotificationRepository.updateNotification(updated)

        assertEquals(1, NotificationRepository.notifications.value.size)
        val current = NotificationRepository.notifications.value[0]
        assertEquals("Nota de voz analizada (0:15)", current.text)
        assertEquals(91.5f, current.riskScore, 0.01f)
        assertTrue(current.isThreat)
    }

    @Test
    fun testAddNotificationReplacesPendingNotificationForSameSenderAndType() {
        val now = System.currentTimeMillis()
        val pending = InterceptedNotification(
            id = "pending-id-1",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Tia Maria",
            text = "Nota de voz (descarga pendiente)",
            timestamp = now - 5000L,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 0f,
            isThreat = false
        )
        NotificationRepository.addNotification(pending)
        assertEquals(1, NotificationRepository.notifications.value.size)
        assertTrue(NotificationRepository.notifications.value[0].text.contains("pendiente"))

        val analyzed = InterceptedNotification(
            id = "analyzed-id-2",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Tia Maria",
            text = "Nota de voz (0:12)",
            timestamp = now,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 88.0f,
            isThreat = true
        )
        NotificationRepository.addNotification(analyzed)

        assertEquals(1, NotificationRepository.notifications.value.size)
        assertEquals("analyzed-id-2", NotificationRepository.notifications.value[0].id)
        assertEquals(88.0f, NotificationRepository.notifications.value[0].riskScore, 0.01f)
        assertTrue(NotificationRepository.notifications.value[0].isThreat)
    }

    @Test
    fun testMediaUriStringSerialization() {
        val notif = InterceptedNotification(
            id = "notif-uri-1",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Banco",
            text = "Audio importante",
            timestamp = 1715000000000L,
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 95.0f,
            isThreat = true,
            mediaUriString = "asset://samples/06_clonacion_ia_spoof_tts_1_spec.png"
        )

        val json = notif.toJson()
        assertEquals("asset://samples/06_clonacion_ia_spoof_tts_1_spec.png", json.getString("mediaUriString"))

        val restored = InterceptedNotification.fromJson(json)
        assertEquals("asset://samples/06_clonacion_ia_spoof_tts_1_spec.png", restored.mediaUriString)
    }

    @Test
    fun testFileBindingMechanism() {
        val path1 = "/storage/emulated/0/WhatsApp/Media/WhatsApp Voice Notes/PTT-001.opus"
        val path2 = "/storage/emulated/0/WhatsApp/Media/WhatsApp Voice Notes/PTT-002.opus"

        assertFalse(NotificationMediaCache.isFileBound(path1))
        assertFalse(NotificationMediaCache.isFileBound(path2))

        NotificationMediaCache.markFileBound(path1)
        assertTrue(NotificationMediaCache.isFileBound(path1))
        assertFalse(NotificationMediaCache.isFileBound(path2))

        NotificationMediaCache.releaseFile(path1)
        assertFalse(NotificationMediaCache.isFileBound(path1))
    }

    @Test
    fun testIsPendingDownloadDetection() {
        val notifPending1 = InterceptedNotification(
            id = "p1",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Mama",
            text = "Audio (Descargando en WhatsApp...)",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.AUDIO_ONLY,
            riskScore = 0.0f,
            isThreat = false
        )
        assertTrue(notifPending1.isPendingDownload())

        val notifPending2 = notifPending1.copy(text = "Descarga pendiente...")
        assertTrue(notifPending2.isPendingDownload())

        val notifResolved = notifPending1.copy(text = "Audio analizado (Voz humana)", riskScore = 4.2f)
        assertFalse(notifResolved.isPendingDownload())
    }

    @Test
    fun testBlockSenderAndIsSenderBlocked() {
        assertFalse(NotificationRepository.isSenderBlocked("Mama (Urgente)"))
        NotificationRepository.blockSender("Mama (Urgente)")
        assertTrue(NotificationRepository.isSenderBlocked("Mama (Urgente)"))
        assertTrue(NotificationRepository.isSenderBlocked("mama (urgente)"))
        assertTrue(NotificationRepository.isSenderBlocked("Mama"))

        NotificationRepository.unblockSender("Mama (Urgente)")
        assertFalse(NotificationRepository.isSenderBlocked("Mama (Urgente)"))
    }

    @Test
    fun testRemoveNotification() {
        val notif = InterceptedNotification(
            id = "notif-remove-1",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Contacto Sospechoso",
            text = "Prueba de eliminacion",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.IMAGE_ONLY,
            riskScore = 80.0f,
            isThreat = true
        )
        NotificationRepository.addNotification(notif)
        assertEquals(1, NotificationRepository.notifications.value.size)

        val removed = NotificationRepository.removeNotification("notif-remove-1")
        assertTrue(removed)
        assertTrue(NotificationRepository.notifications.value.isEmpty())

        val removedAgain = NotificationRepository.removeNotification("notif-remove-1")
        assertFalse(removedAgain)
    }

    @Test
    fun testRemoveNotificationByFilenameOrSender() {
        val notif = InterceptedNotification(
            id = "notif-remove-query",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Remitente Especial",
            text = "Video sospechoso especial",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.VIDEO_MULTIMODAL,
            riskScore = 85.0f,
            isThreat = true,
            mediaUriString = "file:///sdcard/media/especial.mp4"
        )
        NotificationRepository.addNotification(notif)
        assertEquals(1, NotificationRepository.notifications.value.size)

        val removed = NotificationRepository.removeNotificationByFilenameOrSender("especial")
        assertTrue(removed)
        assertTrue(NotificationRepository.notifications.value.isEmpty())
    }

    @Test
    fun testFaceDiscardSerializationAndIsPendingDownload() {
        val fusionDiscard = FusionResult(
            globalRiskPercentage = 0.0f,
            audioFraudProb = 0.0f,
            visionFraudProb = 0.0f,
            riskLevel = RiskLevel.BAJO,
            weightAudio = 0.0f,
            weightVision = 1.0f,
            diagnosticSummary = "Descarte automatico BlazeFace",
            wasFaceDiscarded = true
        )

        val notifDiscard = InterceptedNotification(
            id = "notif-blazeface-discard",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Amigo",
            text = "Paisaje sin personas",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.IMAGE_ONLY,
            riskScore = 0.0f,
            isThreat = false,
            fusionResult = fusionDiscard
        )

        assertTrue(notifDiscard.isFaceDiscarded())
        assertFalse(notifDiscard.isPendingDownload(), "El descarte facial con riesgo 0 no debe considerarse descarga pendiente")

        val json = notifDiscard.toJson()
        val restored = InterceptedNotification.fromJson(json)

        assertTrue(restored.isFaceDiscarded())
        assertFalse(restored.isPendingDownload())
        assertEquals(0.0f, restored.riskScore, 0.01f)
        assertEquals(RiskLevel.BAJO, restored.fusionResult?.riskLevel)
        assertTrue(restored.fusionResult?.wasFaceDiscarded == true)
    }

    @Test
    fun testHasNotificationAndEdgePendingLogic() {
        val fusionDiscard = FusionResult(
            globalRiskPercentage = 0.0f,
            audioFraudProb = 0.0f,
            visionFraudProb = 0.0f,
            riskLevel = RiskLevel.BAJO,
            weightAudio = 0.0f,
            weightVision = 1.0f,
            diagnosticSummary = "Descarte facial",
            wasFaceDiscarded = true
        )
        val discardedWithPendingKeyword = InterceptedNotification(
            id = "discard-pending-text",
            appName = "WhatsApp",
            packageName = "com.whatsapp",
            sender = "Amigo",
            text = "Descarga de imagen de paisaje (sin personas)",
            timestamp = System.currentTimeMillis(),
            mediaType = MediaType.IMAGE_ONLY,
            riskScore = 0.0f,
            isThreat = false,
            fusionResult = fusionDiscard
        )
        NotificationRepository.addNotification(discardedWithPendingKeyword)
        assertTrue(NotificationRepository.hasNotification("discard-pending-text"))
        assertFalse(discardedWithPendingKeyword.isPendingDownload())

        assertTrue(NotificationRepository.removeNotification("discard-pending-text"))
        assertFalse(NotificationRepository.hasNotification("discard-pending-text"))
    }

    @Test
    fun testConcurrentRepositoryOperationsAreThreadSafe() {
        val threads = 8
        val opsPerThread = 25
        val pool = java.util.concurrent.Executors.newFixedThreadPool(threads)
        val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
        val latch = java.util.concurrent.CountDownLatch(threads)

        for (t in 0 until threads) {
            pool.submit {
                try {
                    for (i in 0 until opsPerThread) {
                        val id = "thread-$t-$i"
                        val notif = InterceptedNotification(
                            id = id,
                            appName = "WhatsApp",
                            packageName = "com.whatsapp",
                            sender = "Sender-$t",
                            text = "Msg $i (descargando...)",
                            timestamp = System.currentTimeMillis() + i,
                            mediaType = MediaType.IMAGE_ONLY,
                            riskScore = 0.0f,
                            isThreat = false
                        )
                        NotificationRepository.addNotification(notif)
                        NotificationRepository.updateNotification(
                            notif.copy(text = "Analizado $i", riskScore = 25.0f)
                        )
                        if (i % 3 == 0) {
                            NotificationRepository.blockSender("Blocked-$t-$i")
                        }
                        NotificationRepository.hasNotification(id)
                        if (i % 2 == 0) {
                            NotificationRepository.removeNotification(id)
                        }
                    }
                } catch (e: Throwable) {
                    errors.add(e)
                } finally {
                    latch.countDown()
                }
            }
        }

        latch.await()
        pool.shutdownNow()
        assertTrue(errors.isEmpty(), "Operaciones concurrentes lanzaron excepcion: ${errors.firstOrNull()?.message}")
        assertTrue(NotificationRepository.notifications.value.size <= 50)
    }
}
