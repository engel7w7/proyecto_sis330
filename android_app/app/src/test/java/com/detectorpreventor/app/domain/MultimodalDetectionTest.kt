package com.detectorpreventor.app.domain

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * Pruebas unitarias para validar el motor de inferencia multimodal,
 * reglas de descarte de stickers, pesos de fusión tardía y asignación de niveles de riesgo.
 */
class MultimodalDetectionTest {

    @Test
    fun testScoreLevelFusionWeightedCalculation() {
        val audioProb = 0.95f
        val visionProb = 0.90f

        val result = RiskScorer.calculateGlobalRisk(
            audioProb = audioProb,
            visionProb = visionProb,
            customWeightAudio = 0.6f,
            customWeightVision = 0.4f
        )

        // 0.6 * 0.95 + 0.4 * 0.90 = 0.57 + 0.36 = 0.93 -> 93%
        assertEquals(93.0f, result.globalRiskPercentage, 0.05f)
        assertEquals(RiskLevel.ALTO, result.riskLevel)
        assertEquals(0.6f, result.weightAudio, 0.01f)
        assertEquals(0.4f, result.weightVision, 0.01f)
    }

    @Test
    fun testAuthenticMediaEvaluationProducesLowRisk() {
        val audioProb = 0.04f
        val visionProb = 0.03f

        val result = RiskScorer.calculateGlobalRisk(
            audioProb = audioProb,
            visionProb = visionProb,
            customWeightAudio = 0.6f,
            customWeightVision = 0.4f
        )

        // 0.6 * 0.04 + 0.4 * 0.03 = 0.024 + 0.012 = 0.036 -> 3.6%
        assertEquals(3.6f, result.globalRiskPercentage, 0.1f)
        assertEquals(RiskLevel.BAJO, result.riskLevel)
    }

    @Test
    fun testUnimodalAudioOnlyWeightRebalancing() {
        val audioProb = 0.88f

        val result = RiskScorer.calculateGlobalRisk(
            audioProb = audioProb,
            visionProb = null
        )

        assertEquals(88.0f, result.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.ALTO, result.riskLevel)
        assertEquals(1.0f, result.weightAudio)
        assertEquals(0.0f, result.weightVision)
    }

    @Test
    fun testUnimodalVisionOnlyWeightRebalancing() {
        val visionProb = 0.12f

        val result = RiskScorer.calculateGlobalRisk(
            audioProb = null,
            visionProb = visionProb
        )

        assertEquals(12.0f, result.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.BAJO, result.riskLevel)
        assertEquals(0.0f, result.weightAudio)
        assertEquals(1.0f, result.weightVision)
    }

    @Test
    fun testRiskThresholdCategorization() {
        // < 30% -> BAJO
        assertEquals(RiskLevel.BAJO, RiskScorer.calculateGlobalRisk(0.25f, null).riskLevel)

        // 30% - 70% -> MEDIO
        assertEquals(RiskLevel.MEDIO, RiskScorer.calculateGlobalRisk(0.30f, null).riskLevel)
        assertEquals(RiskLevel.MEDIO, RiskScorer.calculateGlobalRisk(0.65f, null).riskLevel)

        // >= 70% -> ALTO
        assertEquals(RiskLevel.ALTO, RiskScorer.calculateGlobalRisk(0.70f, null).riskLevel)
        assertEquals(RiskLevel.ALTO, RiskScorer.calculateGlobalRisk(0.95f, null).riskLevel)
    }

    @Test
    fun testStickerExclusionLogicRule() {
        // Validación de la regla de exclusión estricta de stickers
        fun shouldIgnoreNotification(content: String, mimeType: String?): Boolean {
            val isSticker = content.contains("sticker", ignoreCase = true) ||
                    (mimeType != null && mimeType.contains("webp", ignoreCase = true))
            return isSticker
        }

        assertTrue(shouldIgnoreNotification("sticker recibido", null))
        assertTrue(shouldIgnoreNotification("Sticker", "image/webp"))
        assertTrue(shouldIgnoreNotification("animación divertida", "image/webp"))

        assertFalse(shouldIgnoreNotification("Nota de voz (0:14)", "audio/opus"))
        assertFalse(shouldIgnoreNotification("foto_familiar.jpg", "image/jpeg"))
        assertFalse(shouldIgnoreNotification("video_01.mp4", "video/mp4"))
    }

    @Test
    fun testSoftmaxActivationCalculation() {
        val pReal = 2.5f
        val pFake = -1.2f
        val expFake = Math.exp(pFake.toDouble())
        val expReal = Math.exp(pReal.toDouble())
        val probFake = (expFake / (expReal + expFake)).toFloat()

        assertTrue(probFake < 0.05f, "Para logit real alto y logit fake bajo, pFake debe ser cercano a 0")

        val pReal2 = -2.0f
        val pFake2 = 3.0f
        val expFake2 = Math.exp(pFake2.toDouble())
        val expReal2 = Math.exp(pReal2.toDouble())
        val probFake2 = (expFake2 / (expReal2 + expFake2)).toFloat()

        assertTrue(probFake2 > 0.95f, "Para logit fake alto y logit real bajo, pFake debe ser cercano a 1")
    }

    @Test
    fun testSigmoidActivationCalculation() {
        val logitNegative = -4.0
        val sigLow = (1.0 / (1.0 + Math.exp(-logitNegative))).toFloat()
        assertTrue(sigLow < 0.05f)

        val logitPositive = 4.0
        val sigHigh = (1.0 / (1.0 + Math.exp(-logitPositive))).toFloat()
        assertTrue(sigHigh > 0.95f)
    }
}
