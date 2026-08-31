package com.detectorpreventor.app.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test

/**
 * Pruebas Unitarias en JUnit 5 para la verificación del módulo RiskScorer (Score-Level Fusion).
 */
class RiskScorerTest {

    @Test
    fun testMultimodalFusionScoreCalculation() {
        val audioProb = 0.80f  // 80% probabilidad de fraude en audio
        val visionProb = 0.90f // 90% probabilidad de fraude en visión

        // Fórmula: Score = (0.6 * 0.80) + (0.4 * 0.90) = 0.48 + 0.36 = 0.84 (84%)
        val result = RiskScorer.calculateGlobalRisk(audioProb, visionProb, 0.6f, 0.4f)

        assertNotNull(result)
        assertEquals(84.0f, result.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.ALTO, result.riskLevel)
    }

    @Test
    fun testUnimodalAudioOnlyFallback() {
        val audioProb = 0.25f

        val result = RiskScorer.calculateGlobalRisk(audioProb, null)

        assertEquals(25.0f, result.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.BAJO, result.riskLevel)
        assertEquals(1.0f, result.weightAudio)
        assertEquals(0.0f, result.weightVision)
    }

    @Test
    fun testUnimodalVisionOnlyFallback() {
        val visionProb = 0.50f

        val result = RiskScorer.calculateGlobalRisk(null, visionProb)

        assertEquals(50.0f, result.globalRiskPercentage, 0.01f)
        assertEquals(RiskLevel.MEDIO, result.riskLevel)
        assertEquals(0.0f, result.weightAudio)
        assertEquals(1.0f, result.weightVision)
    }
}
