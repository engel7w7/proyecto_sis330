package com.detectorpreventor.app.domain

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test

class RiskScorerTest {

    @Test
    fun testMultimodalFusionScoreCalculation() {
        val audioProb = 0.80f
        val visionProb = 0.90f

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
