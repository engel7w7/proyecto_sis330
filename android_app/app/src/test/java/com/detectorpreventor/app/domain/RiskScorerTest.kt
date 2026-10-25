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

    @Test
    fun testDefaultMultimodalWeightsFiftyFiveFortyFive() {
        val audioProb = 0.40f
        val visionProb = 0.60f

        val result = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

        assertNotNull(result)
        // 0.55 * 0.40 + 0.45 * 0.60 = 0.22 + 0.27 = 0.49 -> 49.0%
        assertEquals(49.0f, result.globalRiskPercentage, 0.1f)
        assertEquals(0.55f, result.weightAudio, 0.01f)
        assertEquals(0.45f, result.weightVision, 0.01f)
        assertEquals(RiskLevel.MEDIO, result.riskLevel)
    }

    @Test
    fun testVideoRuleThreeOfFiveMajorityAlteredFrames() {
        val frameProbs = listOf(0.85f, 0.90f, 0.78f, 0.12f, 0.15f)
        val audioProb = 0.10f

        val result = RiskScorer.calculateVideoRisk(audioProb, frameProbs)

        assertNotNull(result)
        assertEquals(RiskLevel.ALTO, result.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(result.globalRiskPercentage >= 85.0f)
        org.junit.jupiter.api.Assertions.assertTrue(result.diagnosticSummary.contains("Regla 3/5 Video"))
    }

    @Test
    fun testVideoRuleThreeOfFiveCleanFramesAndAudio() {
        val frameProbs = listOf(0.12f, 0.10f, 0.15f, 0.08f, 0.11f)
        val audioProb = 0.05f

        val result = RiskScorer.calculateVideoRisk(audioProb, frameProbs)

        assertNotNull(result)
        assertEquals(RiskLevel.BAJO, result.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(result.globalRiskPercentage < 30.0f)
    }
}
