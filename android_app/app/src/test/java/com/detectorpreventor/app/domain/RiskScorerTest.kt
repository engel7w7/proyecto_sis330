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

        assertEquals(85.5f, result.globalRiskPercentage, 0.1f)
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

    @Test
    fun testVideoRuleIgnoresBorderlineJitterBelowMeanThreshold() {
        val frameProbs = listOf(0.51f, 0.50f, 0.50f, 0.42f, 0.45f)
        val audioProb = 0.52f

        val result = RiskScorer.calculateVideoRisk(audioProb, frameProbs)

        assertNotNull(result)
        assertEquals(RiskLevel.MEDIO, result.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(result.globalRiskPercentage < 70.0f)
    }

    @Test
    fun testFormatNotificationSummaryIncludesMediumLevel() {
        assertEquals(
            "Fotografía con indicios moderados de alteración (Verificación recomendada)",
            RiskScorer.formatNotificationSummary("IMAGE", 55.0f, isFaceDetected = true)
        )
        assertEquals(
            "Nota de voz con indicios moderados de clonación (Verificación recomendada)",
            RiskScorer.formatNotificationSummary("AUDIO", 42.0f)
        )
        assertEquals(
            "Video con indicios moderados de alteración (Verificación recomendada)",
            RiskScorer.formatNotificationSummary("VIDEO", 60.0f)
        )
        assertEquals(
            "Fotografía recibida (Descarte automático: Sin rostro humano)",
            RiskScorer.formatNotificationSummary("IMAGE", 0.0f, isFaceDetected = false)
        )
    }

    @Test
    fun testLateFusionQualityWeightingAttenuatesDegradedConflict() {
        val result = RiskScorer.calculateGlobalRisk(
            audioProb = 0.08f,
            visionProb = 0.82f,
            isFaceDetected = true,
            audioConfidence = 1.0f,
            visionSharpness = 0.45f
        )

        org.junit.jupiter.api.Assertions.assertTrue(result.weightAudio > 0.70f)
        org.junit.jupiter.api.Assertions.assertTrue(result.globalRiskPercentage < 70.0f)
    }

    @Test
    fun testLateFusionDualModerateSynergyBoostsToHigh() {
        val result = RiskScorer.calculateGlobalRisk(
            audioProb = 0.60f,
            visionProb = 0.62f,
            isFaceDetected = true
        )

        assertEquals(RiskLevel.ALTO, result.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(result.globalRiskPercentage >= 70.0f)
    }

    @Test
    fun testVideoRuleThreeOfFivePartialFakeAndBimodalConsensus() {
        val partialFake = RiskScorer.calculateVideoRisk(
            audioProb = 0.12f,
            frameProbs = listOf(0.72f, 0.68f, 0.58f, 0.15f, 0.15f),
            isFaceDetected = true
        )
        assertEquals(RiskLevel.ALTO, partialFake.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(partialFake.globalRiskPercentage >= 85.0f)

        val bimodalFake = RiskScorer.calculateVideoRisk(
            audioProb = 0.58f,
            frameProbs = listOf(0.62f, 0.58f, 0.12f, 0.10f, 0.15f),
            isFaceDetected = true
        )
        assertEquals(RiskLevel.ALTO, bimodalFake.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(bimodalFake.diagnosticSummary.contains("Regla 3/5 Video"))
    }

    @Test
    fun testVideoRuleThreeOfFiveUpdatesHeaderSummaryConsistency() {
        val frameDominant = RiskScorer.calculateVideoRisk(
            audioProb = 0.0f,
            frameProbs = listOf(0.65f, 0.70f, 0.68f, 0.60f, 0.62f),
            isFaceDetected = true
        )
        assertEquals(RiskLevel.ALTO, frameDominant.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(frameDominant.globalRiskPercentage >= 85.0f)
        org.junit.jupiter.api.Assertions.assertTrue(
            frameDominant.diagnosticSummary.startsWith("Veredicto Final: 85.0% de Riesgo (ALTO).")
        )
        org.junit.jupiter.api.Assertions.assertFalse(
            frameDominant.diagnosticSummary.contains("de Riesgo (MEDIO)")
        )
        org.junit.jupiter.api.Assertions.assertTrue(
            frameDominant.diagnosticSummary.contains("Regla 3/5 Video")
        )

        val audioCloned = RiskScorer.calculateVideoRisk(
            audioProb = 0.72f,
            frameProbs = listOf(0.05f, 0.05f, 0.05f, 0.05f, 0.05f),
            isFaceDetected = true
        )
        assertEquals(RiskLevel.ALTO, audioCloned.riskLevel)
        org.junit.jupiter.api.Assertions.assertTrue(audioCloned.globalRiskPercentage >= 85.0f)
        val expectedHeader = String.format(
            java.util.Locale.US,
            "Veredicto Final: %.1f%% de Riesgo (ALTO).",
            audioCloned.globalRiskPercentage
        )
        org.junit.jupiter.api.Assertions.assertTrue(
            audioCloned.diagnosticSummary.startsWith(expectedHeader)
        )
        org.junit.jupiter.api.Assertions.assertFalse(
            audioCloned.diagnosticSummary.contains("de Riesgo (MEDIO)")
        )
        org.junit.jupiter.api.Assertions.assertFalse(
            audioCloned.diagnosticSummary.contains("de Riesgo (BAJO)")
        )
    }
}
