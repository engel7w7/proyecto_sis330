package com.sis330.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pruebas Unitarias para el algoritmo de Fusión Tardía y Regla 3/5 en RiskScorer.
 * Proyecto SIS-330 - Ing. Pacheco.
 */
class RiskScorerTest {

    @Test
    fun testRule35TriggeredWhen3OrMoreFramesAreFake() {
        val audioProb = 0.90f
        // 4 frames superan el umbral 0.50f
        val keyframes = listOf(0.85f, 0.92f, 0.78f, 0.65f, 0.20f)

        val result = RiskScorer.calculateRisk(audioProb, keyframes)

        assertTrue(result.rule35Triggered)
        assertEquals(4, result.framesFlaggedCount)
        assertEquals("Estafa Inminente", result.verdict)
        assertTrue(result.globalRiskScore > 70.0f)
    }

    @Test
    fun testRule35AvoidsFalsePositiveWhenOnly1Or2FramesAreNoisy() {
        val audioProb = 0.15f // Audio seguro
        // Solo 2 frames con ruido/anomalía aislada
        val keyframes = listOf(0.60f, 0.55f, 0.10f, 0.12f, 0.08f)

        val result = RiskScorer.calculateRisk(audioProb, keyframes)

        // No debe activar la regla 3/5
        assertTrue(!result.rule35Triggered)
        assertEquals(2, result.framesFlaggedCount)
        assertEquals("Seguro", result.verdict)
        assertTrue(result.globalRiskScore < 35.0f)
    }

    @Test
    fun testSafeMediaReturnsLowRisk() {
        val audioProb = 0.10f
        val keyframes = listOf(0.05f, 0.12f, 0.08f, 0.15f, 0.04f)

        val result = RiskScorer.calculateRisk(audioProb, keyframes)

        assertEquals("Seguro", result.verdict)
        assertTrue(result.globalRiskScore < 20.0f)
    }
}
