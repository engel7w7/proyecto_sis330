package com.detectorpreventor.app.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class AudioClassifierTest {

    @Test
    fun testComputeCalibratedProbFakeTwoLogitsAndTemperature() {
        assertEquals(0.0f, AudioClassifier.computeCalibratedProbFake(FloatArray(0)), 1e-6f)

        val equalProb = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(1.5f, 1.5f),
            temperature = 1.25f,
            fakeIndex = 1
        )
        assertEquals(0.5f, equalProb, 1e-4f)

        val highFake = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(-3.0f, 3.0f),
            temperature = 1.0f,
            fakeIndex = 1
        )
        assertTrue(highFake > 0.99f, "Expected probFake > 0.99 for strong fake logit, got $highFake")

        val softerWithHigherTemp = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(-1.5f, 1.5f),
            temperature = 2.5f,
            fakeIndex = 1
        )
        val sharperWithUnitTemp = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(-1.5f, 1.5f),
            temperature = 1.0f,
            fakeIndex = 1
        )
        assertTrue(softerWithHigherTemp in 0.51f..sharperWithUnitTemp)
    }

    @Test
    fun testComputeCalibratedProbFakeSingleLogitAndClamping() {
        val singleHigh = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(100.0f),
            temperature = 1.0f
        )
        assertEquals(0.999f, singleHigh, 1e-5f)

        val singleLow = AudioClassifier.computeCalibratedProbFake(
            logits = floatArrayOf(-100.0f),
            temperature = 1.0f
        )
        assertEquals(0.001f, singleLow, 1e-5f)
    }

    @Test
    fun testComputeNormalizedGradCamGridProducesSpatialContrast() {
        val gridH = 2
        val gridW = 2
        val channels = 3
        val weights = floatArrayOf(1.0f, -0.5f, 2.0f)
        val featureGrid = floatArrayOf(
            0.0f, 2.0f, 0.0f,
            1.0f, 0.0f, 0.5f,
            2.0f, 0.0f, 1.0f,
            0.5f, 1.0f, 0.5f
        )
        val cam = AudioClassifier.computeNormalizedGradCamGrid(
            featureGrid = featureGrid,
            gridH = gridH,
            gridW = gridW,
            channels = channels,
            weights = weights
        )
        assertEquals(0.0f, cam[0][0], 1e-5f)
        assertEquals(0.5f, cam[0][1], 1e-5f)
        assertEquals(1.0f, cam[1][0], 1e-5f)
        assertEquals(0.25f, cam[1][1], 1e-5f)
    }

    @Test
    fun testNormalizePcm48kPadsCropsPeakNormalizesAndHandlesSilence() {
        val shortClip = floatArrayOf(0.1f, -0.2f, 0.05f)
        val normShort = AudioClassifier.normalizePcm48k(shortClip)
        assertEquals(48000, normShort.size)
        assertEquals(0.5f, normShort[0], 1e-5f)
        assertEquals(-1.0f, normShort[1], 1e-5f)
        assertEquals(0.25f, normShort[2], 1e-5f)
        assertEquals(0.0f, normShort[47999], 1e-6f)

        val longClip = FloatArray(60000) { i -> if (i == 10) -0.4f else 0.02f }
        val normLong = AudioClassifier.normalizePcm48k(longClip)
        assertEquals(48000, normLong.size)
        assertEquals(-1.0f, normLong[10], 1e-5f)

        val nearSilence = FloatArray(1000) { 0.0001f }
        val normSilence = AudioClassifier.normalizePcm48k(nearSilence)
        assertEquals(48000, normSilence.size)
        assertEquals(0.0001f, normSilence[0], 1e-6f)
    }

    @Test
    fun testComputeTemporalFrameImportancePolarityAndAttentionFallback() {
        val numFrames = 3
        val hiddenDim = 2
        val hiddenStates = floatArrayOf(
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 0.0f
        )
        val attWeights = floatArrayOf(0.5f, 0.2f, 0.3f)
        val contrastWeights = floatArrayOf(1.0f, -2.0f)

        val fakeDominantScores = AudioClassifier.computeTemporalFrameImportance(
            hiddenStates = hiddenStates,
            attWeights = attWeights,
            numFrames = numFrames,
            hiddenDim = hiddenDim,
            contrastWeights = contrastWeights,
            isFakeDominant = true
        )
        assertEquals(3, fakeDominantScores.size)
        assertTrue(fakeDominantScores[2] > fakeDominantScores[0])
        assertEquals(0.0f, fakeDominantScores[1], 1e-5f)
        assertEquals(1.0f, fakeDominantScores[2], 1e-5f)

        val realDominantScores = AudioClassifier.computeTemporalFrameImportance(
            hiddenStates = hiddenStates,
            attWeights = attWeights,
            numFrames = numFrames,
            hiddenDim = hiddenDim,
            contrastWeights = contrastWeights,
            isFakeDominant = false
        )
        assertEquals(3, realDominantScores.size)
        assertEquals(1.0f, realDominantScores[1], 1e-5f)
        assertEquals(0.0f, realDominantScores[0], 1e-5f)

        val fallbackScores = AudioClassifier.computeTemporalFrameImportance(
            hiddenStates = FloatArray(numFrames * hiddenDim) { 0.0f },
            attWeights = attWeights,
            numFrames = numFrames,
            hiddenDim = hiddenDim,
            contrastWeights = contrastWeights,
            isFakeDominant = true
        )
        assertEquals(1.0f, fallbackScores[0], 1e-5f)
        assertEquals(0.0f, fallbackScores[1], 1e-5f)
    }
}

