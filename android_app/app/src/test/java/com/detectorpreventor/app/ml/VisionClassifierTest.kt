package com.detectorpreventor.app.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class VisionClassifierTest {

    @Test
    fun testAdaptiveCropPreservesSquarePortraitForCloseFace() {
        val rect = VisionClassifier.computeAdaptiveFaceCropRect(
            imgWidth = 600,
            imgHeight = 600,
            faceLeft = 140,
            faceTop = 120,
            faceRight = 460,
            faceBottom = 480,
            forceTightCrop = false
        )

        assertEquals(0, rect.left)
        assertEquals(0, rect.top)
        assertEquals(600, rect.right)
        assertEquals(600, rect.bottom)
    }

    @Test
    fun testAdaptiveCropExpandsGenerouslyOnTallPortraitScreenshot() {
        val rect = VisionClassifier.computeAdaptiveFaceCropRect(
            imgWidth = 540,
            imgHeight = 1080,
            faceLeft = 120,
            faceTop = 300,
            faceRight = 420,
            faceBottom = 600,
            forceTightCrop = false
        )

        assertEquals(0, rect.left)
        assertEquals(540, rect.right)
        assertTrue(rect.top < 150)
        assertTrue(rect.bottom > 740)
    }

    @Test
    fun testAdaptiveCropUsesTightMarginOnWideSceneOrForcedVideoFrame() {
        val sceneRect = VisionClassifier.computeAdaptiveFaceCropRect(
            imgWidth = 1280,
            imgHeight = 720,
            faceLeft = 520,
            faceTop = 200,
            faceRight = 680,
            faceBottom = 380,
            forceTightCrop = false
        )
        assertEquals(496, sceneRect.left)
        assertEquals(173, sceneRect.top)
        assertEquals(704, sceneRect.right)
        assertEquals(407, sceneRect.bottom)

        val forcedVideoRect = VisionClassifier.computeAdaptiveFaceCropRect(
            imgWidth = 600,
            imgHeight = 600,
            faceLeft = 140,
            faceTop = 120,
            faceRight = 460,
            faceBottom = 480,
            forceTightCrop = true
        )
        assertTrue(forcedVideoRect.width < 600)
        assertTrue(forcedVideoRect.height < 600)
        assertEquals(92, forcedVideoRect.left)
        assertEquals(508, forcedVideoRect.right)
    }
}

