package com.detectorpreventor.app.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class AudioSpectrogramGeneratorTest {

    @Test
    fun testMultiScaleRgbMatchesPythonFrontendReference() {
        val sampleCount = 48000
        val signal = FloatArray(sampleCount) { n ->
            val t = n.toDouble() / 16000.0
            val noise = ((((n.toLong() * 1103515245L + 12345L) and 0x7FFFL) / 16384.0) - 1.0) * 0.005
            (0.5 * sin(2.0 * PI * 440.0 * t) +
                0.25 * sin(2.0 * PI * 5200.0 * t) +
                0.1 * cos(2.0 * PI * 1850.0 * t) +
                noise).toFloat()
        }

        val res = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(signal)
        assertEquals(224 * 224, res.pixels.size)
        assertTrue(res.isVoiceDetected)
        assertTrue(res.dynamicRangeDb >= 40.0f)

        val expectedSpots = arrayOf(
            intArrayOf(0, 0, 0, 129, 43),
            intArrayOf(10, 50, 0, 132, 51),
            intArrayOf(78, 100, 0, 129, 32),
            intArrayOf(78, 112, 0, 132, 17),
            intArrayOf(112, 112, 34, 140, 159),
            intArrayOf(171, 60, 56, 163, 92),
            intArrayOf(200, 150, 5, 145, 19),
            intArrayOf(211, 10, 15, 123, 0),
            intArrayOf(211, 120, 16, 124, 0),
            intArrayOf(223, 223, 0, 62, 0)
        )

        var maxChannelDiff = 0
        for (spot in expectedSpots) {
            val py = spot[0]
            val px = spot[1]
            val argb = res.pixels[py * 224 + px]
            val r = (argb shr 16) and 0xFF
            val g = (argb shr 8) and 0xFF
            val b = argb and 0xFF
            maxChannelDiff = maxOf(
                maxChannelDiff,
                abs(r - spot[2]),
                abs(g - spot[3]),
                abs(b - spot[4])
            )
            assertTrue(abs(r - spot[2]) <= 1, "R mismatch at ($py,$px): got=$r exp=${spot[2]}")
            assertTrue(abs(g - spot[3]) <= 1, "G mismatch at ($py,$px): got=$g exp=${spot[3]}")
            assertTrue(abs(b - spot[4]) <= 1, "B mismatch at ($py,$px): got=$b exp=${spot[4]}")
        }

        val chSums = DoubleArray(3)
        for (argb in res.pixels) {
            chSums[0] += ((argb shr 16) and 0xFF).toDouble()
            chSums[1] += ((argb shr 8) and 0xFF).toDouble()
            chSums[2] += (argb and 0xFF).toDouble()
        }
        val nPx = res.pixels.size.toDouble()
        assertEquals(21.145, chSums[0] / nPx, 0.05)
        assertEquals(127.565, chSums[1] / nPx, 0.05)
        assertEquals(65.701, chSums[2] / nPx, 0.05)
        assertTrue(maxChannelDiff <= 1)
    }

    @Test
    fun testVadAndTilingMatchPythonFrontendReference() {
        val sig = FloatArray(48000)
        for (n in 0 until 11200) {
            sig[n] = (sin(2.0 * PI * 50.0 * n / 16000.0) * 0.002).toFloat()
        }
        for (n in 0 until 14400) {
            val t = n.toDouble() / 16000.0
            val noise = ((((n.toLong() * 1103515245L + 12345L) and 0x7FFFL) / 16384.0) - 1.0) * 0.003
            sig[11200 + n] = (0.7 * sin(2.0 * PI * 320.0 * t) + 0.2 * cos(2.0 * PI * 1900.0 * t) + noise).toFloat()
        }
        val res = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(sig)
        val expectedSpots = arrayOf(
            intArrayOf(15, 20, 0, 118, 73),
            intArrayOf(112, 112, 10, 117, 157),
            intArrayOf(170, 50, 8, 132, 73),
            intArrayOf(200, 180, 62, 119, 191)
        )
        for (spot in expectedSpots) {
            val argb = res.pixels[spot[0] * 224 + spot[1]]
            val r = (argb shr 16) and 0xFF
            val g = (argb shr 8) and 0xFF
            val b = argb and 0xFF
            assertTrue(abs(r - spot[2]) <= 1, "VAD R mismatch at (${spot[0]},${spot[1]}): got=$r exp=${spot[2]}")
            assertTrue(abs(g - spot[3]) <= 1, "VAD G mismatch at (${spot[0]},${spot[1]}): got=$g exp=${spot[3]}")
            assertTrue(abs(b - spot[4]) <= 1, "VAD B mismatch at (${spot[0]},${spot[1]}): got=$b exp=${spot[4]}")
        }
        val chSums = DoubleArray(3)
        for (argb in res.pixels) {
            chSums[0] += ((argb shr 16) and 0xFF).toDouble()
            chSums[1] += ((argb shr 8) and 0xFF).toDouble()
            chSums[2] += (argb and 0xFF).toDouble()
        }
        val nPx = res.pixels.size.toDouble()
        assertEquals(26.322, chSums[0] / nPx, 0.05)
        assertEquals(125.504, chSums[1] / nPx, 0.05)
        assertEquals(88.788, chSums[2] / nPx, 0.05)
    }

    @Test
    fun testSilenceAndAmplitudeInvariance() {
        val silence = FloatArray(48000)
        val silRes = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(silence)
        assertEquals(224 * 224, silRes.pixels.size)
        assertFalse(silRes.isVoiceDetected)

        val toneFull = FloatArray(48000) { n ->
            val t = n.toDouble() / 16000.0
            val pseudoNoise = (((n * 1103515245 + 12345) and 0x7FFF) / 16384.0 - 1.0) * 0.005
            (0.6 * sin(2.0 * PI * 320.0 * t) +
                0.25 * sin(2.0 * PI * 2400.0 * t) +
                0.10 * cos(2.0 * PI * 5100.0 * t) +
                pseudoNoise).toFloat()
        }
        val toneScaled = FloatArray(48000) { n -> toneFull[n] * 0.2f }
        val r1 = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(toneFull)
        val r2 = AudioSpectrogramGenerator.computeMultiScaleRgbPixels(toneScaled)
        var maxDiff = 0
        for (i in r1.pixels.indices) {
            val a1 = r1.pixels[i]
            val a2 = r2.pixels[i]
            maxDiff = maxOf(
                maxDiff,
                abs(((a1 shr 16) and 0xFF) - ((a2 shr 16) and 0xFF)),
                abs(((a1 shr 8) and 0xFF) - ((a2 shr 8) and 0xFF)),
                abs((a1 and 0xFF) - (a2 and 0xFF))
            )
        }
        assertTrue(maxDiff <= 1, "Expected amplitude-invariant normalization, maxDiff=$maxDiff")
    }

    @Test
    fun testParseWavPcmToMono16kDecodesRiffPcm() {
        val sampleRate = 16000
        val frames = 1600
        val dataBytes = frames * 2
        val wav = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        wav.put("RIFF".toByteArray(Charsets.US_ASCII))
        wav.putInt(36 + dataBytes)
        wav.put("WAVE".toByteArray(Charsets.US_ASCII))
        wav.put("fmt ".toByteArray(Charsets.US_ASCII))
        wav.putInt(16)
        wav.putShort(1.toShort()) // PCM
        wav.putShort(1.toShort()) // mono
        wav.putInt(sampleRate)
        wav.putInt(sampleRate * 2)
        wav.putShort(2.toShort())
        wav.putShort(16.toShort())
        wav.put("data".toByteArray(Charsets.US_ASCII))
        wav.putInt(dataBytes)
        for (i in 0 until frames) {
            val s = (sin(2.0 * PI * 440.0 * i / sampleRate) * 16384.0).toInt().toShort()
            wav.putShort(s)
        }
        val parsed = AudioSpectrogramGenerator.parseWavPcmToMono16k(wav.array())
        assertTrue(parsed != null && parsed.size == frames)
    }

    @Test
    fun testDownsampleRejectsNyquistAliasingAt44100Hz() {
        val sr = 44100
        val n = sr // 1 second
        val sig = FloatArray(n) { i ->
            val t = i.toDouble() / sr.toDouble()
            // 13 kHz tone (> 8 kHz Nyquist of 16 kHz) that naive linear interpolation aliases into ~3 kHz
            sin(2.0 * PI * 13000.0 * t).toFloat()
        }
        val down = AudioSpectrogramGenerator.resampleAudio(sig, sr, 16000)
        assertEquals((n * (16000.0 / sr)).toInt(), down.size)
        // Skip filter edge transient (first/last 64 samples), verify >40 dB attenuation (rms < 0.01)
        var sumSq = 0.0
        var cnt = 0
        for (i in 64 until down.size - 64) {
            sumSq += (down[i] * down[i]).toDouble()
            cnt++
        }
        val rms = kotlin.math.sqrt(sumSq / cnt)
        assertTrue(rms < 0.01, "Expected >40 dB aliasing rejection above 8 kHz, got rms=$rms")
    }
}

