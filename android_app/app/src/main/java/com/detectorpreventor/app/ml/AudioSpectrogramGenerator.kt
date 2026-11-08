package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

object AudioSpectrogramGenerator {

    private const val TAG = "AudioSpectrogram"
    private const val TARGET_SAMPLE_RATE = 16000
    private const val TARGET_SAMPLES = 48000
    private const val FFT_SIZE = 512
    private const val HOP_SIZE = 256
    private const val NUM_MELS = 128
    private const val BITMAP_SIZE = 224

    data class AudioAnalysisResult(
        val spectrogramBitmap: Bitmap,
        val isVoiceDetected: Boolean,
        val hfToLfRatio: Float,
        val spectralFlux: Float,
        val spectralFlatness: Float,
        val rolloffFreq: Float,
        val dynamicRangeDb: Float
    )

    fun generateSpectrogram(context: Context, uri: Uri): AudioAnalysisResult {
        var pcmData: FloatArray? = null
        try {
            pcmData = decodeAudioUriToPcm(context, uri)
        } catch (e: Exception) {
            Log.w(TAG, "MediaCodec decoder fallo, intentando fallback de lectura directa: ${e.message}")
        }

        if (pcmData == null || pcmData.isEmpty()) {
            pcmData = readRawPcmFromStream(context, uri)
        }

        if (pcmData.isEmpty()) {
            Log.w(TAG, "No se obtuvieron muestras de audio validas, generando espectrograma sintetico de contingencia.")
            return AudioAnalysisResult(
                spectrogramBitmap = createFallbackSpectrogram(),
                isVoiceDetected = false,
                hfToLfRatio = 0.15f,
                spectralFlux = 0.00001f,
                spectralFlatness = 0.30f,
                rolloffFreq = 1800f,
                dynamicRangeDb = 25f
            )
        }

        val paddedData = if (pcmData.size < TARGET_SAMPLES) {
            val padded = FloatArray(TARGET_SAMPLES)
            System.arraycopy(pcmData, 0, padded, 0, pcmData.size)
            padded
        } else {
            pcmData.copyOfRange(0, TARGET_SAMPLES)
        }

        return computeMelSpectrogramAndFeatures(paddedData)
    }

    private fun decodeAudioUriToPcm(context: Context, uri: Uri): FloatArray? {
        val tempFile = File(context.cacheDir, "temp_decode_${System.currentTimeMillis()}.audio")
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            } ?: return null

            val extractor = MediaExtractor()
            extractor.setDataSource(tempFile.absolutePath)

            var audioTrackIndex = -1
            var inputFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    inputFormat = format
                    break
                }
            }

            if (audioTrackIndex == -1 || inputFormat == null) {
                extractor.release()
                return null
            }

            extractor.selectTrack(audioTrackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME) ?: ""
            val decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(inputFormat, null, null, 0)
            decoder.start()

            val srcSampleRate = if (inputFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE) else 44100
            val channelCount = if (inputFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

            val bufferInfo = MediaCodec.BufferInfo()
            val rawPcmSamples = mutableListOf<Float>()
            var isEos = false
            val timeoutUs = 5000L

            while (!isEos && rawPcmSamples.size < (TARGET_SAMPLES * 3)) {
                val inIndex = decoder.dequeueInputBuffer(timeoutUs)
                if (inIndex >= 0) {
                    val inputBuffer = decoder.getInputBuffer(inIndex)
                    if (inputBuffer != null) {
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            isEos = true
                        } else {
                            decoder.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                var outIndex = decoder.dequeueOutputBuffer(bufferInfo, timeoutUs)
                while (outIndex >= 0) {
                    val outputBuffer = decoder.getOutputBuffer(outIndex)
                    if (outputBuffer != null && bufferInfo.size > 0) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        outputBuffer.order(ByteOrder.LITTLE_ENDIAN)

                        val shortBuffer = outputBuffer.asShortBuffer()
                        while (shortBuffer.hasRemaining()) {
                            if (channelCount > 1) {
                                var sum = 0f
                                for (c in 0 until channelCount) {
                                    if (shortBuffer.hasRemaining()) {
                                        sum += shortBuffer.get() / 32768.0f
                                    }
                                }
                                rawPcmSamples.add(sum / channelCount)
                            } else {
                                rawPcmSamples.add(shortBuffer.get() / 32768.0f)
                            }
                        }
                    }
                    decoder.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        isEos = true
                        break
                    }
                    outIndex = decoder.dequeueOutputBuffer(bufferInfo, 0)
                }
            }

            decoder.stop()
            decoder.release()
            extractor.release()

            if (rawPcmSamples.isEmpty()) return null

            val originalArray = rawPcmSamples.toFloatArray()
            return if (srcSampleRate != TARGET_SAMPLE_RATE) {
                resampleAudio(originalArray, srcSampleRate, TARGET_SAMPLE_RATE)
            } else {
                originalArray
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error en decodificador MediaCodec: ${e.message}")
            return null
        } finally {
            if (tempFile.exists()) {
                tempFile.delete()
            }
        }
    }

    private fun readRawPcmFromStream(context: Context, uri: Uri): FloatArray {
        return try {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return FloatArray(0)
            if (bytes.size < 44) return FloatArray(0)

            val isWav = bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() &&
                    bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte()

            val startIndex = if (isWav) 44 else 0
            val numShorts = (bytes.size - startIndex) / 2
            val samples = FloatArray(minOf(numShorts, TARGET_SAMPLES))

            val buffer = ByteBuffer.wrap(bytes, startIndex, samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in samples.indices) {
                samples[i] = buffer.short / 32768.0f
            }
            samples
        } catch (e: Exception) {
            FloatArray(0)
        }
    }

    private fun resampleAudio(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        val ratio = dstRate.toDouble() / srcRate.toDouble()
        val outLength = (input.size * ratio).toInt()
        val output = FloatArray(outLength)

        for (i in 0 until outLength) {
            val srcIndex = i / ratio
            val index0 = srcIndex.toInt().coerceIn(0, input.size - 1)
            val index1 = (index0 + 1).coerceIn(0, input.size - 1)
            val frac = (srcIndex - index0).toFloat()
            output[i] = input[index0] * (1.0f - frac) + input[index1] * frac
        }
        return output
    }

    private fun computeMelSpectrogramAndFeatures(audioData: FloatArray): AudioAnalysisResult {
        var maxVal = 0.0f
        for (sample in audioData) {
            val abs = abs(sample)
            if (abs > maxVal) maxVal = abs
        }
        if (maxVal > 0.001f) {
            for (i in audioData.indices) {
                audioData[i] = audioData[i] / maxVal
            }
        }

        val numFrames = (audioData.size - FFT_SIZE) / HOP_SIZE + 1
        val window = FloatArray(FFT_SIZE) { i ->
            (0.5 * (1.0 - cos(2.0 * Math.PI * i / (FFT_SIZE - 1)))).toFloat()
        }

        val numBins = FFT_SIZE / 2 + 1
        val spectrogram = Array(numFrames) { FloatArray(numBins) }

        val real = FloatArray(FFT_SIZE)
        val imag = FloatArray(FFT_SIZE)

        for (t in 0 until numFrames) {
            val offset = t * HOP_SIZE
            for (i in 0 until FFT_SIZE) {
                real[i] = audioData[offset + i] * window[i]
                imag[i] = 0.0f
            }

            fftRadix2(real, imag)

            for (k in 0 until numBins) {
                val mag = sqrt(real[k] * real[k] + imag[k] * imag[k])
                spectrogram[t][k] = mag
            }
        }

        val melFilters = createMelFilterbank(NUM_MELS, FFT_SIZE, TARGET_SAMPLE_RATE)
        val melSpec = Array(NUM_MELS) { FloatArray(numFrames) }

        var minDb = Float.MAX_VALUE
        var maxDb = -Float.MAX_VALUE

        for (m in 0 until NUM_MELS) {
            val filter = melFilters[m]
            for (t in 0 until numFrames) {
                var energy = 0.0f
                for (k in 0 until numBins) {
                    energy += spectrogram[t][k] * filter[k]
                }
                val db = 10.0f * log10(maxOf(1e-10f, energy))
                melSpec[m][t] = db
                if (db < minDb) minDb = db
                if (db > maxDb) maxDb = db
            }
        }

        val rangeDb = maxOf(1.0f, maxDb - minDb)
        val bitmap = Bitmap.createBitmap(BITMAP_SIZE, BITMAP_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(BITMAP_SIZE * BITMAP_SIZE)

        for (py in 0 until BITMAP_SIZE) {
            val melIdx = ((BITMAP_SIZE - 1 - py).toFloat() / (BITMAP_SIZE - 1) * (NUM_MELS - 1)).toInt().coerceIn(0, NUM_MELS - 1)
            for (px in 0 until BITMAP_SIZE) {
                val frameIdx = (px.toFloat() / (BITMAP_SIZE - 1) * (numFrames - 1)).toInt().coerceIn(0, numFrames - 1)
                val normVal = ((melSpec[melIdx][frameIdx] - minDb) / rangeDb).coerceIn(0.0f, 1.0f)

                val rgb = magmaColormap(normVal)
                pixels[py * BITMAP_SIZE + px] = rgb
            }
        }
        bitmap.setPixels(pixels, 0, BITMAP_SIZE, 0, 0, BITMAP_SIZE, BITMAP_SIZE)

        val idx4k = (numBins * 4000) / (TARGET_SAMPLE_RATE / 2)
        val idx2k = (numBins * 2000) / (TARGET_SAMPLE_RATE / 2)

        var hfEnergySum = 0.0
        var lfEnergySum = 0.0
        var totalEnergy = 0.0
        var temporalVariance = 0.0
        var logSum = 0.0
        var linearSum = 0.0
        var activeFrames = 0

        val cumEnergyByFreq = DoubleArray(numBins)

        for (t in 0 until numFrames) {
            var frameEnergy = 0.0
            for (k in 0 until numBins) {
                val m = spectrogram[t][k].toDouble()
                val p = m * m
                cumEnergyByFreq[k] += p
                frameEnergy += p

                if (k >= idx4k) hfEnergySum += m
                if (k < idx2k) lfEnergySum += m

                linearSum += m + 1e-6
                logSum += ln(m + 1e-6)

                if (t > 0) {
                    val prevM = spectrogram[t - 1][k].toDouble()
                    temporalVariance += (m - prevM) * (m - prevM)
                }
            }
            totalEnergy += frameEnergy
            if (frameEnergy > 0.05) activeFrames++
        }

        val totalElements = (numFrames * numBins).toDouble()
        val geomMean = exp(logSum / totalElements)
        val arithMean = linearSum / totalElements
        val flatness = (geomMean / maxOf(1e-6, arithMean)).toFloat()
        val hfRatio = (hfEnergySum / maxOf(1.0, lfEnergySum)).toFloat()
        val flux = (temporalVariance / totalElements).toFloat()

        var rolloffBin = 0
        var accum = 0.0
        val targetRolloff = totalEnergy * 0.85
        for (k in 0 until numBins) {
            accum += cumEnergyByFreq[k]
            if (accum >= targetRolloff) {
                rolloffBin = k
                break
            }
        }
        val rolloffFreq = (rolloffBin.toFloat() / numBins) * 8000f
        val isVoice = (activeFrames.toFloat() / numFrames) >= 0.20f && lfEnergySum > 10.0

        return AudioAnalysisResult(
            spectrogramBitmap = bitmap,
            isVoiceDetected = isVoice,
            hfToLfRatio = hfRatio,
            spectralFlux = flux,
            spectralFlatness = flatness,
            rolloffFreq = rolloffFreq,
            dynamicRangeDb = rangeDb
        )
    }

    private fun createMelFilterbank(numMels: Int, fftSize: Int, sampleRate: Int): Array<FloatArray> {
        val numBins = fftSize / 2 + 1
        val melMin = 0.0
        val melMax = 2595.0 * log10(1.0 + (sampleRate / 2.0) / 700.0)

        val melPoints = DoubleArray(numMels + 2)
        val binPoints = IntArray(numMels + 2)

        for (i in 0 until numMels + 2) {
            melPoints[i] = melMin + (melMax - melMin) * i / (numMels + 1)
            val hz = 700.0 * (10.0.pow(melPoints[i] / 2595.0) - 1.0)
            binPoints[i] = floor((fftSize + 1) * hz / sampleRate).toInt().coerceIn(0, numBins - 1)
        }

        val filterbank = Array(numMels) { FloatArray(numBins) }
        for (m in 1..numMels) {
            val left = binPoints[m - 1]
            val center = binPoints[m]
            val right = binPoints[m + 1]

            for (k in left until center) {
                if (center > left) {
                    filterbank[m - 1][k] = (k - left).toFloat() / (center - left)
                }
            }
            for (k in center until right) {
                if (right > center) {
                    filterbank[m - 1][k] = (right - k).toFloat() / (right - center)
                }
            }
        }
        return filterbank
    }

    private fun fftRadix2(real: FloatArray, imag: FloatArray) {
        val n = real.size
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]
                real[i] = real[j]
                real[j] = tempR
                val tempI = imag[i]
                imag[i] = imag[j]
                imag[j] = tempI
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * Math.PI / len
            val wStepR = cos(angle).toFloat()
            val wStepI = sin(angle).toFloat()

            var i = 0
            while (i < n) {
                var wR = 1.0f
                var wI = 0.0f
                for (k in 0 until halfLen) {
                    val pos = i + k
                    val match = pos + halfLen

                    val uR = real[pos]
                    val uI = imag[pos]
                    val vR = real[match] * wR - imag[match] * wI
                    val vI = real[match] * wI + imag[match] * wR

                    real[pos] = uR + vR
                    imag[pos] = uI + vI
                    real[match] = uR - vR
                    imag[match] = uI - vI

                    val nextWR = wR * wStepR - wI * wStepI
                    val nextWI = wR * wStepI + wI * wStepR
                    wR = nextWR
                    wI = nextWI
                }
                i += len
            }
            len *= 2
        }
    }

    private fun magmaColormap(norm: Float): Int {
        val vd = norm.coerceIn(0.0f, 1.0f).toDouble()
        val r = (255 * (1.2 * vd.pow(0.8) - 0.2 * vd.pow(3.0))).toInt().coerceIn(15, 255)
        val g = (255 * (vd.pow(1.8) * 0.9)).toInt().coerceIn(23, 255)
        val b = (255 * (1.0 - (vd - 0.5).pow(2.0) * 1.8)).toInt().coerceIn(42, 255)
        return Color.rgb(r, g, b)
    }

    private fun createFallbackSpectrogram(): Bitmap {
        val bmp = Bitmap.createBitmap(BITMAP_SIZE, BITMAP_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(BITMAP_SIZE * BITMAP_SIZE)
        for (y in 0 until BITMAP_SIZE) {
            val ny = y.toDouble() / BITMAP_SIZE.toDouble()
            for (x in 0 until BITMAP_SIZE) {
                val nx = x.toDouble() / BITMAP_SIZE.toDouble()
                val formant = exp(-((ny - 0.75).pow(2.0)) / 0.02) + exp(-((ny - 0.60).pow(2.0)) / 0.03)
                val noise = (sin(nx * 30.0) * 0.05 + cos(ny * 20.0) * 0.05).coerceIn(0.0, 0.1)
                val total = (formant * 0.4 + noise).toFloat().coerceIn(0.0f, 1.0f)
                pixels[y * BITMAP_SIZE + x] = magmaColormap(total)
            }
        }
        bmp.setPixels(pixels, 0, BITMAP_SIZE, 0, 0, BITMAP_SIZE, BITMAP_SIZE)
        return bmp
    }
}

