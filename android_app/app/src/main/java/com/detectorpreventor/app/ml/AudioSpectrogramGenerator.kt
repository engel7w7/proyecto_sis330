package com.detectorpreventor.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.*

object AudioSpectrogramGenerator {

    private const val TAG = "AudioSpectrogram"
    private const val TARGET_SAMPLE_RATE = 16000
    private const val TARGET_SAMPLES = 48000
    private const val FFT_LONG = 1024
    private const val FFT_SHORT = 256
    private const val HOP_SIZE = 210
    private const val SHORT_OFFSET = 384
    private const val NUM_FRAMES = 224
    private const val PRE_EMPHASIS = 0.97f
    private const val BITMAP_SIZE = 224
    private const val VAD_FRAME_HOP = 160
    private const val VAD_REL_THR_PRIMARY = 0.025f
    private const val VAD_REL_THR_FALLBACK = 0.008f
    private const val VAD_MIN_SPAN_SAMPLES = 3200
    private const val TILE_CROSSFADE = 160

    data class AudioAnalysisResult(
        val spectrogramBitmap: Bitmap,
        val isVoiceDetected: Boolean,
        val hfToLfRatio: Float,
        val spectralFlux: Float,
        val spectralFlatness: Float,
        val rolloffFreq: Float,
        val dynamicRangeDb: Float,
        val pcm48k: FloatArray? = null,
        val pcmFull16k: FloatArray? = null,
        val audioConfidence: Float = 1.0f
    )

    internal data class MultiScaleRgbPixels(
        val pixels: IntArray,
        val isVoiceDetected: Boolean,
        val hfToLfRatio: Float,
        val spectralFlux: Float,
        val spectralFlatness: Float,
        val rolloffFreq: Float,
        val dynamicRangeDb: Float
    )

    fun generateSpectrogram(context: Context, uri: Uri): AudioAnalysisResult {
        return generateSpectrogramIfAudioPresent(context, uri) ?: run {
            Log.w(TAG, "No se obtuvieron muestras de audio validas, generando espectrograma sintetico de contingencia.")
            AudioAnalysisResult(
                spectrogramBitmap = createFallbackSpectrogram(),
                isVoiceDetected = false,
                hfToLfRatio = 0.15f,
                spectralFlux = 0.00001f,
                spectralFlatness = 0.30f,
                rolloffFreq = 1800f,
                dynamicRangeDb = 25f
            )
        }
    }

    fun analyzePcm(audioData: FloatArray): AudioAnalysisResult {
        return computeLinearMultiScaleSpectrogramAndFeatures(audioData)
    }

    internal fun extractActiveTiledPcm48k(audioData: FloatArray): FloatArray {
        val out = FloatArray(TARGET_SAMPLES)
        val n = audioData.size
        if (n == 0) return out
        var maxAbs = 0.0f
        for (sample in audioData) {
            val a = abs(sample)
            if (a > maxAbs) maxAbs = a
        }
        if (maxAbs <= 1e-5f) return out

        var s0 = 0
        var s1 = n
        val nf = n / VAD_FRAME_HOP
        if (nf >= 4) {
            val rms = FloatArray(nf)
            var rmax = 0.0f
            for (f in 0 until nf) {
                var sumSq = 0.0
                val base = f * VAD_FRAME_HOP
                for (j in 0 until VAD_FRAME_HOP) {
                    val v = audioData[base + j].toDouble()
                    sumSq += v * v
                }
                val r = sqrt(sumSq / VAD_FRAME_HOP.toDouble() + 1e-12).toFloat()
                rms[f] = r
                if (r > rmax) rmax = r
            }
            if (rmax > 1e-5f) {
                for (thr in floatArrayOf(VAD_REL_THR_PRIMARY, VAD_REL_THR_FALLBACK)) {
                    val limit = thr * rmax
                    var firstAct = -1
                    var lastAct = -1
                    for (f in 0 until nf) {
                        if (rms[f] > limit) {
                            if (firstAct < 0) firstAct = f
                            lastAct = f
                        }
                    }
                    if (firstAct >= 0) {
                        val cand0 = maxOf(0, (firstAct - 1) * VAD_FRAME_HOP)
                        val cand1 = minOf(n, (lastAct + 2) * VAD_FRAME_HOP)
                        if ((cand1 - cand0) >= VAD_MIN_SPAN_SAMPLES || n < VAD_MIN_SPAN_SAMPLES) {
                            s0 = cand0
                            s1 = cand1
                            break
                        }
                    }
                }
            }
        }

        val end = minOf(s1, s0 + TARGET_SAMPLES)
        val m = end - s0
        if (m <= 0) return out
        if (m >= TARGET_SAMPLES) {
            System.arraycopy(audioData, s0, out, 0, TARGET_SAMPLES)
        } else {
            val cf = minOf(TILE_CROSSFADE, m / 4)
            val step = if (cf > 0) (m - cf) else m
            var pos = 0
            var first = true
            while (pos < TARGET_SAMPLES) {
                val rem = TARGET_SAMPLES - pos
                val copyLen = minOf(m, rem)
                if (first || cf <= 0) {
                    System.arraycopy(audioData, s0, out, pos, copyLen)
                    first = false
                } else {
                    val blend = minOf(cf, copyLen)
                    for (j in 0 until blend) {
                        val w = j.toFloat() / cf.toFloat()
                        out[pos + j] = out[pos + j] * (1.0f - w) + audioData[s0 + j] * w
                    }
                    if (copyLen > blend) {
                        System.arraycopy(audioData, s0 + blend, out, pos + blend, copyLen - blend)
                    }
                }
                if (copyLen < m || step <= 0) {
                    break
                }
                pos += step
            }
        }

        var pk = 0.0f
        for (sample in out) {
            val a = abs(sample)
            if (a > pk) pk = a
        }
        if (pk > 0.001f) {
            val invPk = 1.0f / pk
            for (i in out.indices) {
                out[i] *= invPk
            }
        }
        return out
    }

    internal fun extractMultiWindowTiledPcm48k(audioData: FloatArray, maxWindows: Int = 3): List<FloatArray> {
        val n = audioData.size
        if (n <= 72000 || maxWindows <= 1) {
            return listOf(extractActiveTiledPcm48k(audioData))
        }
        val nf = n / VAD_FRAME_HOP
        var s0 = 0
        var s1 = n
        if (nf >= 4) {
            val rms = FloatArray(nf)
            var rmax = 0.0f
            for (f in 0 until nf) {
                var sumSq = 0.0
                val base = f * VAD_FRAME_HOP
                for (j in 0 until VAD_FRAME_HOP) {
                    val v = audioData[base + j].toDouble()
                    sumSq += v * v
                }
                val r = sqrt(sumSq / VAD_FRAME_HOP.toDouble() + 1e-12).toFloat()
                rms[f] = r
                if (r > rmax) rmax = r
            }
            if (rmax > 1e-5f) {
                for (thr in floatArrayOf(VAD_REL_THR_PRIMARY, VAD_REL_THR_FALLBACK)) {
                    val limit = thr * rmax
                    var firstAct = -1
                    var lastAct = -1
                    for (f in 0 until nf) {
                        if (rms[f] > limit) {
                            if (firstAct < 0) firstAct = f
                            lastAct = f
                        }
                    }
                    if (firstAct >= 0) {
                        val cand0 = maxOf(0, (firstAct - 1) * VAD_FRAME_HOP)
                        val cand1 = minOf(n, (lastAct + 2) * VAD_FRAME_HOP)
                        if ((cand1 - cand0) >= VAD_MIN_SPAN_SAMPLES) {
                            s0 = cand0
                            s1 = cand1
                            break
                        }
                    }
                }
            }
        }
        val activeLen = s1 - s0
        if (activeLen <= 72000) {
            return listOf(extractActiveTiledPcm48k(audioData))
        }
        val windows = mutableListOf<FloatArray>()
        val targetCount = minOf(maxWindows, if (activeLen >= 112000) 3 else 2)
        val maxStart = maxOf(s0, s1 - TARGET_SAMPLES)
        for (wi in 0 until targetCount) {
            val start = if (targetCount == 1) s0 else (s0 + ((maxStart - s0).toLong() * wi) / (targetCount - 1)).toInt()
            val end = minOf(s1, start + TARGET_SAMPLES)
            val slice = audioData.copyOfRange(start, end)
            windows.add(extractActiveTiledPcm48k(slice))
        }
        return windows
    }

    internal fun estimateAudioConfidence(audioData: FloatArray): Float {
        if (audioData.isEmpty()) return 0.40f
        val nf = audioData.size / VAD_FRAME_HOP
        if (nf < 8) return 0.50f
        val rms = FloatArray(nf)
        var maxRms = 0.0f
        for (f in 0 until nf) {
            var sumSq = 0.0
            val base = f * VAD_FRAME_HOP
            for (j in 0 until VAD_FRAME_HOP) {
                val v = audioData[base + j].toDouble()
                sumSq += v * v
            }
            val r = sqrt(sumSq / VAD_FRAME_HOP.toDouble() + 1e-12).toFloat()
            rms[f] = r
            if (r > maxRms) maxRms = r
        }
        if (maxRms <= 1e-4f) return 0.40f
        val sorted = rms.sortedArray()
        val speechRef = sorted[((nf - 1) * 0.90).toInt().coerceIn(0, nf - 1)]
        val noiseFloor = maxOf(1e-6f, sorted[((nf - 1) * 0.15).toInt().coerceIn(0, nf - 1)])
        val snrDb = (20.0 * log10((speechRef / noiseFloor).toDouble().coerceAtLeast(1.0))).toFloat()
        val activeThr = maxRms * VAD_REL_THR_PRIMARY
        val activeSec = (rms.count { it > activeThr } * VAD_FRAME_HOP).toFloat() / TARGET_SAMPLE_RATE.toFloat()
        val snrFactor = when {
            snrDb >= 20.0f -> 1.0f
            snrDb <= 8.0f -> 0.45f
            else -> 0.45f + ((snrDb - 8.0f) / 12.0f) * 0.55f
        }
        val durFactor = when {
            activeSec >= 0.70f -> 1.0f
            activeSec <= 0.25f -> 0.50f
            else -> 0.50f + ((activeSec - 0.25f) / 0.45f) * 0.50f
        }
        return minOf(snrFactor, durFactor).coerceIn(0.40f, 1.0f)
    }

    internal fun buildPcm48kFromWavBytes(bytes: ByteArray): FloatArray? {
        val mono16k = parseWavPcmToMono16k(bytes) ?: return null
        if (mono16k.isEmpty()) return null
        return extractActiveTiledPcm48k(mono16k)
    }

    fun generateSpectrogramIfAudioPresent(context: Context, uri: Uri): AudioAnalysisResult? {
        var pcmData: FloatArray? = null
        try {
            pcmData = decodeAudioUriToPcm(context, uri)
        } catch (e: Exception) {
            Log.w(TAG, "MediaCodec decoder fallo, intentando lectura WAV directa: ${e.message}")
        }

        if (pcmData == null || pcmData.isEmpty()) {
            pcmData = readRawPcmFromStream(context, uri)
        }

        if (pcmData.isEmpty()) {
            Log.i(TAG, "Pista de audio ausente o vacia para URI: $uri")
            return null
        }

        return computeLinearMultiScaleSpectrogramAndFeatures(pcmData)
    }

    private fun resolveAudioTempExtension(context: Context, uri: Uri): String {
        val allowed = setOf("opus", "ogg", "m4a", "mp3", "aac", "amr", "wav", "flac", "3ga", "mp4", "webm")
        var displayName: String? = null
        if (uri.scheme == "content") {
            try {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (idx >= 0) displayName = cursor.getString(idx)
                    }
                }
            } catch (_: Exception) {
            }
        }
        val candidateName = displayName ?: uri.lastPathSegment ?: ""
        val extFromName = candidateName.substringAfterLast('.', "").substringBefore('?').lowercase(Locale.ROOT).trim()
        if (extFromName in allowed) return ".$extFromName"

        val mime = try {
            context.contentResolver.getType(uri)?.lowercase(Locale.ROOT)
        } catch (_: Exception) {
            null
        }
        return when {
            mime == null -> ".audio"
            mime.contains("opus") -> ".opus"
            mime.contains("ogg") -> ".ogg"
            mime.contains("mp4") || mime.contains("m4a") || mime.contains("aac") -> ".m4a"
            mime.contains("mpeg") || mime.contains("mp3") -> ".mp3"
            mime.contains("amr") -> ".amr"
            mime.contains("wav") -> ".wav"
            mime.contains("flac") -> ".flac"
            else -> ".audio"
        }
    }

    private fun decodeAudioUriToPcm(context: Context, uri: Uri): FloatArray? {
        val tempExt = resolveAudioTempExtension(context, uri)
        val tempFile = File.createTempFile("temp_decode_", tempExt, context.cacheDir)
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            } ?: return null

            val localExtractor = MediaExtractor()
            extractor = localExtractor
            localExtractor.setDataSource(tempFile.absolutePath)

            var audioTrackIndex = -1
            var inputFormat: MediaFormat? = null
            for (i in 0 until localExtractor.trackCount) {
                val format = localExtractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    inputFormat = format
                    break
                }
            }

            if (audioTrackIndex == -1 || inputFormat == null) {
                return null
            }

            localExtractor.selectTrack(audioTrackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME) ?: ""
            val localDecoder = MediaCodec.createDecoderByType(mime)
            decoder = localDecoder
            localDecoder.configure(inputFormat, null, null, 0)
            localDecoder.start()

            var srcSampleRate = if (inputFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            } else {
                44100
            }
            var channelCount = maxOf(
                1,
                if (inputFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                    inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                } else {
                    1
                }
            )
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
            try {
                val outFmt = localDecoder.outputFormat
                if (outFmt.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                    pcmEncoding = outFmt.getInteger(MediaFormat.KEY_PCM_ENCODING)
                }
                if (outFmt.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                    srcSampleRate = outFmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                }
                if (outFmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                    channelCount = maxOf(1, outFmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                }
            } catch (_: Throwable) {
            }

            val bufferInfo = MediaCodec.BufferInfo()
            val rawPcmSamples = mutableListOf<Float>()
            var maxMonoSamplesNeeded = maxOf(TARGET_SAMPLES * 4, (srcSampleRate * 12.0).toInt())
            var sawInputEos = false
            var sawOutputEos = false
            var noProgressLoops = 0
            val maxNoProgressLoops = 250
            val timeoutUs = 5000L

            while (!sawOutputEos && rawPcmSamples.size < maxMonoSamplesNeeded && noProgressLoops < maxNoProgressLoops) {
                var madeProgress = false

                if (!sawInputEos) {
                    val inIndex = localDecoder.dequeueInputBuffer(timeoutUs)
                    if (inIndex >= 0) {
                        madeProgress = true
                        val inputBuffer = localDecoder.getInputBuffer(inIndex)
                        if (inputBuffer != null) {
                            val sampleSize = localExtractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                localDecoder.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                sawInputEos = true
                            } else {
                                localDecoder.queueInputBuffer(inIndex, 0, sampleSize, localExtractor.sampleTime, 0)
                                localExtractor.advance()
                            }
                        } else {
                            localDecoder.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEos = true
                        }
                    }
                }

                var outIndex = localDecoder.dequeueOutputBuffer(bufferInfo, timeoutUs)
                while (outIndex >= 0) {
                    madeProgress = true
                    val outputBuffer = localDecoder.getOutputBuffer(outIndex)
                    if (outputBuffer != null && bufferInfo.size > 0) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        outputBuffer.order(ByteOrder.LITTLE_ENDIAN)

                        if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            val floatBuffer = outputBuffer.asFloatBuffer()
                            while (floatBuffer.hasRemaining()) {
                                if (channelCount > 1) {
                                    var sum = 0f
                                    var count = 0
                                    for (c in 0 until channelCount) {
                                        if (floatBuffer.hasRemaining()) {
                                            sum += floatBuffer.get().coerceIn(-1.0f, 1.0f)
                                            count++
                                        }
                                    }
                                    if (count > 0) {
                                        rawPcmSamples.add(sum / count)
                                    }
                                } else {
                                    rawPcmSamples.add(floatBuffer.get().coerceIn(-1.0f, 1.0f))
                                }
                            }
                        } else {
                            val shortBuffer = outputBuffer.asShortBuffer()
                            while (shortBuffer.hasRemaining()) {
                                if (channelCount > 1) {
                                    var sum = 0f
                                    var count = 0
                                    for (c in 0 until channelCount) {
                                        if (shortBuffer.hasRemaining()) {
                                            sum += shortBuffer.get() / 32768.0f
                                            count++
                                        }
                                    }
                                    if (count > 0) {
                                        rawPcmSamples.add(sum / count)
                                    }
                                } else {
                                    rawPcmSamples.add(shortBuffer.get() / 32768.0f)
                                }
                            }
                        }
                    }
                    localDecoder.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        sawOutputEos = true
                        break
                    }
                    if (rawPcmSamples.size >= maxMonoSamplesNeeded) {
                        break
                    }
                    outIndex = localDecoder.dequeueOutputBuffer(bufferInfo, 0)
                }

                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    madeProgress = true
                    try {
                        val outFmt = localDecoder.outputFormat
                        if (outFmt.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            pcmEncoding = outFmt.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        }
                        if (outFmt.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                            srcSampleRate = outFmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            maxMonoSamplesNeeded = maxOf(TARGET_SAMPLES * 4, (srcSampleRate * 12.0).toInt())
                        }
                        if (outFmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                            channelCount = maxOf(1, outFmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
                        }
                    } catch (_: Throwable) {
                    }
                } else if (outIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                    madeProgress = true
                }

                noProgressLoops = if (madeProgress) 0 else (noProgressLoops + 1)
            }

            if (rawPcmSamples.isEmpty()) return null

            val originalArray = rawPcmSamples.toFloatArray()
            return if (srcSampleRate != TARGET_SAMPLE_RATE) {
                resampleAudio(originalArray, srcSampleRate, TARGET_SAMPLE_RATE)
            } else {
                originalArray
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error en decodificador MediaCodec: ${e.message}")
            return null
        } finally {
            try {
                decoder?.stop()
            } catch (_: Throwable) {
            }
            try {
                decoder?.release()
            } catch (_: Throwable) {
            }
            try {
                extractor?.release()
            } catch (_: Throwable) {
            }
            try {
                if (tempFile.exists()) {
                    tempFile.delete()
                }
            } catch (_: Throwable) {
            }
        }
    }

    internal fun parseWavPcmToMono16k(bytes: ByteArray): FloatArray? {
        if (bytes.size < 44) return null
        val isRiff = bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() &&
            bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte()
        val isWave = bytes[8] == 'W'.code.toByte() && bytes[9] == 'A'.code.toByte() &&
            bytes[10] == 'V'.code.toByte() && bytes[11] == 'E'.code.toByte()
        if (!isRiff || !isWave) return null

        var offset = 12
        var channels = 1
        var sampleRate = TARGET_SAMPLE_RATE
        var bitsPerSample = 16
        var audioFormat = 1
        var dataOffset = -1
        var dataSize = 0

        while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val chunkSize = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            val safeSize = chunkSize.coerceAtLeast(0)
            val payloadStart = offset + 8
            if (chunkId == "fmt " && payloadStart + 16 <= bytes.size) {
                val bb = ByteBuffer.wrap(bytes, payloadStart, 16).order(ByteOrder.LITTLE_ENDIAN)
                audioFormat = bb.short.toInt() and 0xFFFF
                channels = maxOf(1, bb.short.toInt() and 0xFFFF)
                sampleRate = maxOf(8000, bb.int)
                bb.int
                bb.short
                bitsPerSample = bb.short.toInt() and 0xFFFF
            } else if (chunkId == "data" && payloadStart <= bytes.size) {
                dataOffset = payloadStart
                dataSize = minOf(safeSize, bytes.size - payloadStart)
                break
            }
            val paddedSize = safeSize + (safeSize and 1)
            val next = payloadStart + paddedSize
            if (next <= offset) break
            offset = next
        }

        if (dataOffset < 0 || dataSize <= 0 || audioFormat != 1 || bitsPerSample != 16) {
            return null
        }

        val maxMonoNeeded = maxOf(TARGET_SAMPLES * 3, (sampleRate * 3.5).toInt())
        val totalFrames = dataSize / (2 * channels)
        val framesToRead = minOf(totalFrames, maxMonoNeeded)
        if (framesToRead <= 0) return null

        val bb = ByteBuffer.wrap(bytes, dataOffset, framesToRead * 2 * channels).order(ByteOrder.LITTLE_ENDIAN)
        val mono = FloatArray(framesToRead)
        for (i in 0 until framesToRead) {
            var sum = 0f
            for (c in 0 until channels) {
                sum += bb.short / 32768.0f
            }
            mono[i] = sum / channels
        }
        return if (sampleRate != TARGET_SAMPLE_RATE) {
            resampleAudio(mono, sampleRate, TARGET_SAMPLE_RATE)
        } else {
            mono
        }
    }

    private fun readRawPcmFromStream(context: Context, uri: Uri): FloatArray {
        return try {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return FloatArray(0)
            parseWavPcmToMono16k(bytes) ?: FloatArray(0)
        } catch (e: Exception) {
            FloatArray(0)
        }
    }

    internal fun resampleAudio(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate <= 0 || dstRate <= 0 || input.isEmpty() || srcRate == dstRate) return input
        val ratio = dstRate.toDouble() / srcRate.toDouble()
        val outLength = (input.size * ratio).toInt()
        if (outLength <= 0) return FloatArray(0)
        val output = FloatArray(outLength)

        if (dstRate >= srcRate) {
            for (i in 0 until outLength) {
                val srcIndex = i / ratio
                val index0 = srcIndex.toInt().coerceIn(0, input.size - 1)
                val index1 = (index0 + 1).coerceIn(0, input.size - 1)
                val frac = (srcIndex - index0).toFloat()
                output[i] = input[index0] * (1.0f - frac) + input[index1] * frac
            }
            return output
        }

        val fc = 0.45 * ratio
        val radius = ceil(8.0 / ratio).toInt().coerceIn(8, 48)
        val n = input.size
        for (i in 0 until outLength) {
            val srcCenter = i / ratio
            val base = floor(srcCenter).toInt()
            val kMin = maxOf(0, base - radius + 1)
            val kMax = minOf(n - 1, base + radius)
            var acc = 0.0
            var wSum = 0.0
            for (k in kMin..kMax) {
                val d = k - srcCenter
                val u = abs(d) / radius.toDouble()
                if (u < 1.0) {
                    val win = 0.42 + 0.50 * cos(PI * u) + 0.08 * cos(2.0 * PI * u)
                    val sinc = if (abs(d) < 1e-6) {
                        2.0 * fc
                    } else {
                        sin(2.0 * PI * fc * d) / (PI * d)
                    }
                    val w = win * sinc
                    acc += input[k] * w
                    wSum += w
                }
            }
            output[i] = if (wSum > 1e-9) {
                (acc / wSum).toFloat()
            } else {
                input[srcCenter.toInt().coerceIn(0, n - 1)]
            }
        }
        return output
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)

    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private class MelPoolMap(fftSize: Int) {
        val isPool = BooleanArray(BITMAP_SIZE)
        val b0 = IntArray(BITMAP_SIZE)
        val b1 = IntArray(BITMAP_SIZE)
        val k0 = IntArray(BITMAP_SIZE)
        val k1 = IntArray(BITMAP_SIZE)
        val frac = FloatArray(BITMAP_SIZE)

        init {
            val nBins = fftSize / 2 + 1
            val nyq = TARGET_SAMPLE_RATE * 0.5
            val melMax = hzToMel(nyq)
            val halfFft = (fftSize / 2).toDouble()
            for (py in 0 until BITMAP_SIZE) {
                val idxNorm = (BITMAP_SIZE - 1 - py).toDouble()
                val denom = (BITMAP_SIZE - 1).toDouble()
                val mCenter = (idxNorm / denom) * melMax
                val mLow = maxOf(0.0, ((idxNorm - 0.5) / denom) * melMax)
                val mHigh = minOf(melMax, ((idxNorm + 0.5) / denom) * melMax)
                val bCenter = (melToHz(mCenter) / nyq) * halfFft
                val bLow = (melToHz(mLow) / nyq) * halfFft
                val bHigh = (melToHz(mHigh) / nyq) * halfFft
                val i0 = ceil(bLow).toInt()
                val i1 = floor(bHigh).toInt()
                if (i1 >= i0) {
                    isPool[py] = true
                    b0[py] = maxOf(0, i0)
                    b1[py] = minOf(nBins - 1, i1)
                } else {
                    isPool[py] = false
                    val kk0 = maxOf(0, minOf(nBins - 2, floor(bCenter).toInt()))
                    k0[py] = kk0
                    k1[py] = kk0 + 1
                    frac[py] = (bCenter - kk0).coerceIn(0.0, 1.0).toFloat()
                }
            }
        }

        fun bandMag(mag: FloatArray, py: Int): Float {
            return if (isPool[py]) {
                val s = b0[py]
                val e = b1[py]
                var acc = 0.0f
                for (b in s..e) acc += mag[b]
                acc / (e - s + 1).toFloat()
            } else {
                val f = frac[py]
                (1.0f - f) * mag[k0[py]] + f * mag[k1[py]]
            }
        }
    }

    private fun computeLinearMultiScaleSpectrogramAndFeatures(audioData: FloatArray): AudioAnalysisResult {
        val safePcm = extractActiveTiledPcm48k(audioData)
        val computed = computeMultiScaleRgbPixels(safePcm, applyVadTile = false)
        val bitmap = Bitmap.createBitmap(BITMAP_SIZE, BITMAP_SIZE, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(computed.pixels, 0, BITMAP_SIZE, 0, 0, BITMAP_SIZE, BITMAP_SIZE)
        return AudioAnalysisResult(
            spectrogramBitmap = bitmap,
            isVoiceDetected = computed.isVoiceDetected,
            hfToLfRatio = computed.hfToLfRatio,
            spectralFlux = computed.spectralFlux,
            spectralFlatness = computed.spectralFlatness,
            rolloffFreq = computed.rolloffFreq,
            dynamicRangeDb = computed.dynamicRangeDb,
            pcm48k = safePcm,
            pcmFull16k = if (audioData.size > 72000) audioData else null,
            audioConfidence = estimateAudioConfidence(audioData)
        )
    }

    internal fun computeMultiScaleRgbPixels(audioData: FloatArray, applyVadTile: Boolean = true): MultiScaleRgbPixels {
        val safeAudio = if (applyVadTile) {
            extractActiveTiledPcm48k(audioData)
        } else {
            val copy = FloatArray(TARGET_SAMPLES)
            val copyLen = minOf(audioData.size, TARGET_SAMPLES)
            if (copyLen > 0) {
                System.arraycopy(audioData, 0, copy, 0, copyLen)
            }
            copy
        }

        var maxVal = 0.0f
        for (sample in safeAudio) {
            val a = abs(sample)
            if (a > maxVal) maxVal = a
        }
        val isActiveSignal = maxVal > 0.001f
        if (isActiveSignal) {
            val invMax = 1.0f / maxVal
            for (i in safeAudio.indices) {
                safeAudio[i] *= invMax
            }
        }

        val win1024 = FloatArray(FFT_LONG) { i ->
            (0.5 * (1.0 - cos(2.0 * Math.PI * i / (FFT_LONG - 1)))).toFloat()
        }
        val win256 = FloatArray(FFT_SHORT) { i ->
            (0.5 * (1.0 - cos(2.0 * Math.PI * i / (FFT_SHORT - 1)))).toFloat()
        }

        val numBins1 = FFT_LONG / 2 + 1
        val melMap1024 = MelPoolMap(FFT_LONG)
        val melMap256 = MelPoolMap(FFT_SHORT)

        val db1 = FloatArray(BITMAP_SIZE * NUM_FRAMES)
        var maxDb1 = -Float.MAX_VALUE
        var minDb1 = Float.MAX_VALUE

        val spec1 = Array(NUM_FRAMES) { FloatArray(numBins1) }
        val real1 = DoubleArray(FFT_LONG)
        val imag1 = DoubleArray(FFT_LONG)
        val mag1 = FloatArray(numBins1)

        for (t in 0 until NUM_FRAMES) {
            val offset = t * HOP_SIZE
            for (i in 0 until FFT_LONG) {
                real1[i] = (safeAudio[offset + i] * win1024[i]).toDouble()
                imag1[i] = 0.0
            }
            fftRadix2(real1, imag1)
            for (k in 0 until numBins1) {
                val re = real1[k].toFloat()
                val im = imag1[k].toFloat()
                val m = sqrt(re * re + im * im + 1e-12f)
                mag1[k] = m
                spec1[t][k] = m
            }
            for (py in 0 until BITMAP_SIZE) {
                val band = melMap1024.bandMag(mag1, py)
                val vDb = (20.0 * log10(maxOf(1e-6f, band).toDouble())).toFloat()
                db1[py * NUM_FRAMES + t] = vDb
                if (vDb > maxDb1) maxDb1 = vDb
                if (vDb < minDb1) minDb1 = vDb
            }
        }

        val floor1 = maxOf(minDb1, maxDb1 - 80.0f)
        val denom1 = maxOf(1.0f, maxDb1 - floor1)

        val ch1 = FloatArray(BITMAP_SIZE * NUM_FRAMES)
        for (py in 0 until BITMAP_SIZE) {
            val rowBase = py * NUM_FRAMES
            var sum = 0.0
            for (t in 0 until NUM_FRAMES) {
                sum += db1[rowBase + t].toDouble()
            }
            val mu = sum / NUM_FRAMES.toDouble()
            var varSum = 0.0
            for (t in 0 until NUM_FRAMES) {
                val d = db1[rowBase + t].toDouble() - mu
                varSum += d * d
            }
            val stdF = maxOf(1.5, sqrt(varSum / NUM_FRAMES.toDouble()))
            for (t in 0 until NUM_FRAMES) {
                val z = (db1[rowBase + t].toDouble() - mu) / stdF
                ch1[rowBase + t] = (0.5 + z / 6.0).toFloat().coerceIn(0.0f, 1.0f)
            }
        }

        val xPe = FloatArray(TARGET_SAMPLES)
        xPe[0] = safeAudio[0]
        var pMax = abs(xPe[0])
        for (i in 1 until TARGET_SAMPLES) {
            val v = safeAudio[i] - PRE_EMPHASIS * safeAudio[i - 1]
            xPe[i] = v
            val a = abs(v)
            if (a > pMax) pMax = a
        }
        if (isActiveSignal) {
            val invPMax = 1.0f / maxOf(1e-6f, pMax)
            for (i in xPe.indices) {
                xPe[i] *= invPMax
            }
        }

        val numBins2 = FFT_SHORT / 2 + 1
        val db2 = FloatArray(BITMAP_SIZE * NUM_FRAMES)
        var maxDb2 = -Float.MAX_VALUE
        var minDb2 = Float.MAX_VALUE
        val real2 = DoubleArray(FFT_SHORT)
        val imag2 = DoubleArray(FFT_SHORT)
        val mag2 = FloatArray(numBins2)

        for (t in 0 until NUM_FRAMES) {
            val offset = SHORT_OFFSET + t * HOP_SIZE
            for (i in 0 until FFT_SHORT) {
                real2[i] = (xPe[offset + i] * win256[i]).toDouble()
                imag2[i] = 0.0
            }
            fftRadix2(real2, imag2)
            for (k in 0 until numBins2) {
                val re = real2[k].toFloat()
                val im = imag2[k].toFloat()
                mag2[k] = sqrt(re * re + im * im + 1e-12f)
            }
            for (py in 0 until BITMAP_SIZE) {
                val band = melMap256.bandMag(mag2, py)
                val vDb = (20.0 * log10(maxOf(1e-6f, band).toDouble())).toFloat()
                db2[py * NUM_FRAMES + t] = vDb
                if (vDb > maxDb2) maxDb2 = vDb
                if (vDb < minDb2) minDb2 = vDb
            }
        }

        val floor2 = maxOf(minDb2, maxDb2 - 80.0f)
        val denom2 = maxOf(1.0f, maxDb2 - floor2)

        val pixels = IntArray(BITMAP_SIZE * BITMAP_SIZE)
        for (i in pixels.indices) {
            val c0 = ((db1[i] - floor1) / denom1).coerceIn(0.0f, 1.0f)
            val c1 = ch1[i]
            val c2 = ((db2[i] - floor2) / denom2).coerceIn(0.0f, 1.0f)
            val r = (c0 * 255.0f).roundToInt().coerceIn(0, 255)
            val g = (c1 * 255.0f).roundToInt().coerceIn(0, 255)
            val b = (c2 * 255.0f).roundToInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val idx4k = (numBins1 * 4000) / (TARGET_SAMPLE_RATE / 2)
        val idx2k = (numBins1 * 2000) / (TARGET_SAMPLE_RATE / 2)
        var hfEnergySum = 0.0
        var lfEnergySum = 0.0
        var totalEnergy = 0.0
        var temporalVariance = 0.0
        var logSum = 0.0
        var linearSum = 0.0
        var activeFrames = 0
        val cumEnergyByFreq = DoubleArray(numBins1)

        for (t in 0 until NUM_FRAMES) {
            var frameEnergy = 0.0
            for (k in 0 until numBins1) {
                val m = spec1[t][k].toDouble()
                val p = m * m
                cumEnergyByFreq[k] += p
                frameEnergy += p
                if (k >= idx4k) hfEnergySum += m
                if (k < idx2k) lfEnergySum += m
                linearSum += m + 1e-6
                logSum += ln(m + 1e-6)
                if (t > 0) {
                    val prevM = spec1[t - 1][k].toDouble()
                    temporalVariance += (m - prevM) * (m - prevM)
                }
            }
            totalEnergy += frameEnergy
            if (frameEnergy > 0.05) activeFrames++
        }

        val totalElements = (NUM_FRAMES * numBins1).toDouble()
        val geomMean = exp(logSum / totalElements)
        val arithMean = linearSum / totalElements
        val flatness = (geomMean / maxOf(1e-6, arithMean)).toFloat()
        val hfRatio = (hfEnergySum / maxOf(1.0, lfEnergySum)).toFloat()
        val flux = (temporalVariance / totalElements).toFloat()

        var rolloffBin = 0
        var accum = 0.0
        val targetRolloff = totalEnergy * 0.85
        for (k in 0 until numBins1) {
            accum += cumEnergyByFreq[k]
            if (accum >= targetRolloff) {
                rolloffBin = k
                break
            }
        }
        val rolloffFreq = (rolloffBin.toFloat() / numBins1) * 8000f
        val isVoice = (activeFrames.toFloat() / NUM_FRAMES) >= 0.20f && lfEnergySum > 10.0

        return MultiScaleRgbPixels(
            pixels = pixels,
            isVoiceDetected = isVoice,
            hfToLfRatio = hfRatio,
            spectralFlux = flux,
            spectralFlatness = flatness,
            rolloffFreq = rolloffFreq,
            dynamicRangeDb = denom1
        )
    }

    private fun fftRadix2(real: DoubleArray, imag: DoubleArray) {
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
            val wStepR = cos(angle)
            val wStepI = sin(angle)

            var i = 0
            while (i < n) {
                var wR = 1.0
                var wI = 0.0
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

    private fun createFallbackSpectrogram(): Bitmap {
        val bmp = Bitmap.createBitmap(BITMAP_SIZE, BITMAP_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(BITMAP_SIZE * BITMAP_SIZE)
        for (y in 0 until BITMAP_SIZE) {
            val ny = y.toDouble() / BITMAP_SIZE.toDouble()
            for (x in 0 until BITMAP_SIZE) {
                val nx = x.toDouble() / BITMAP_SIZE.toDouble()
                val formant = exp(-((ny - 0.75).pow(2.0)) / 0.02) + exp(-((ny - 0.60).pow(2.0)) / 0.03)
                val noise = (sin(nx * 30.0) * 0.05 + cos(ny * 20.0) * 0.05).coerceIn(0.0, 0.1)
                val base = (formant * 0.4 + noise).toFloat().coerceIn(0.0f, 1.0f)
                val r = (base * 255.0f).roundToInt().coerceIn(0, 255)
                val g = 128
                val b = ((base * 0.6f) * 255.0f).roundToInt().coerceIn(0, 255)
                pixels[y * BITMAP_SIZE + x] = Color.rgb(r, g, b)
            }
        }
        bmp.setPixels(pixels, 0, BITMAP_SIZE, 0, 0, BITMAP_SIZE, BITMAP_SIZE)
        return bmp
    }
}

