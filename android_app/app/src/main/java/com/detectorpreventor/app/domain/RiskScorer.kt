package com.detectorpreventor.app.domain

enum class RiskLevel {
    BAJO,
    MEDIO,
    ALTO
}

data class FusionResult(
    val globalRiskPercentage: Float,
    val audioFraudProb: Float,
    val visionFraudProb: Float,
    val riskLevel: RiskLevel,
    val weightAudio: Float,
    val weightVision: Float,
    val diagnosticSummary: String,
    val wasFaceDiscarded: Boolean = false
)

object RiskScorer {

    private const val DEFAULT_WEIGHT_AUDIO = 0.55f
    private const val DEFAULT_WEIGHT_VISION = 0.45f

    fun calculateGlobalRisk(
        audioProb: Float?,
        visionProb: Float?,
        customWeightAudio: Float = DEFAULT_WEIGHT_AUDIO,
        customWeightVision: Float = DEFAULT_WEIGHT_VISION,
        isFaceDetected: Boolean = true,
        audioConfidence: Float = 1.0f,
        visionSharpness: Float = 1.0f
    ): FusionResult {
        if (!isFaceDetected && audioProb == null) {
            val summary = "• Descarte Automático por Ausencia Facial: No se reconoció ningún rostro humano en la imagen analizada.\n• La muestra se clasifica como SEGURA / SIN RIESGO (0.0%)."
            return FusionResult(
                globalRiskPercentage = 0.0f,
                audioFraudProb = 0.0f,
                visionFraudProb = 0.0f,
                riskLevel = RiskLevel.BAJO,
                weightAudio = 0.0f,
                weightVision = 1.0f,
                diagnosticSummary = summary,
                wasFaceDiscarded = true
            )
        }

        val effectiveVisionProb = if (!isFaceDetected) 0.0f else visionProb

        val (finalAudioProb, finalVisionProb, wAudio, wVision) = when {
            audioProb != null && effectiveVisionProb != null -> {
                val confA = audioConfidence.coerceIn(0.35f, 1.0f)
                val confV = visionSharpness.coerceIn(0.35f, 1.0f)
                val effWeightAudio = (customWeightAudio * confA).coerceAtLeast(0.05f)
                val effWeightVision = (customWeightVision * confV).coerceAtLeast(0.05f)
                val totalWeight = effWeightAudio + effWeightVision
                val normAudioWeight = effWeightAudio / totalWeight
                val normVisionWeight = effWeightVision / totalWeight
                Tuple4(audioProb, effectiveVisionProb, normAudioWeight, normVisionWeight)
            }
            audioProb != null -> {
                Tuple4(audioProb, 0f, 1.0f, 0.0f)
            }
            effectiveVisionProb != null -> {
                Tuple4(0f, effectiveVisionProb, 0.0f, 1.0f)
            }
            else -> {
                Tuple4(0f, 0f, 0.5f, 0.5f)
            }
        }

        val weightedScore = (wAudio * finalAudioProb) + (wVision * finalVisionProb)
        val finalScore = if (audioProb != null && effectiveVisionProb != null) {
            val maxThreat = maxOf(finalAudioProb, finalVisionProb)
            val minThreat = minOf(finalAudioProb, finalVisionProb)
            val threatDegraded = (finalAudioProb >= maxThreat && audioConfidence < 0.70f) ||
                (finalVisionProb >= maxThreat && visionSharpness < 0.70f)
            when {
                maxThreat >= 0.75f && threatDegraded && minThreat <= 0.15f ->
                    maxOf(weightedScore, maxThreat * 0.68f)
                maxThreat >= 0.75f ->
                    maxOf(weightedScore, maxThreat * 0.95f)
                maxThreat >= 0.56f && minThreat >= 0.56f -> {
                    val noisyOr = 1.0f - (1.0f - finalAudioProb) * (1.0f - finalVisionProb)
                    val synergy = 0.55f * weightedScore + 0.45f * noisyOr
                    maxOf(weightedScore, synergy.coerceAtLeast(0.71f))
                }
                else -> weightedScore
            }
        } else {
            weightedScore
        }
        val globalPercentage = (finalScore * 100.0f).coerceIn(0.0f, 100.0f)

        val riskLevel = when {
            globalPercentage < 30.0f -> RiskLevel.BAJO
            globalPercentage < 70.0f -> RiskLevel.MEDIO
            else -> RiskLevel.ALTO
        }

        val summary = generateDiagnosticText(
            globalPercentage,
            finalAudioProb,
            finalVisionProb,
            wAudio,
            wVision,
            riskLevel,
            isFaceDetected,
            audioConfidence,
            visionSharpness
        )

        return FusionResult(
            globalRiskPercentage = globalPercentage,
            audioFraudProb = finalAudioProb,
            visionFraudProb = finalVisionProb,
            riskLevel = riskLevel,
            weightAudio = wAudio,
            weightVision = wVision,
            diagnosticSummary = summary,
            wasFaceDiscarded = !isFaceDetected
        )
    }

    fun calculateVideoRisk(
        audioProb: Float?,
        frameProbs: List<Float>,
        isFaceDetected: Boolean = true,
        frameQualities: List<Float> = emptyList(),
        audioConfidence: Float = 1.0f
    ): FusionResult {
        val hasValidQualities = frameQualities.size == frameProbs.size && frameProbs.isNotEmpty()
        val avgVisionProb = if (frameProbs.isEmpty()) {
            0.0f
        } else if (hasValidQualities) {
            var num = 0.0f
            var den = 0.0f
            for (i in frameProbs.indices) {
                val q = frameQualities[i].coerceIn(0.25f, 1.0f)
                num += frameProbs[i] * q
                den += q
            }
            if (den > 1e-4f) num / den else frameProbs.average().toFloat()
        } else {
            frameProbs.average().toFloat()
        }
        val avgSharpness = if (hasValidQualities) {
            frameQualities.average().toFloat().coerceIn(0.35f, 1.0f)
        } else {
            1.0f
        }

        val alteredFramesCount = frameProbs.count { it >= 0.5f }
        val sortedDesc = frameProbs.sortedDescending()
        val top3Mean = if (sortedDesc.size >= 3) (sortedDesc[0] + sortedDesc[1] + sortedDesc[2]) / 3.0f else avgVisionProb
        val top2Mean = if (sortedDesc.size >= 2) (sortedDesc[0] + sortedDesc[1]) / 2.0f else avgVisionProb

        val isMajorityAltered = frameProbs.isNotEmpty() &&
            (alteredFramesCount >= 3 || (frameProbs.size in 1..4 && alteredFramesCount.toFloat() / frameProbs.size >= 0.6f)) &&
            (avgVisionProb >= 0.52f || (top3Mean >= 0.60f && avgVisionProb >= 0.44f))
        val isAudioCloned = (audioProb ?: 0f) >= 0.62f
        val isBimodalAgreement = (audioProb ?: 0f) >= 0.56f && alteredFramesCount >= 2 && top2Mean >= 0.56f

        val baseFusion = calculateGlobalRisk(
            audioProb = audioProb,
            visionProb = avgVisionProb,
            customWeightAudio = DEFAULT_WEIGHT_AUDIO,
            customWeightVision = DEFAULT_WEIGHT_VISION,
            isFaceDetected = isFaceDetected,
            audioConfidence = audioConfidence,
            visionSharpness = avgSharpness
        )

        return if (isAudioCloned || isMajorityAltered || isBimodalAgreement) {
            val elevatedScore = maxOf(baseFusion.globalRiskPercentage, 85.0f)
            val reason = when {
                isAudioCloned && isMajorityAltered ->
                    "Pista vocal clonada y mayoria de fotogramas alterados ($alteredFramesCount/${frameProbs.size})"
                isAudioCloned ->
                    "Pista de audio clasificada matematicamente como clonada (CNN Espectral)"
                isMajorityAltered ->
                    "Deteccion sostenida de alteraciones faciales: Regla 3/5 ($alteredFramesCount/${frameProbs.size} frames manipulados)"
                else ->
                    "Consenso bimodal audio-video detectado: Regla 3/5 ($alteredFramesCount/${frameProbs.size} frames y audio sospechoso)"
            }
            val cleanSummary = baseFusion.diagnosticSummary.replaceFirst(
                Regex("^Veredicto Final: [^\\n]+"),
                String.format(java.util.Locale.US, "Veredicto Final: %.1f%% de Riesgo (ALTO).", elevatedScore)
            )
            baseFusion.copy(
                globalRiskPercentage = elevatedScore,
                riskLevel = RiskLevel.ALTO,
                diagnosticSummary = cleanSummary + "\n• Alerta Critica (Regla 3/5 Video): $reason."
            )
        } else {
            baseFusion
        }
    }

    private fun generateDiagnosticText(
        scorePercent: Float,
        pAudio: Float,
        pVision: Float,
        wA: Float,
        wV: Float,
        level: RiskLevel,
        isFaceDetected: Boolean,
        audioConfidence: Float = 1.0f,
        visionSharpness: Float = 1.0f
    ): String {
        val sb = StringBuilder()
        sb.append(String.format("Veredicto Final: %.1f%% de Riesgo (%s).\n", scorePercent, level.name))
        
        if (!isFaceDetected) {
            sb.append("• Advertencia: Sin detección facial en el canal visual (neutralizado a 0.0% por descarte automático).\n")
        }
        if (audioConfidence < 0.70f && wA > 0f) {
            sb.append("• Señal de audio con ruido elevado o corta duración activa: peso espectral atenuado.\n")
        }
        if (visionSharpness < 0.70f && wV > 0f) {
            sb.append("• Nitidez facial reducida por desenfoque o compresión: peso visual atenuado.\n")
        }

        if (wA > 0f && wV > 0f) {
            sb.append(String.format("• Fusión Tardía Multimodal: Audio (%.0f%% peso) + Visión (%.0f%% peso)\n", wA * 100, wV * 100))
            sb.append(String.format("• Probabilidad Deepfake Audio (Espectrograma Mel 224x224): %.1f%%\n", pAudio * 100))
            sb.append(String.format("• Probabilidad Inconsistencia Facial: %.1f%%", pVision * 100))
        } else if (wA > 0f) {
            sb.append(String.format("• Análisis Unimodal de Audio (CNN Espectral): Probabilidad Deepfake %.1f%%", pAudio * 100))
        } else {
            sb.append(String.format("• Análisis Unimodal Visual: Probabilidad Inconsistencia Facial %.1f%%", pVision * 100))
        }
        return sb.toString()
    }

    fun formatNotificationSummary(
        channel: String,
        riskPercent: Float,
        isFaceDetected: Boolean = true,
        isSenderBlocked: Boolean = false
    ): String {
        val isHigh = riskPercent >= 70f || isSenderBlocked
        val isMedium = !isHigh && riskPercent >= 30f
        return when (channel.uppercase()) {
            "VIDEO" -> when {
                isHigh -> "Video sospechoso con posible alteración audiovisual"
                isMedium -> "Video con indicios moderados de alteración (Verificación recomendada)"
                else -> "Video entrante analizado (Contenido audiovisual auténtico)"
            }
            "IMAGE" -> when {
                !isFaceDetected -> "Fotografía recibida (Descarte automático: Sin rostro humano)"
                isHigh -> "Fotografía sospechosa (Alerta: Posible FaceSwap / Deepfake)"
                isMedium -> "Fotografía con indicios moderados de alteración (Verificación recomendada)"
                else -> "Fotografía entrante analizada (Rostro humano auténtico)"
            }
            else -> when {
                isHigh -> "Nota de voz sospechosa (Alerta: Posible clonación sintética)"
                isMedium -> "Nota de voz con indicios moderados de clonación (Verificación recomendada)"
                else -> "Nota de voz entrante analizada (Voz humana auténtica)"
            }
        }
    }

    private data class Tuple4<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)
}

