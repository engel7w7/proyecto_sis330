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

    private const val DEFAULT_WEIGHT_AUDIO = 0.6f
    private const val DEFAULT_WEIGHT_VISION = 0.4f

    fun calculateGlobalRisk(
        audioProb: Float?,
        visionProb: Float?,
        customWeightAudio: Float = DEFAULT_WEIGHT_AUDIO,
        customWeightVision: Float = DEFAULT_WEIGHT_VISION,
        isFaceDetected: Boolean = true
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
                val totalWeight = customWeightAudio + customWeightVision
                val normAudioWeight = customWeightAudio / totalWeight
                val normVisionWeight = customWeightVision / totalWeight
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
        val maxThreat = if (audioProb != null && effectiveVisionProb != null) maxOf(finalAudioProb, finalVisionProb) else weightedScore
        val finalScore = if (maxThreat >= 0.75f) maxOf(weightedScore, maxThreat * 0.95f) else weightedScore
        val globalPercentage = (finalScore * 100.0f).coerceIn(0.0f, 100.0f)

        val riskLevel = when {
            globalPercentage < 30.0f -> RiskLevel.BAJO
            globalPercentage < 70.0f -> RiskLevel.MEDIO
            else -> RiskLevel.ALTO
        }

        val summary = generateDiagnosticText(globalPercentage, finalAudioProb, finalVisionProb, wAudio, wVision, riskLevel, isFaceDetected)

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

    private fun generateDiagnosticText(
        scorePercent: Float,
        pAudio: Float,
        pVision: Float,
        wA: Float,
        wV: Float,
        level: RiskLevel,
        isFaceDetected: Boolean
    ): String {
        val sb = StringBuilder()
        sb.append(String.format("Veredicto Final: %.1f%% de Riesgo (%s).\n", scorePercent, level.name))
        
        if (!isFaceDetected) {
            sb.append("• Advertencia: Sin detección facial en el canal visual (neutralizado a 0.0% por descarte automático).\n")
        }

        if (wA > 0f && wV > 0f) {
            sb.append(String.format("• Fusión Tardía Multimodal: Audio (%.0f%% peso) + Visión (%.0f%% peso)\n", wA * 100, wV * 100))
            sb.append(String.format("• Probabilidad Deepfake Audio (STFT): %.1f%%\n", pAudio * 100))
            sb.append(String.format("• Probabilidad Inconsistencia Facial: %.1f%%", pVision * 100))
        } else if (wA > 0f) {
            sb.append(String.format("• Análisis Unimodal de Audio (STFT): Probabilidad Deepfake %.1f%%", pAudio * 100))
        } else {
            sb.append(String.format("• Análisis Unimodal Visual: Probabilidad Inconsistencia Facial %.1f%%", pVision * 100))
        }
        return sb.toString()
    }

    private data class Tuple4<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)
}

