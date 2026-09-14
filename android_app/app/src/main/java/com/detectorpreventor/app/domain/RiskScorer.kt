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
    val diagnosticSummary: String
)

/**
 * Módulo de Fusión Tardía (Score-Level Fusion):
 * Combina ponderadamente las probabilidades calculadas por los modelos expertos de Audio y Visión.
 */
object RiskScorer {

    private const val DEFAULT_WEIGHT_AUDIO = 0.6f
    private const val DEFAULT_WEIGHT_VISION = 0.4f

    /**
     * Calcula el porcentaje de riesgo global a partir de las probabilidades individuales.
     * Fórmula: Score = (w_a * P_audio) + (w_v * P_vision)
     */
    fun calculateGlobalRisk(
        audioProb: Float?,
        visionProb: Float?,
        customWeightAudio: Float = DEFAULT_WEIGHT_AUDIO,
        customWeightVision: Float = DEFAULT_WEIGHT_VISION
    ): FusionResult {
        val (finalAudioProb, finalVisionProb, wAudio, wVision) = when {
            audioProb != null && visionProb != null -> {
                val totalWeight = customWeightAudio + customWeightVision
                val normAudioWeight = customWeightAudio / totalWeight
                val normVisionWeight = customWeightVision / totalWeight
                Tuple4(audioProb, visionProb, normAudioWeight, normVisionWeight)
            }
            audioProb != null -> {
                Tuple4(audioProb, 0f, 1.0f, 0.0f)
            }
            visionProb != null -> {
                Tuple4(0f, visionProb, 0.0f, 1.0f)
            }
            else -> {
                Tuple4(0f, 0f, 0.5f, 0.5f)
            }
        }

        val score = (wAudio * finalAudioProb) + (wVision * finalVisionProb)
        val globalPercentage = (score * 100.0f).coerceIn(0.0f, 100.0f)

        val riskLevel = when {
            globalPercentage < 30.0f -> RiskLevel.BAJO
            globalPercentage < 70.0f -> RiskLevel.MEDIO
            else -> RiskLevel.ALTO
        }

        val summary = generateDiagnosticText(globalPercentage, finalAudioProb, finalVisionProb, wAudio, wVision, riskLevel)

        return FusionResult(
            globalRiskPercentage = globalPercentage,
            audioFraudProb = finalAudioProb,
            visionFraudProb = finalVisionProb,
            riskLevel = riskLevel,
            weightAudio = wAudio,
            weightVision = wVision,
            diagnosticSummary = summary
        )
    }

    private fun generateDiagnosticText(
        scorePercent: Float,
        pAudio: Float,
        pVision: Float,
        wA: Float,
        wV: Float,
        level: RiskLevel
    ): String {
        val sb = StringBuilder()
        sb.append(String.format("Veredicto Final: %.1f%% de Riesgo (%s).\n", scorePercent, level.name))
        
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

