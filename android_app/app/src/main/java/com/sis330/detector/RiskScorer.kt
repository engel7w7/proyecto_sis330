package com.sis330.detector

/**
 * Algoritmo de Fusión Tardía a Nivel de Puntuación (Score-Level Fusion).
 * Proyecto SIS-330: Detector Móvil Multimodal de Estafas Digitales (Edge AI).
 *
 * Combina la probabilidad de manipulación del Experto de Audio (MobileNetV3) con las
 * probabilidades de los 5 keyframes visuales del Experto de Visión (EfficientNet-B0),
 * aplicando la "Regla 3/5" para la mitigación robusta de falsos positivos visuales.
 */
object RiskScorer {

    // Umbral de decisión binaria para considerar un keyframe como manipulado / anómalo
    private const val FRAME_FRAUD_THRESHOLD = 0.50f

    // Ponderación de Fusión Multimodal (55% Audio, 45% Visión)
    private const val WEIGHT_AUDIO = 0.55f
    private const val WEIGHT_VISION = 0.45f

    // Umbrales del Veredicto Final
    private const val THRESHOLD_SUSPICIOUS = 35.0f
    private const val THRESHOLD_HIGH_RISK = 70.0f

    /**
     * Resultado estructurado del análisis de riesgo multimodal.
     */
    data class RiskResult(
        val globalRiskScore: Float,       // Porcentaje de riesgo de 0.0% a 100.0%
        val verdict: String,              // "Seguro", "Sospechoso" o "Estafa Inminente"
        val audioProb: Float,             // Probabilidad individual de audio (0.0f a 1.0f)
        val visualProb: Float,            // Probabilidad individual visual agregada (0.0f a 1.0f)
        val framesFlaggedCount: Int,      // Cantidad de keyframes que superaron el umbral (0 a 5)
        val rule35Triggered: Boolean,     // Verdadero si se cumplió la Regla 3/5
        val diagnosticDetail: String      // Justificación técnica detallada
    )

    /**
     * Calcula el Score Global de Riesgo ponderado y el Veredicto.
     *
     * @param audioFraudProb Probabilidad de fraude de audio calculada por MobileNetV3 (0.0 a 1.0)
     * @param visualKeyframeProbs Lista con las probabilidades de los 5 keyframes visuales (0.0 a 1.0)
     * @return RiskResult con el Score Global, Veredicto y desglose de diagnóstico.
     */
    fun calculateRisk(
        audioFraudProb: Float,
        visualKeyframeProbs: List<Float>
    ): RiskResult {
        // Asegurar que la probabilidad de audio esté acotada en [0, 1]
        val cleanAudioProb = audioFraudProb.coerceIn(0.0f, 1.0f)

        // Normalizar la lista de keyframes para que tenga exactamente 5 elementos
        val keyframes = if (visualKeyframeProbs.isEmpty()) {
            List(5) { 0.0f }
        } else {
            visualKeyframeProbs.map { it.coerceIn(0.0f, 1.0f) }.take(5).let { list ->
                if (list.size < 5) list + List(5 - list.size) { list.last() } else list
            }
        }

        // =========================================================================
        // REGLA 3/5 (Consenso de Keyframes Visuales):
        // En video/fotogramas faciales, artefactos aislados de compresión o iluminación
        // pueden inducir falsos positivos en 1 o 2 frames.
        // La regla 3/5 exige que AL MENOS 3 DE LOS 5 KEYFRAMES superen el umbral para
        // validar una inconsistencia facial sistemática (Deepfake).
        // =========================================================================
        val framesFlagged = keyframes.count { it >= FRAME_FRAUD_THRESHOLD }
        val rule35Triggered = framesFlagged >= 3

        val aggregatedVisualProb: Float = if (rule35Triggered) {
            // Si 3 o más frames están manipulados, se promedian los 3 más altos con un factor de certeza
            val top3Frames = keyframes.sortedDescending().take(3)
            (top3Frames.average().toFloat() * 1.05f).coerceIn(0.0f, 1.0f)
        } else {
            // Si menos de 3 frames superan el umbral, se descarta la manipulación facial sistemática
            (keyframes.average().toFloat() * 0.70f).coerceIn(0.0f, 1.0f)
        }

        // =========================================================================
        // FUSIÓN TARDÍA MULTIMODAL PONDERADA (SCORE-LEVEL FUSION):
        // Score = (w_audio * P_audio) + (w_vision * P_vision)
        // =========================================================================
        val fusedScore = (WEIGHT_AUDIO * cleanAudioProb) + (WEIGHT_VISION * aggregatedVisualProb)
        val globalPercentage = (fusedScore * 100.0f).coerceIn(0.0f, 100.0f)

        // =========================================================================
        // DETERMINACIÓN DEL VEREDICTO DE CIBERSEGURIDAD
        // =========================================================================
        val verdict = when {
            globalPercentage < THRESHOLD_SUSPICIOUS -> "Seguro"
            globalPercentage < THRESHOLD_HIGH_RISK -> "Sospechoso"
            else -> "Estafa Inminente"
        }

        val diagnosticDetail = buildDiagnosticDetail(
            globalPercentage, verdict, cleanAudioProb, aggregatedVisualProb, framesFlagged, rule35Triggered
        )

        return RiskResult(
            globalRiskScore = globalPercentage,
            verdict = verdict,
            audioProb = cleanAudioProb,
            visualProb = aggregatedVisualProb,
            framesFlaggedCount = framesFlagged,
            rule35Triggered = rule35Triggered,
            diagnosticDetail = diagnosticDetail
        )
    }

    private fun buildDiagnosticDetail(
        score: Float,
        verdict: String,
        pAudio: Float,
        pVision: Float,
        flaggedCount: Int,
        rule35: Boolean
    ): String {
        return buildString {
            append("Veredicto: $verdict (Score: ${String.format("%.1f", score)}%)\n")
            append("• Audio (Voz/STFT): ${String.format("%.1f", pAudio * 100)}% prob. clonación\n")
            append("• Visión (Keyframes): ${String.format("%.1f", pVision * 100)}% prob. deepfake\n")
            append("• Regla 3/5: $flaggedCount/5 frames anómalos (${if (rule35) "ACTIVADA - Manipulación Confirmada" else "DESACTIVADA - Falso Positivo Evitado"})")
        }
    }
}
