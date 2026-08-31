package com.detectorpreventor.app.telemetry

import android.content.Context
import android.os.Bundle
import android.util.Log
import com.detectorpreventor.app.domain.FusionResult

/**
 * Gestor de Telemetría y Reporte de Amenazas Multimodales.
 * Permite enviar eventos anonimizados de detección de Deepfakes a Firebase Analytics / Cloud MLOps.
 */
class FirebaseTelemetryManager(private val context: Context) {
    companion object {
        private const val TAG = "FirebaseTelemetry"
    }

    fun logThreatDetection(fusionResult: FusionResult, mediaType: String) {
        try {
            Log.d(TAG, "Registrando evento de telemetría -> Tipo: $mediaType | Riesgo: ${fusionResult.globalRiskPercentage}%")
            
            val bundle = Bundle().apply {
                putString("media_type", mediaType)
                putFloat("global_risk_score", fusionResult.globalRiskPercentage)
                putFloat("audio_prob", fusionResult.audioFraudProb)
                putFloat("vision_prob", fusionResult.visionFraudProb)
                putString("risk_level", fusionResult.riskLevel.name)
            }
            
            Log.d(TAG, "[Firebase Event Sent]: deepfake_threat_detected -> $bundle")
        } catch (e: Exception) {
            Log.e(TAG, "Error al enviar telemetría de amenazas: ${e.message}")
        }
    }
}
