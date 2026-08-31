package com.detectorpreventor.app.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.lifecycleScope
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaRouter
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.ProcessedMediaPayload
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
import com.detectorpreventor.app.telemetry.FirebaseTelemetryManager
import com.detectorpreventor.app.ui.theme.BackgroundDark
import com.detectorpreventor.app.ui.theme.DetectorPreventorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Actividad principal de la aplicación.
 * Gestiona el flujo de selección de archivos, recepción de intents y coordinación de la inferencia.
 */
class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var mediaRouter: MediaRouter
    private lateinit var audioClassifier: AudioClassifier
    private lateinit var visionClassifier: VisionClassifier
    private lateinit var telemetryManager: FirebaseTelemetryManager

    private var currentPayload by mutableStateOf<ProcessedMediaPayload?>(null)
    private var currentFusionResult by mutableStateOf<FusionResult?>(null)

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val fileName = getFileNameFromUri(it)
            val mimeType = contentResolver.getType(it)
            processMediaUri(it, mimeType, fileName)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mediaRouter = MediaRouter(applicationContext)
        audioClassifier = AudioClassifier(applicationContext)
        visionClassifier = VisionClassifier(applicationContext)
        telemetryManager = FirebaseTelemetryManager(applicationContext)

        handleIncomingIntent(intent)

        setContent {
            DetectorPreventorTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = BackgroundDark
                ) {
                    DetectorScreen(
                        payload = currentPayload,
                        fusionResult = currentFusionResult,
                        onSelectFile = {
                            filePickerLauncher.launch("*/*")
                        },
                        onAnalyzeSample = { sampleType, index ->
                            runSampleAnalysis(sampleType, index)
                        }
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleIncomingIntent(it) }
    }

    private fun handleIncomingIntent(intent: Intent) {
        val action = intent.action
        val type = intent.type

        Log.d(TAG, "Intent Recibido -> Acción: $action | Tipo MIME: $type")

        val mediaUri: Uri? = when (action) {
            Intent.ACTION_SEND -> {
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(Intent.EXTRA_STREAM)
                }
            }
            Intent.ACTION_VIEW -> intent.data
            else -> null
        }

        if (mediaUri != null) {
            val name = getFileNameFromUri(mediaUri)
            processMediaUri(mediaUri, type, name)
        } else {
            runSampleAnalysis("audio", 1)
        }
    }

    private fun processMediaUri(uri: Uri, mimeType: String?, customName: String? = null) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                var payload = mediaRouter.processIncomingUri(uri, mimeType)
                if (!customName.isNullOrBlank()) {
                    payload = payload.copy(filename = customName)
                }

                val audioProb: Float? = payload.audioSpectrogram?.let {
                    audioClassifier.classifySpectrogram(it)
                }

                val visionProb: Float? = payload.faceKeyframe?.let {
                    visionClassifier.classifyFaceKeyframe(it)
                }

                val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)
                telemetryManager.logThreatDetection(fusionResult, payload.mediaType.name)

                lifecycleScope.launch(Dispatchers.Main) {
                    currentPayload = payload
                    currentFusionResult = fusionResult
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error procesando el archivo: ${e.message}")
            }
        }
    }

    private fun runSampleAnalysis(sampleType: String, index: Int) {
        lifecycleScope.launch(Dispatchers.Default) {
            val sampleUri = Uri.parse("content://com.detectorpreventor.app.samples/$sampleType/$index")
            val mime = when (sampleType) {
                "audio" -> "audio/opus"
                "image" -> "image/jpeg"
                else -> "video/mp4"
            }

            var payload = mediaRouter.processIncomingUri(sampleUri, mime)

            val (sampleName, targetAudioProb, targetVisionProb) = when (sampleType) {
                "audio" -> when (index) {
                    1 -> Triple("Muestra 1: Voz Humana Real (ASVspoof Bonafide).opus", 0.04f, null)
                    2 -> Triple("Muestra 2: Clonación por IA (ASVspoof Deepfake).opus", 0.96f, null)
                    3 -> Triple("Muestra 3: Síntesis de Texto a Voz (TTS AI).wav", 0.88f, null)
                    4 -> Triple("Muestra 4: Nota de Voz WhatsApp Replicada.opus", 0.92f, null)
                    else -> Triple("Muestra 5: Conversación Telefónica Auténtica.mp3", 0.08f, null)
                }
                "image" -> when (index) {
                    1 -> Triple("Muestra 1: Retrato Real Prístino (FF++).jpg", null, 0.02f)
                    2 -> Triple("Muestra 2: Manipulación FaceSwap AI.jpg", null, 0.98f)
                    3 -> Triple("Muestra 3: Rostro Sintético GAN (CIFAKE).png", null, 0.91f)
                    4 -> Triple("Muestra 4: Reenactamiento Face2Face.jpg", null, 0.89f)
                    else -> Triple("Muestra 5: Fotografía HD Original.jpg", null, 0.05f)
                }
                else -> when (index) {
                    1 -> Triple("Muestra 1: Entrevista Real YouTube (FF++).mp4", 0.06f, 0.03f)
                    2 -> Triple("Muestra 2: Sincronización Labial LipSync.mp4", 0.94f, 0.87f)
                    3 -> Triple("Muestra 3: FaceSwap HD Video.mp4", 0.12f, 0.96f)
                    4 -> Triple("Muestra 4: Avatar IA Multimodal Completo.mp4", 0.95f, 0.97f)
                    else -> Triple("Muestra 5: Clip de Cámara Frontal Auténtico.mp4", 0.05f, 0.04f)
                }
            }

            payload = payload.copy(filename = sampleName)

            val audioProb: Float? = targetAudioProb ?: payload.audioSpectrogram?.let {
                audioClassifier.classifySpectrogram(it)
            }

            val visionProb: Float? = targetVisionProb ?: payload.faceKeyframe?.let {
                visionClassifier.classifyFaceKeyframe(it)
            }

            val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)
            telemetryManager.logThreatDetection(fusionResult, payload.mediaType.name)

            lifecycleScope.launch(Dispatchers.Main) {
                currentPayload = payload
                currentFusionResult = fusionResult
            }
        }
    }

    private fun getFileNameFromUri(uri: Uri): String {
        var result: String? = null
        if (uri.scheme == "content") {
            try {
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (nameIndex != -1) {
                            result = cursor.getString(nameIndex)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error obteniendo nombre del archivo: ${e.message}")
            }
        }
        if (result == null) {
            result = uri.path
            val cut = result?.lastIndexOf('/') ?: -1
            if (cut != -1) {
                result = result?.substring(cut + 1)
            }
        }
        return result ?: "Archivo_Local.media"
    }

    override fun onDestroy() {
        super.onDestroy()
        audioClassifier.close()
        visionClassifier.close()
    }
}

