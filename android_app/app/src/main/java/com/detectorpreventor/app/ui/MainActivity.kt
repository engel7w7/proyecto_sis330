package com.detectorpreventor.app.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
import com.detectorpreventor.app.notifications.InterceptedNotification
import com.detectorpreventor.app.notifications.NotificationMonitorService
import com.detectorpreventor.app.notifications.NotificationRepository
import com.detectorpreventor.app.ui.theme.BackgroundDark
import com.detectorpreventor.app.ui.theme.DetectorPreventorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.UUID

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

    private var currentPayload by mutableStateOf<ProcessedMediaPayload?>(null)
    private var currentFusionResult by mutableStateOf<FusionResult?>(null)
    private var activeScreen by mutableStateOf<ScreenNav>(ScreenNav.Scanner)

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
                        activeScreen = activeScreen,
                        onScreenChange = { activeScreen = it },
                        onSelectFile = {
                            filePickerLauncher.launch("*/*")
                        },
                        onAnalyzeSample = { sampleType, index ->
                            runSampleAnalysis(sampleType, index)
                        },
                        onInspectNotification = { notif ->
                            inspectNotification(notif)
                        },
                        onSimulateNotification = { mediaType, isThreat, sender, text ->
                            simulateIncomingNotification(mediaType, isThreat, sender, text)
                        }
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        NotificationMonitorService.ensureServiceBound(this)
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleIncomingIntent(it) }
    }

    private fun handleIncomingIntent(intent: Intent) {
        val action = intent.action
        val type = intent.type

        Log.d(TAG, "Intent Recibido -> Acción: $action | Tipo MIME: $type")

        val screenExtra = intent.getStringExtra("screen")
        if (!screenExtra.isNullOrBlank()) {
            activeScreen = when (screenExtra.lowercase()) {
                "upload" -> ScreenNav.Upload
                "notifications", "whatsapp" -> ScreenNav.Notifications
                "benchmark", "dataset" -> ScreenNav.Benchmark
                "info", "mlops" -> ScreenNav.SystemInfo
                else -> ScreenNav.Scanner
            }
        }

        val sampleType = intent.getStringExtra("sample_type")
        val sampleIndex = intent.getIntExtra("sample_index", -1)
        if (!sampleType.isNullOrBlank() && sampleIndex > 0) {
            runSampleAnalysis(sampleType, sampleIndex)
            return
        }

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
        } else if (currentPayload == null) {
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

            val sampleName = when (sampleType) {
                "audio" -> when (index) {
                    1 -> "Muestra 1: Voz Humana Real (ASVspoof Bonafide).opus"
                    2 -> "Muestra 2: Clonación por IA (ASVspoof Deepfake).opus"
                    3 -> "Muestra 3: Síntesis de Texto a Voz (TTS AI).wav"
                    4 -> "Muestra 4: Nota de Voz WhatsApp Replicada.opus"
                    else -> "Muestra 5: Conversación Telefónica Auténtica.mp3"
                }
                "image" -> when (index) {
                    1 -> "Muestra 1: Retrato Real Prístino (FF++).jpg"
                    2 -> "Muestra 2: Manipulación FaceSwap AI.jpg"
                    3 -> "Muestra 3: Rostro Sintético GAN (CIFAKE).png"
                    4 -> "Muestra 4: Reenactamiento Face2Face.jpg"
                    else -> "Muestra 5: Fotografía HD Original.jpg"
                }
                else -> when (index) {
                    1 -> "Muestra 1: Entrevista Real YouTube (FF++).mp4"
                    2 -> "Muestra 2: Sincronización Labial LipSync.mp4"
                    3 -> "Muestra 3: FaceSwap HD Video.mp4"
                    4 -> "Muestra 4: Avatar IA Multimodal Completo.mp4"
                    else -> "Muestra 5: Clip de Cámara Frontal Auténtico.mp4"
                }
            }

            payload = payload.copy(filename = sampleName)

            val audioProb: Float? = payload.audioSpectrogram?.let {
                audioClassifier.classifySpectrogram(it)
            }

            val visionProb: Float? = payload.faceKeyframe?.let {
                visionClassifier.classifyFaceKeyframe(it)
            }

            val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

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

    private fun inspectNotification(notif: InterceptedNotification) {
        lifecycleScope.launch(Dispatchers.Default) {
            val sampleUri = Uri.parse("content://com.detectorpreventor.app.notifications/${notif.id}")
            val mime = when (notif.mediaType) {
                MediaType.AUDIO_ONLY -> "audio/opus"
                MediaType.VIDEO_MULTIMODAL -> "video/mp4"
                else -> "image/jpeg"
            }
            var payload = mediaRouter.processIncomingUri(sampleUri, mime)
            payload = payload.copy(filename = "${notif.appName}: ${notif.sender} - ${notif.text}")

            val audioProb: Float? = payload.audioSpectrogram?.let {
                audioClassifier.classifySpectrogram(it)
            }
            val visionProb: Float? = payload.faceKeyframe?.let {
                visionClassifier.classifyFaceKeyframe(it)
            }

            val fusionResult = RiskScorer.calculateGlobalRisk(audioProb, visionProb)

            lifecycleScope.launch(Dispatchers.Main) {
                currentPayload = payload
                currentFusionResult = fusionResult
                activeScreen = ScreenNav.Scanner
            }
        }
    }

    private fun simulateIncomingNotification(mediaType: MediaType, isThreat: Boolean, sender: String, text: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            val (audioProb, visionProb) = when (mediaType) {
                MediaType.AUDIO_ONLY -> {
                    val asset = if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png"
                    val bitmap = loadAssetBitmap(asset)
                    val prob = bitmap?.let { audioClassifier.classifySpectrogram(it) }
                    Pair(prob, null)
                }
                MediaType.IMAGE_ONLY -> {
                    val asset = if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg"
                    val bitmap = loadAssetBitmap(asset)
                    val prob = bitmap?.let { visionClassifier.classifyFaceKeyframe(it) }
                    Pair(null, prob)
                }
                MediaType.VIDEO_MULTIMODAL -> {
                    val faceAsset = if (isThreat) "samples/image_fake_face.jpg" else "samples/image_real_face.jpg"
                    val audioAsset = if (isThreat) "samples/audio_fake_spec.png" else "samples/audio_real_spec.png"
                    val fBitmap = loadAssetBitmap(faceAsset)
                    val aBitmap = loadAssetBitmap(audioAsset)
                    val vProb = fBitmap?.let { visionClassifier.classifyFaceKeyframe(it) }
                    val aProb = aBitmap?.let { audioClassifier.classifySpectrogram(it) }
                    Pair(aProb, vProb)
                }
                else -> Pair(null, null)
            }

            val fusion = RiskScorer.calculateGlobalRisk(audioProb, visionProb)
            val notif = InterceptedNotification(
                id = UUID.randomUUID().toString(),
                appName = "WhatsApp",
                packageName = "com.whatsapp",
                sender = sender,
                text = text,
                timestamp = System.currentTimeMillis(),
                mediaType = mediaType,
                riskScore = fusion.globalRiskPercentage,
                isThreat = fusion.globalRiskPercentage >= 70f,
                fusionResult = fusion
            )
            NotificationRepository.addNotification(notif)
            Log.i(TAG, "Notificacion simulada evaluada por TFLite: [WhatsApp] $sender - Riesgo: ${fusion.globalRiskPercentage}%")
        }
    }

    private fun loadAssetBitmap(path: String): Bitmap? {
        return try {
            assets.open(path).use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando asset $path: ${e.message}")
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        audioClassifier.close()
        visionClassifier.close()
    }
}

