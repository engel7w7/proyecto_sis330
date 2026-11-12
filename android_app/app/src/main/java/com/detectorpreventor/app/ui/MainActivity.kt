package com.detectorpreventor.app.ui

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.provider.Settings
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
import com.detectorpreventor.app.domain.WhatsAppAudioItem
import com.detectorpreventor.app.domain.WhatsAppMediaScanner
import com.detectorpreventor.app.domain.WhatsAppStorageAccessState
import com.detectorpreventor.app.ml.AudioClassifier
import com.detectorpreventor.app.ml.VisionClassifier
import com.detectorpreventor.app.notifications.InterceptedNotification
import com.detectorpreventor.app.notifications.NotificationMediaCache
import com.detectorpreventor.app.notifications.NotificationMonitorService
import com.detectorpreventor.app.notifications.NotificationRepository
import com.detectorpreventor.app.ui.theme.BackgroundDark
import com.detectorpreventor.app.ui.theme.DetectorPreventorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

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
    private var currentlyInspectedNotificationId: String? = null

    private var whatsAppAudios by mutableStateOf<List<WhatsAppAudioItem>>(emptyList())
    private var whatsAppAccessState by mutableStateOf(WhatsAppStorageAccessState(false, false, 0, false))
    private var isScanningWhatsApp by mutableStateOf(false)

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val fileName = getFileNameFromUri(it)
            val mimeType = contentResolver.getType(it)
            processMediaUri(it, mimeType, fileName)
        }
    }

    private val manageAllFilesLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        refreshWhatsAppAudios()
    }

    private val whatsAppTreeLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { treeUri: Uri? ->
        if (treeUri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    treeUri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (e: Exception) {
                Log.w(TAG, "No se pudo persistir permiso de carpeta SAF: ${e.message}")
            }
            refreshWhatsAppAudios()
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        Log.i(TAG, "Permisos multimedia evaluados: $permissions")
        refreshWhatsAppAudios()
    }

    private fun refreshWhatsAppAudios() {
        lifecycleScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.Main) {
                if (!isFinishing && !isDestroyed) {
                    isScanningWhatsApp = true
                }
            }
            val access = WhatsAppMediaScanner.getStorageAccessState(applicationContext)
            val items = WhatsAppMediaScanner.scanWhatsAppAudio(applicationContext, maxItems = 60)
            withContext(Dispatchers.Main) {
                if (!isFinishing && !isDestroyed) {
                    whatsAppAccessState = access
                    whatsAppAudios = items
                    isScanningWhatsApp = false
                }
            }
        }
    }

    private fun requestAllFilesAccess() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            try {
                val intent = Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                manageAllFilesLauncher.launch(intent)
            } catch (_: Exception) {
                try {
                    manageAllFilesLauncher.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
                } catch (e: Exception) {
                    Log.w(TAG, "No se pudo abrir configuracion de acceso total a archivos: ${e.message}")
                }
            }
        }
    }

    private fun checkAndRequestMediaPermissions() {
        val permissionsToRequest = mutableListOf<String>()
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.READ_MEDIA_IMAGES) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(android.Manifest.permission.READ_MEDIA_IMAGES)
            }
            if (checkSelfPermission(android.Manifest.permission.READ_MEDIA_VIDEO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(android.Manifest.permission.READ_MEDIA_VIDEO)
            }
            if (checkSelfPermission(android.Manifest.permission.READ_MEDIA_AUDIO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(android.Manifest.permission.READ_MEDIA_AUDIO)
            }
            if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(android.Manifest.permission.POST_NOTIFICATIONS)
            }
        } else {
            if (checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        checkAndRequestMediaPermissions()

        mediaRouter = MediaRouter(applicationContext)
        audioClassifier = AudioClassifier(applicationContext)
        visionClassifier = VisionClassifier(applicationContext)

        refreshWhatsAppAudios()
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
                        },
                        onDismissInspectedNotification = {
                            dismissCurrentInspectedNotification()
                        },
                        whatsAppAudios = whatsAppAudios,
                        whatsAppAccessState = whatsAppAccessState,
                        isScanningWhatsApp = isScanningWhatsApp,
                        onRefreshWhatsAppAudios = {
                            refreshWhatsAppAudios()
                        },
                        onRequestAllFilesAccess = {
                            requestAllFilesAccess()
                        },
                        onPickWhatsAppFolder = {
                            whatsAppTreeLauncher.launch(null)
                        },
                        onAnalyzeWhatsAppAudio = { item ->
                            item.uri?.let { uri -> processMediaUri(uri, null, item.displayName) }
                        }
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        NotificationMonitorService.ensureServiceBound(this)
        NotificationRepository.autoResolvePendingNotifications(this)
        refreshWhatsAppAudios()
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

        val simulateMediaType = intent.getStringExtra("simulate_media_type")
        if (!simulateMediaType.isNullOrBlank()) {
            val isThreat = intent.getBooleanExtra("simulate_threat", false)
            val isFaceDiscard = intent.getBooleanExtra("simulate_face_discard", false)
            val sender = intent.getStringExtra("simulate_sender") ?: "Contacto WhatsApp"
            val text = intent.getStringExtra("simulate_text") ?: (if (isFaceDiscard) "Foto paisaje sin personas (descarte_blazeface.jpg)" else "Mensaje de prueba")
            val mType = when (simulateMediaType.lowercase()) {
                "image" -> MediaType.IMAGE_ONLY
                "audio" -> MediaType.AUDIO_ONLY
                "video" -> MediaType.VIDEO_MULTIMODAL
                else -> MediaType.IMAGE_ONLY
            }
            simulateIncomingNotification(mType, isThreat, sender, text, isFaceDiscard)
            return
        }

        val inspectNotifId = intent.getStringExtra("inspect_notif_id")
        if (!inspectNotifId.isNullOrBlank()) {
            val target = NotificationRepository.notifications.value.find { it.id == inspectNotifId }
            if (target != null) {
                inspectNotification(target)
                return
            }
        }

        if (intent.getBooleanExtra("dismiss_inspected", false)) {
            dismissCurrentInspectedNotification()
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

    private fun evaluatePayload(rawPayload: ProcessedMediaPayload): Pair<ProcessedMediaPayload, FusionResult> {
        val audioProb: Float? = if (rawPayload.mediaType != MediaType.IMAGE_ONLY) {
            when {
                rawPayload.audioPcm != null -> audioClassifier.classifyPcm(rawPayload.audioPcm, null)
                rawPayload.audioSpectrogram != null -> audioClassifier.classifySpectrogram(rawPayload.audioSpectrogram, null)
                else -> null
            }
        } else null

        val isFaceDetected = rawPayload.faceDetected
        val frames = if (rawPayload.mediaType != MediaType.AUDIO_ONLY && isFaceDetected) {
            rawPayload.faceKeyframes.ifEmpty { listOfNotNull(rawPayload.faceKeyframe) }
        } else {
            emptyList()
        }

        val frameProbs = mutableListOf<Float>()
        var highestRiskFrame: Bitmap? = null
        var highestRiskGradCam: Bitmap? = null
        var highestFrameProb = -1.0f

        for (frame in frames) {
            val p = visionClassifier.classifyFaceKeyframe(frame, null, requireFaceDetection = false)
            frameProbs.add(p)
            if (p > highestFrameProb) {
                highestFrameProb = p
                highestRiskFrame = frame
                highestRiskGradCam = visionClassifier.getLastGradCamHeatmap()
            }
        }

        val visionGradCam = if (rawPayload.mediaType != MediaType.AUDIO_ONLY && isFaceDetected) {
            highestRiskGradCam
        } else null
        val audioGradCam = if (audioProb != null) audioClassifier.getLastGradCamHeatmap() else null
        val primaryGradCam = when {
            rawPayload.mediaType == MediaType.AUDIO_ONLY -> audioGradCam
            !isFaceDetected -> null
            else -> visionGradCam ?: audioGradCam
        }

        val finalPayload = rawPayload.copy(
            faceKeyframe = highestRiskFrame ?: rawPayload.faceKeyframe,
            faceDetected = isFaceDetected,
            discardReason = if (!isFaceDetected) "Descarte automático: No se detectó rostro humano" else rawPayload.discardReason,
            gradCamBitmap = primaryGradCam,
            audioGradCamBitmap = audioGradCam
        )

        val audioConf = if (rawPayload.mediaType != MediaType.IMAGE_ONLY && (rawPayload.audioPcm != null || rawPayload.audioSpectrogram != null)) {
            rawPayload.audioConfidence
        } else 1.0f

        val fusionResult = if (rawPayload.mediaType == MediaType.VIDEO_MULTIMODAL) {
            RiskScorer.calculateVideoRisk(
                audioProb = audioProb,
                frameProbs = frameProbs,
                isFaceDetected = isFaceDetected,
                frameQualities = rawPayload.faceKeyframesQualities,
                audioConfidence = audioConf
            )
        } else {
            val singleVisionProb = if (rawPayload.mediaType == MediaType.AUDIO_ONLY) {
                null
            } else if (isFaceDetected) {
                frameProbs.firstOrNull()
            } else {
                0.0f
            }
            val singleVisionSharpness = if (isFaceDetected && rawPayload.mediaType != MediaType.AUDIO_ONLY) {
                rawPayload.faceKeyframesQualities.firstOrNull()
                    ?: frames.firstOrNull()?.let { VisionClassifier.computeLaplacianSharpness(it) }
                    ?: 1.0f
            } else 1.0f
            RiskScorer.calculateGlobalRisk(
                audioProb = audioProb,
                visionProb = singleVisionProb,
                isFaceDetected = isFaceDetected,
                audioConfidence = audioConf,
                visionSharpness = singleVisionSharpness
            )
        }
        return Pair(finalPayload, fusionResult)
    }

    private fun processMediaUri(uri: Uri, mimeType: String?, customName: String? = null) {
        currentlyInspectedNotificationId = null
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                var rawPayload = mediaRouter.processIncomingUri(uri, mimeType)
                if (!customName.isNullOrBlank()) {
                    rawPayload = rawPayload.copy(filename = customName)
                }

                val (payload, fusionResult) = evaluatePayload(rawPayload)
                Log.i(
                    TAG,
                    "Riesgo calculado (${payload.filename}) -> tipo=${payload.mediaType}, rostro=${payload.faceDetected}, audioProb=${fusionResult.audioFraudProb}, visionProb=${fusionResult.visionFraudProb}, global=${fusionResult.globalRiskPercentage}%"
                )

                withContext(Dispatchers.Main) {
                    if (!isFinishing && !isDestroyed) {
                        currentPayload = payload
                        currentFusionResult = fusionResult
                        activeScreen = ScreenNav.Scanner
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error procesando el archivo: ${t.message}", t)
            }
        }
    }

    private fun runSampleAnalysis(sampleType: String, index: Int) {
        currentlyInspectedNotificationId = null
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val sampleUri = Uri.parse("content://com.detectorpreventor.app.samples/$sampleType/$index")
                val mime = when (sampleType) {
                    "audio" -> "audio/opus"
                    "image" -> "image/jpeg"
                    else -> "video/mp4"
                }

                var rawPayload = mediaRouter.processIncomingUri(sampleUri, mime)

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
                        3 -> "Muestra 3: Rostro Sintético GAN (StyleGAN).png"
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

                rawPayload = rawPayload.copy(filename = sampleName)
                val (payload, fusionResult) = evaluatePayload(rawPayload)

                withContext(Dispatchers.Main) {
                    if (!isFinishing && !isDestroyed) {
                        currentPayload = payload
                        currentFusionResult = fusionResult
                        activeScreen = ScreenNav.Scanner
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error ejecutando analisis de muestra ($sampleType/$index): ${t.message}", t)
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

    private fun dismissCurrentInspectedNotification() {
        val notifId = currentlyInspectedNotificationId
        if (notifId != null) {
            NotificationRepository.removeNotification(notifId)
            currentlyInspectedNotificationId = null
        }
        currentPayload = null
        currentFusionResult = null
        activeScreen = ScreenNav.Notifications
    }

    private fun inspectNotification(notif: InterceptedNotification) {
        currentlyInspectedNotificationId = notif.id
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                if (notif.isFaceDiscarded()) {
                    val payload = ProcessedMediaPayload(
                        mediaType = MediaType.IMAGE_ONLY,
                        audioSpectrogram = null,
                        faceKeyframe = NotificationMediaCache.getFaceBitmap(this@MainActivity, notif.id),
                        filename = "${notif.appName}: ${notif.sender} - ${notif.text}",
                        faceDetected = false,
                        discardReason = "Descarte automático BlazeFace: Ausencia de rostro humano confirmado por Edge AI."
                    )
                    val fusion = notif.fusionResult ?: RiskScorer.calculateGlobalRisk(
                        audioProb = null,
                        visionProb = null,
                        isFaceDetected = false
                    )
                    withContext(Dispatchers.Main) {
                        if (!isFinishing && !isDestroyed) {
                            currentPayload = payload
                            currentFusionResult = fusion
                            activeScreen = ScreenNav.Scanner
                        }
                    }
                    return@launch
                }

                if (notif.mediaType == MediaType.UNKNOWN) {
                    val fusionResult = notif.fusionResult ?: FusionResult(
                        globalRiskPercentage = notif.riskScore,
                        audioFraudProb = 0.0f,
                        visionFraudProb = 0.0f,
                        riskLevel = if (notif.riskScore >= 70f) RiskLevel.ALTO else if (notif.riskScore >= 35f) RiskLevel.MEDIO else RiskLevel.BAJO,
                        weightAudio = 0.5f,
                        weightVision = 0.5f,
                        diagnosticSummary = notif.text
                    )
                    val payload = ProcessedMediaPayload(
                        mediaType = MediaType.UNKNOWN,
                        audioSpectrogram = null,
                        faceKeyframe = null,
                        filename = "${notif.appName}: ${notif.sender}",
                        faceDetected = false,
                        discardReason = "Análisis contextual de texto y enlace web (sin archivo multimedia adjunto)"
                    )
                    withContext(Dispatchers.Main) {
                        if (!isFinishing && !isDestroyed) {
                            currentPayload = payload
                            currentFusionResult = fusionResult
                            activeScreen = ScreenNav.Scanner
                        }
                    }
                    return@launch
                }

                // Auto-resolve if pending before inspecting
                val currentNotif = if (notif.isPendingDownload()) {
                    val resolved = NotificationRepository.resolveSinglePendingNotification(this@MainActivity, notif)
                    NotificationRepository.updateNotification(resolved)
                    resolved
                } else {
                    notif
                }

                val sampleUri = Uri.parse("content://com.detectorpreventor.app.notifications/${currentNotif.id}")
                val mime = when (currentNotif.mediaType) {
                    MediaType.AUDIO_ONLY -> "audio/opus"
                    MediaType.VIDEO_MULTIMODAL -> "video/mp4"
                    else -> "image/jpeg"
                }
                var rawPayload = mediaRouter.processIncomingUri(sampleUri, mime)
                rawPayload = rawPayload.copy(filename = "${currentNotif.appName}: ${currentNotif.sender} - ${currentNotif.text}")
                val (payload, fusionResult) = evaluatePayload(rawPayload)

                val isBlocked = NotificationRepository.isSenderBlocked(currentNotif.sender)
                val adjustedRisk = if (isBlocked) maxOf(fusionResult.globalRiskPercentage, 85.0f) else fusionResult.globalRiskPercentage
                val adjustedFusion = if (isBlocked) {
                    fusionResult.copy(
                        globalRiskPercentage = adjustedRisk,
                        riskLevel = RiskLevel.ALTO,
                        diagnosticSummary = "[REMITENTE BLOQUEADO] ${fusionResult.diagnosticSummary}"
                    )
                } else {
                    fusionResult
                }

                val resolvedNotif = currentNotif.copy(
                    riskScore = adjustedFusion.globalRiskPercentage,
                    isThreat = isBlocked || adjustedFusion.globalRiskPercentage >= 70f,
                    fusionResult = adjustedFusion
                )
                NotificationRepository.updateNotification(resolvedNotif)

                withContext(Dispatchers.Main) {
                    if (!isFinishing && !isDestroyed) {
                        currentPayload = payload
                        currentFusionResult = adjustedFusion
                        activeScreen = ScreenNav.Scanner
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Error inspeccionando notificacion (${notif.id}): ${t.message}", t)
            }
        }
    }

    private fun simulateIncomingNotification(
        mediaType: MediaType,
        isThreat: Boolean,
        sender: String,
        text: String,
        forceFaceDiscard: Boolean = false
    ) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val notifId = UUID.randomUUID().toString()
                val now = System.currentTimeMillis()
                val isFaceDiscard = forceFaceDiscard ||
                    text.contains("sin rostro", ignoreCase = true) ||
                    text.contains("descarte", ignoreCase = true) ||
                    text.contains("paisaje", ignoreCase = true)

                val mediaUriToStore = when (mediaType) {
                    MediaType.AUDIO_ONLY -> if (isThreat) "asset://samples/06_clonacion_ia_spoof_tts_1_spec.png" else "asset://samples/01_voz_humana_real_bonafide_1_spec.png"
                    MediaType.IMAGE_ONLY -> if (isFaceDiscard) "synthetic://landscape_no_face" else if (isThreat) "asset://samples/06_deepfake_rostro_ia_1.jpg" else "asset://samples/01_retrato_humano_real_1.jpg"
                    MediaType.VIDEO_MULTIMODAL -> if (isThreat) "asset://samples/06_deepfake_rostro_ia_1.jpg" else "asset://samples/01_retrato_humano_real_1.jpg"
                    else -> null
                }

                val pendingDesc = when (mediaType) {
                    MediaType.AUDIO_ONLY -> "$text (Descargando nota de voz en WhatsApp...)"
                    MediaType.IMAGE_ONLY -> if (isFaceDiscard) "$text (Descargando imagen para BlazeFace...)" else "$text (Descargando fotografía en WhatsApp...)"
                    MediaType.VIDEO_MULTIMODAL -> "$text (Descargando video en WhatsApp...)"
                    else -> "$text (Descargando...)"
                }
                val pendingNotif = InterceptedNotification(
                    id = notifId,
                    appName = "WhatsApp",
                    packageName = "com.whatsapp",
                    sender = sender,
                    text = pendingDesc,
                    timestamp = now,
                    mediaType = mediaType,
                    riskScore = 0.0f,
                    isThreat = false,
                    mediaUriString = mediaUriToStore,
                    fusionResult = null
                )
                NotificationRepository.addNotification(pendingNotif)

                // Simulated download delay
                kotlinx.coroutines.delay(1000L)

                if (!NotificationRepository.hasNotification(notifId)) {
                    Log.i(TAG, "Notificacion simulada $notifId descartada antes de resolucion.")
                    return@launch
                }

                val resolvedItem = NotificationRepository.resolveSinglePendingNotification(this@MainActivity, pendingNotif)
                val finalNotif = resolvedItem.copy(
                    mediaUriString = resolvedItem.mediaUriString ?: mediaUriToStore
                )
                NotificationRepository.updateNotification(finalNotif)
                Log.i(TAG, "Notificacion simulada resuelta exitosamente por TFLite: [WhatsApp] $sender - Riesgo: ${finalNotif.riskScore}%")
            } catch (t: Throwable) {
                Log.e(TAG, "Error simulando notificacion entrante: ${t.message}", t)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        audioClassifier.close()
        visionClassifier.close()
    }
}
