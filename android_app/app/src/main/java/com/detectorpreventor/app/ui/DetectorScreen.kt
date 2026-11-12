package com.detectorpreventor.app.ui

import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.content.Intent
import android.os.Build
import android.provider.Settings
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.ProcessedMediaPayload
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.domain.WhatsAppAudioItem
import com.detectorpreventor.app.domain.WhatsAppStorageAccessState
import com.detectorpreventor.app.notifications.InterceptedNotification
import com.detectorpreventor.app.notifications.NotificationMonitorService
import com.detectorpreventor.app.notifications.NotificationRepository
import com.detectorpreventor.app.ui.theme.*
import java.util.UUID

sealed class ScreenNav(val route: String, val title: String, val icon: ImageVector) {
    object Scanner : ScreenNav("scanner", "Escáner", Icons.Default.Shield)
    object Upload : ScreenNav("upload", "Cargar", Icons.Default.FolderOpen)
    object Notifications : ScreenNav("notifications", "WhatsApp", Icons.Default.NotificationsActive)
    object Benchmark : ScreenNav("benchmark", "Dataset", Icons.Default.Analytics)
    object SystemInfo : ScreenNav("info", "MLOps", Icons.Default.Memory)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetectorScreen(
    payload: ProcessedMediaPayload?,
    fusionResult: FusionResult?,
    activeScreen: ScreenNav = ScreenNav.Scanner,
    onScreenChange: (ScreenNav) -> Unit = {},
    onSelectFile: () -> Unit,
    onAnalyzeSample: (String, Int) -> Unit,
    onInspectNotification: (InterceptedNotification) -> Unit = {},
    onSimulateNotification: (MediaType, Boolean, String, String) -> Unit = { _, _, _, _ -> },
    onDismissInspectedNotification: () -> Unit = {},
    whatsAppAudios: List<WhatsAppAudioItem> = emptyList(),
    whatsAppAccessState: WhatsAppStorageAccessState = WhatsAppStorageAccessState(false, false, 0, false),
    isScanningWhatsApp: Boolean = false,
    onRefreshWhatsAppAudios: () -> Unit = {},
    onRequestAllFilesAccess: () -> Unit = {},
    onPickWhatsAppFolder: () -> Unit = {},
    onAnalyzeWhatsAppAudio: (WhatsAppAudioItem) -> Unit = {}
) {
    var isHeatmapEnabled by remember { mutableStateOf(true) }

    val navItems = listOf(
        ScreenNav.Scanner,
        ScreenNav.Upload,
        ScreenNav.Notifications,
        ScreenNav.Benchmark,
        ScreenNav.SystemInfo
    )

    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = SurfaceDark,
                tonalElevation = 8.dp
            ) {
                navItems.forEach { screen ->
                    val isSelected = activeScreen == screen
                    NavigationBarItem(
                        selected = isSelected,
                        onClick = { onScreenChange(screen) },
                        icon = {
                            Icon(
                                imageVector = screen.icon,
                                contentDescription = screen.title,
                                tint = if (isSelected) PrimaryIndigo else TextSecondary
                            )
                        },
                        label = {
                            Text(
                                text = screen.title,
                                fontSize = 10.sp,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                color = if (isSelected) PrimaryIndigo else TextSecondary
                            )
                        },
                        colors = NavigationBarItemDefaults.colors(
                            indicatorColor = PrimaryIndigo.copy(alpha = 0.15f)
                        )
                    )
                }
            }
        },
        containerColor = BackgroundDark
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(BackgroundDark)
        ) {
            when (activeScreen) {
                is ScreenNav.Scanner -> ScannerView(
                    payload = payload,
                    fusionResult = fusionResult,
                    isHeatmapEnabled = isHeatmapEnabled,
                    onToggleHeatmap = { isHeatmapEnabled = it },
                    onNavigateToUpload = { onScreenChange(ScreenNav.Upload) },
                    onAnalyzeSample = onAnalyzeSample,
                    onDismissInspectedNotification = onDismissInspectedNotification
                )
                is ScreenNav.Upload -> UploadView(
                    payload = payload,
                    onSelectFile = onSelectFile,
                    onNavigateToScanner = { onScreenChange(ScreenNav.Scanner) },
                    whatsAppAudios = whatsAppAudios,
                    whatsAppAccessState = whatsAppAccessState,
                    isScanningWhatsApp = isScanningWhatsApp,
                    onRefreshWhatsAppAudios = onRefreshWhatsAppAudios,
                    onRequestAllFilesAccess = onRequestAllFilesAccess,
                    onPickWhatsAppFolder = onPickWhatsAppFolder,
                    onAnalyzeWhatsAppAudio = onAnalyzeWhatsAppAudio
                )
                is ScreenNav.Notifications -> NotificationsView(
                    onInspectNotification = { notif ->
                        onInspectNotification(notif)
                        onScreenChange(ScreenNav.Scanner)
                    },
                    onSimulateNotification = onSimulateNotification
                )
                is ScreenNav.Benchmark -> BenchmarkView(
                    onAnalyzeSample = { type, idx ->
                        onAnalyzeSample(type, idx)
                        onScreenChange(ScreenNav.Scanner)
                    }
                )
                is ScreenNav.SystemInfo -> SystemInfoView()
            }
        }
    }
}

@Composable
fun ScannerView(
    payload: ProcessedMediaPayload?,
    fusionResult: FusionResult?,
    isHeatmapEnabled: Boolean,
    onToggleHeatmap: (Boolean) -> Unit,
    onNavigateToUpload: () -> Unit,
    onAnalyzeSample: (String, Int) -> Unit = { _, _ -> },
    onDismissInspectedNotification: () -> Unit = {}
) {
    val context = LocalContext.current
    val blockedSenders by NotificationRepository.blockedSenders.collectAsState()

    val cleanSenderName = remember(payload?.filename) {
        val fn = payload?.filename ?: ""
        when {
            fn.contains(": ") -> {
                val afterColon = fn.substringAfter(": ")
                if (afterColon.contains(" - ")) afterColon.substringBefore(" - ").trim() else afterColon.trim()
            }
            fn.isNotBlank() && !fn.startsWith("Muestra ") -> {
                val base = fn.substringBeforeLast(".")
                if (base.length > 28) base.take(28) + "..." else base
            }
            fn.startsWith("Muestra ") -> {
                "Remitente Muestra Oficial"
            }
            else -> "Remitente Desconocido"
        }
    }

    val isSenderBlocked = remember(blockedSenders, cleanSenderName) {
        NotificationRepository.isSenderBlocked(cleanSenderName)
    }

    var showBlockConfirmDialog by remember { mutableStateOf(false) }
    var showUnblockConfirmDialog by remember { mutableStateOf(false) }
    var isAlertDismissed by remember(payload) { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Detector Preventor",
                    color = TextPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Protección Multimodal Edge AI contra Fraude",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Spacer(modifier = Modifier.width(8.dp))

            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .background(AccentCyan.copy(alpha = 0.15f), RoundedCornerShape(20.dp))
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                ) {
                    Text(
                        text = "Edge AI",
                        color = AccentCyan,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1
                    )
                }

                Box(
                    modifier = Modifier
                        .background(PrimaryIndigo.copy(alpha = 0.2f), RoundedCornerShape(20.dp))
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = "Offline",
                            tint = PrimaryIndigo,
                            modifier = Modifier.size(12.dp)
                        )
                        Spacer(modifier = Modifier.width(3.dp))
                        Text(
                            text = "Offline",
                            color = PrimaryIndigo,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Shield,
                        contentDescription = null,
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "Archivo Analizado Actual:",
                            color = TextSecondary,
                            fontSize = 11.sp
                        )
                        Text(
                            text = payload?.filename ?: "Esperando Selección de Archivo...",
                            color = TextPrimary,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                OutlinedButton(
                    onClick = onNavigateToUpload,
                    shape = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                    modifier = Modifier.height(36.dp)
                ) {
                    Text("Cambiar", fontSize = 11.sp, color = PrimaryIndigo)
                }
            }
        }

        if (fusionResult?.wasFaceDiscarded == true || (payload?.mediaType == MediaType.IMAGE_ONLY && payload.faceDetected == false)) {
            Spacer(modifier = Modifier.height(16.dp))
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF0F2537)),
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, AccentCyan.copy(alpha = 0.7f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Face,
                        contentDescription = null,
                        tint = AccentCyan,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "Descarte Automático por Ausencia Facial",
                            color = AccentCyan,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "BlazeFace analizó la imagen y verificó que no contiene rostros humanos. El canal visual se neutraliza automáticamente a 0.0% de riesgo.",
                            color = TextSecondary,
                            fontSize = 11.sp,
                            lineHeight = 15.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Explicabilidad XAI (Grad-CAM Heatmap)",
                color = TextPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium
            )
            Switch(
                checked = isHeatmapEnabled,
                onCheckedChange = onToggleHeatmap,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = PrimaryIndigo
                )
            )
        }

        var multimodalTab by remember(payload) { mutableStateOf(0) }

        if (payload?.mediaType == MediaType.VIDEO_MULTIMODAL && payload.faceKeyframe != null && payload.audioSpectrogram != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { multimodalTab = 0 },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (multimodalTab == 0) PrimaryIndigo else SurfaceDark
                    ),
                    shape = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    modifier = Modifier.weight(1f).height(34.dp)
                ) {
                    Text(
                        "Fotograma Video (Visión)",
                        fontSize = 11.sp,
                        color = if (multimodalTab == 0) Color.White else TextSecondary
                    )
                }
                Button(
                    onClick = { multimodalTab = 1 },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (multimodalTab == 1) PrimaryIndigo else SurfaceDark
                    ),
                    shape = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    modifier = Modifier.weight(1f).height(34.dp)
                ) {
                    Text(
                        "Espectrograma Audio (Voz)",
                        fontSize = 11.sp,
                        color = if (multimodalTab == 1) Color.White else TextSecondary
                    )
                }
            }
        }

        val previewBitmap = if (payload?.mediaType == MediaType.VIDEO_MULTIMODAL && payload.faceKeyframe != null && payload.audioSpectrogram != null) {
            if (multimodalTab == 0) payload.faceKeyframe else payload.audioSpectrogram
        } else {
            payload?.faceKeyframe ?: payload?.audioSpectrogram
        }
        val riskFactor = (fusionResult?.globalRiskPercentage ?: 50f) / 100f

        val currentGradCam = if (payload?.mediaType == MediaType.VIDEO_MULTIMODAL) {
            if (multimodalTab == 0) payload.gradCamBitmap else (payload.audioGradCamBitmap ?: payload.gradCamBitmap)
        } else if (payload?.mediaType == MediaType.AUDIO_ONLY) {
            payload.audioGradCamBitmap ?: payload.gradCamBitmap
        } else {
            payload?.gradCamBitmap
        }

        HeatmapOverlay(
            bitmap = previewBitmap,
            gradCamBitmap = currentGradCam,
            isHeatmapActive = isHeatmapEnabled,
            riskFactor = riskFactor
        )

        Spacer(modifier = Modifier.height(16.dp))

        if (fusionResult != null) {
            RiskIndicatorCard(fusionResult = fusionResult)
        } else {
            Card(
                colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 12.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Selecciona una muestra del Dataset o Carga un archivo para iniciar la inferencia Edge AI.",
                        color = TextSecondary,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Normal
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Bolt,
                        contentDescription = null,
                        tint = AccentCyan,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Pruebas Rápidas (Inferencia TFLite)",
                        color = TextPrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Prueba el rendimiento del modelo sobre muestras reales del dataset:",
                    color = TextSecondary,
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { onAnalyzeSample("audio", 1) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("Audio Real", fontSize = 11.sp, color = RiskLowGreen, fontWeight = FontWeight.Bold)
                    }
                    OutlinedButton(
                        onClick = { onAnalyzeSample("audio", 2) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("Audio Clonado", fontSize = 11.sp, color = RiskHighRed, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(modifier = Modifier.height(6.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { onAnalyzeSample("image", 1) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("Rostro Real", fontSize = 11.sp, color = RiskLowGreen, fontWeight = FontWeight.Bold)
                    }
                    OutlinedButton(
                        onClick = { onAnalyzeSample("image", 2) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("FaceSwap IA", fontSize = 11.sp, color = RiskHighRed, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(modifier = Modifier.height(6.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { onAnalyzeSample("video", 1) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("Video Real", fontSize = 11.sp, color = RiskLowGreen, fontWeight = FontWeight.Bold)
                    }
                    OutlinedButton(
                        onClick = { onAnalyzeSample("video", 2) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                    ) {
                        Text("Video Deepfake", fontSize = 11.sp, color = RiskHighRed, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (isSenderBlocked) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1018)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, RiskHighRed.copy(alpha = 0.8f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 14.dp)
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Shield,
                        contentDescription = null,
                        tint = RiskHighRed,
                        modifier = Modifier.size(26.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = "REMITENTE EN LISTA NEGRA",
                                color = RiskHighRed,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Black,
                                letterSpacing = 0.5.sp
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Box(
                                modifier = Modifier
                                    .background(RiskHighRed.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            ) {
                                Text("Aislado", color = RiskHighRed, fontSize = 9.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                        Spacer(modifier = Modifier.height(3.dp))
                        Text(
                            text = "'$cleanSenderName' está bloqueado en este dispositivo. Sus mensajes, audios y llamadas quedan silenciados y aislados preventivamente por el motor Edge AI.",
                            color = TextSecondary,
                            fontSize = 11.sp,
                            lineHeight = 15.sp
                        )
                    }
                }
            }
        }

        if (isAlertDismissed && !isSenderBlocked) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF131D2A)),
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(1.dp, Color(0xFF334155)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 14.dp)
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = RiskLowGreen,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "Alerta pericial descartada para esta sesión de análisis.",
                        color = TextSecondary,
                        fontSize = 11.sp
                    )
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (isSenderBlocked) {
                OutlinedButton(
                    onClick = { showUnblockConfirmDialog = true },
                    colors = ButtonDefaults.outlinedButtonColors(containerColor = Color(0xFF1E293B)),
                    border = BorderStroke(1.dp, RiskHighRed.copy(alpha = 0.6f)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.LockOpen,
                        contentDescription = "Desbloquear",
                        tint = AccentCyan,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Desbloquear",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            } else {
                Button(
                    onClick = { showBlockConfirmDialog = true },
                    colors = ButtonDefaults.buttonColors(containerColor = RiskHighRed),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Block,
                        contentDescription = "Bloquear",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Bloquear Remitente",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            OutlinedButton(
                onClick = {
                    isAlertDismissed = true
                    onDismissInspectedNotification()
                    Toast.makeText(context, "Muestra descartada y removida del historial interceptado.", Toast.LENGTH_SHORT).show()
                },
                colors = ButtonDefaults.outlinedButtonColors(containerColor = ButtonGray),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(48.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = "Descartar",
                    tint = Color.White,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "Descartar",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        if (showBlockConfirmDialog) {
            AlertDialog(
                onDismissRequest = { showBlockConfirmDialog = false },
                icon = {
                    Icon(
                        imageVector = Icons.Default.Shield,
                        contentDescription = null,
                        tint = RiskHighRed,
                        modifier = Modifier.size(34.dp)
                    )
                },
                title = {
                    Text(
                        text = "Bloqueo y Aislamiento Preventivo",
                        color = TextPrimary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                },
                text = {
                    Column {
                        Text(
                            text = "¿Confirmas el bloqueo pericial del contacto:",
                            color = TextSecondary,
                            fontSize = 12.sp
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = cleanSenderName,
                            color = RiskHighRed,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Acciones automáticas del Escudo Edge AI:\n" +
                                   "• Registro en Lista Negra persistente en hardware local.\n" +
                                   "• Silenciamiento y neutralización de futuros deepfakes y audios de este remitente.\n" +
                                   "• Privacidad por Diseño: no se envía información a servidores externos.",
                            color = TextSecondary,
                            fontSize = 11.sp,
                            lineHeight = 16.sp
                        )
                    }
                },
                confirmButton = {
                    Button(
                        onClick = {
                            NotificationRepository.blockSender(cleanSenderName)
                            showBlockConfirmDialog = false
                            Toast.makeText(
                                context,
                                "Remitente '$cleanSenderName' bloqueado y aislado localmente.",
                                Toast.LENGTH_LONG
                            ).show()
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = RiskHighRed),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Confirmar Bloqueo", color = Color.White, fontWeight = FontWeight.Bold)
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = { showBlockConfirmDialog = false }
                    ) {
                        Text("Cancelar", color = TextSecondary)
                    }
                },
                containerColor = SurfaceDark,
                shape = RoundedCornerShape(16.dp)
            )
        }

        if (showUnblockConfirmDialog) {
            AlertDialog(
                onDismissRequest = { showUnblockConfirmDialog = false },
                icon = {
                    Icon(
                        imageVector = Icons.Default.LockOpen,
                        contentDescription = null,
                        tint = AccentCyan,
                        modifier = Modifier.size(34.dp)
                    )
                },
                title = {
                    Text(
                        text = "Desbloquear Remitente",
                        color = TextPrimary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                },
                text = {
                    Text(
                        text = "¿Deseas remover a '$cleanSenderName' de la Lista Negra Local? Sus mensajes futuros volverán a ser analizados normalmente.",
                        color = TextSecondary,
                        fontSize = 12.sp
                    )
                },
                confirmButton = {
                    Button(
                        onClick = {
                            NotificationRepository.unblockSender(cleanSenderName)
                            showUnblockConfirmDialog = false
                            Toast.makeText(
                                context,
                                "Remitente '$cleanSenderName' desbloqueado con éxito.",
                                Toast.LENGTH_SHORT
                            ).show()
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Confirmar Desbloqueo", color = Color.White, fontWeight = FontWeight.Bold)
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = { showUnblockConfirmDialog = false }
                    ) {
                        Text("Cancelar", color = TextSecondary)
                    }
                },
                containerColor = SurfaceDark,
                shape = RoundedCornerShape(16.dp)
            )
        }
    }
}

@Composable
fun UploadView(
    payload: ProcessedMediaPayload?,
    onSelectFile: () -> Unit,
    onNavigateToScanner: () -> Unit,
    whatsAppAudios: List<WhatsAppAudioItem> = emptyList(),
    whatsAppAccessState: WhatsAppStorageAccessState = WhatsAppStorageAccessState(false, false, 0, false),
    isScanningWhatsApp: Boolean = false,
    onRefreshWhatsAppAudios: () -> Unit = {},
    onRequestAllFilesAccess: () -> Unit = {},
    onPickWhatsAppFolder: () -> Unit = {},
    onAnalyzeWhatsAppAudio: (WhatsAppAudioItem) -> Unit = {}
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Carga Libre y Audios de WhatsApp",
            color = TextPrimary,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Inspecciona notas de voz (.opus) de WhatsApp o archivos almacenados en tu telefono",
            color = TextSecondary,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(20.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onSelectFile() }
                .border(1.dp, PrimaryIndigo.copy(alpha = 0.5f), RoundedCornerShape(16.dp))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .background(PrimaryIndigo.copy(alpha = 0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.FolderOpen,
                        contentDescription = "Upload",
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(28.dp)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "Toca aqui para seleccionar un archivo",
                    color = TextPrimary,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Formatos soportados: .opus, .mp3, .wav, .jpg, .png, .mp4, .avi",
                    color = TextSecondary,
                    fontSize = 12.sp
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = onSelectFile,
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text("Abrir Galeria / Archivos", fontSize = 12.sp, color = Color.White)
                }
            }
        }

        if (payload != null) {
            Spacer(modifier = Modifier.height(14.dp))
            Button(
                onClick = onNavigateToScanner,
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(46.dp)
            ) {
                Text("Ver Diagnostico de '${payload.filename}'", color = Color.White, fontSize = 13.sp)
            }
        }

        Spacer(modifier = Modifier.height(18.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, AccentCyan.copy(alpha = 0.35f), RoundedCornerShape(14.dp))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(
                            imageVector = Icons.Default.GraphicEq,
                            contentDescription = null,
                            tint = AccentCyan,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Audios y Notas de Voz WhatsApp (.opus)",
                                color = TextPrimary,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = if (isScanningWhatsApp) "Escaneando almacenamiento..." else "${whatsAppAudios.size} archivos detectados",
                                color = AccentCyan,
                                fontSize = 11.sp
                            )
                        }
                    }

                    IconButton(onClick = onRefreshWhatsAppAudios) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Actualizar audios de WhatsApp",
                            tint = AccentCyan
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                        .padding(10.dp)
                ) {
                    Column {
                        Text(
                            text = "Ruta WhatsApp (Android 11+): Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes",
                            color = TextSecondary,
                            fontSize = 10.sp,
                            lineHeight = 14.sp
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        val accessSummary = buildString {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                                append(if (whatsAppAccessState.hasAllFilesAccess) "Acceso Total: Concedido" else "Acceso Total: Requerido para omitir .nomedia")
                            } else {
                                append(if (whatsAppAccessState.hasAudioMediaPermission) "Permiso Audio: Concedido" else "Permiso Audio: Requerido")
                            }
                            append(" | Carpetas SAF: ${whatsAppAccessState.persistedTreeCount}")
                        }
                        Text(
                            text = accessSummary,
                            color = if (whatsAppAccessState.hasAllFilesAccess || whatsAppAccessState.persistedTreeCount > 0) RiskLowGreen else Color(0xFFEAB308),
                            fontSize = 10.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !whatsAppAccessState.hasAllFilesAccess) {
                        Button(
                            onClick = onRequestAllFilesAccess,
                            colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Security,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Acceso Total", fontSize = 11.sp, color = Color.White)
                        }
                    }

                    OutlinedButton(
                        onClick = onPickWhatsAppFolder,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(
                            imageVector = Icons.Default.FolderOpen,
                            contentDescription = null,
                            tint = AccentCyan,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Elegir Carpeta", fontSize = 11.sp, color = AccentCyan)
                    }

                    OutlinedButton(
                        onClick = onRefreshWhatsAppAudios,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier.weight(0.8f)
                    ) {
                        Text("Escanear", fontSize = 11.sp, color = TextPrimary)
                    }
                }

                if (isScanningWhatsApp) {
                    Spacer(modifier = Modifier.height(10.dp))
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth(),
                        color = AccentCyan,
                        trackColor = Color(0xFF1E293B)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                if (!isScanningWhatsApp && whatsAppAudios.isEmpty()) {
                    Text(
                        text = "No se detectaron notas de voz accesibles todavia. Pulsa 'Acceso Total' o selecciona la carpeta 'WhatsApp Voice Notes' con 'Elegir Carpeta' para listar tus archivos .opus locales.",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        lineHeight = 15.sp
                    )
                } else {
                    whatsAppAudios.forEach { item ->
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                                .border(1.dp, CardBorder, RoundedCornerShape(10.dp))
                                .clickable { onAnalyzeWhatsAppAudio(item) }
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Box(
                                            modifier = Modifier
                                                .background(PrimaryIndigo.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
                                                .padding(horizontal = 6.dp, vertical = 2.dp)
                                        ) {
                                            Text(
                                                text = item.categoryLabel,
                                                color = AccentCyan,
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = item.sourceKind,
                                            color = TextSecondary,
                                            fontSize = 9.sp
                                        )
                                    }
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = item.displayName,
                                        color = TextPrimary,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        maxLines = 1
                                    )
                                    Text(
                                        text = "${item.folderHint} | ${item.formattedDate} | ${item.formattedSize}",
                                        color = TextSecondary,
                                        fontSize = 10.sp,
                                        maxLines = 1
                                    )
                                }

                                Spacer(modifier = Modifier.width(8.dp))

                                Button(
                                    onClick = { onAnalyzeWhatsAppAudio(item) },
                                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                    modifier = Modifier.height(32.dp)
                                ) {
                                    Text("Analizar", fontSize = 10.sp, color = Color.White)
                                }
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = null,
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "Integracion Directa con WhatsApp / Telegram",
                        color = TextPrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Tambien puedes analizar cualquier audio o video recibido en WhatsApp manteniendo presionado el archivo y seleccionando 'Compartir -> Detector Preventor'.",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }
        }
    }
}

@Composable
fun BenchmarkView(
    onAnalyzeSample: (String, Int) -> Unit
) {
    var selectedCategory by remember { mutableStateOf("audio") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text(
            text = "Dataset de Prueba Académico",
            color = TextPrimary,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "5 Muestras de prueba pre-evaluadas por cada tipo de detector",
            color = TextSecondary,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FilterChip(
                selected = selectedCategory == "audio",
                onClick = { selectedCategory = "audio" },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.GraphicEq,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = if (selectedCategory == "audio") Color.White else PrimaryIndigo
                    )
                },
                label = { Text("Audio (5)", fontSize = 12.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = PrimaryIndigo,
                    selectedLabelColor = Color.White
                ),
                modifier = Modifier.weight(1f)
            )
            FilterChip(
                selected = selectedCategory == "image",
                onClick = { selectedCategory = "image" },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Image,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = if (selectedCategory == "image") Color.White else PrimaryIndigo
                    )
                },
                label = { Text("Imagen (5)", fontSize = 12.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = PrimaryIndigo,
                    selectedLabelColor = Color.White
                ),
                modifier = Modifier.weight(1f)
            )
            FilterChip(
                selected = selectedCategory == "video",
                onClick = { selectedCategory = "video" },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Videocam,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = if (selectedCategory == "video") Color.White else PrimaryIndigo
                    )
                },
                label = { Text("Video (5)", fontSize = 12.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = PrimaryIndigo,
                    selectedLabelColor = Color.White
                ),
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        val samples = when (selectedCategory) {
            "audio" -> listOf(
                Pair("Muestra 1: Voz Humana Real (ASVspoof Bonafide)", "Audios limpios de voz humana real. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)"),
                Pair("Muestra 2: Clonación por IA (ASVspoof Deepfake)", "Voz clonada mediante modelo neuronal sintético. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 3: Síntesis de Texto a Voz (TTS AI)", "Audio generado por motor de texto a voz. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 4: Nota de Voz WhatsApp Replicada", "Simulación de estafa por nota de voz en WhatsApp. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 5: Conversación Telefónica Auténtica", "Grabación bonafide de llamada de voz. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)")
            )
            "image" -> listOf(
                Pair("Muestra 1: Retrato Real Prístino (FF++)", "Rostro real extraído de FaceForensics++. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)"),
                Pair("Muestra 2: Rostro Sintético Deepfake AI", "Manipulación facial detectada por red neuronal. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 3: Alteración Facial Moderada (GAN)", "Indicadores intermedios de generación sintética. Categoria de referencia: RIESGO MEDIO (inferencia TFLite en vivo)"),
                Pair("Muestra 4: Reenactamiento Sutil (Face2Face)", "Alteración sutil en rasgos locales de expresión. Categoria de referencia: RIESGO MEDIO (inferencia TFLite en vivo)"),
                Pair("Muestra 5: Fotografía HD Original", "Fotografía prístina de alta resolución. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)")
            )
            else -> listOf(
                Pair("Muestra 1: Entrevista Real YouTube (FF++)", "Video original prístino sin alteraciones. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)"),
                Pair("Muestra 2: Sincronización Labial LipSync AI", "Video con movimiento de labios alterado por IA. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 3: FaceSwap HD Video", "Secuencia de video con rostro sustituido. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 4: Avatar IA Multimodal Completo", "Video sintético con audio y rostro generados. Categoria de referencia: ALTO RIESGO (inferencia TFLite en vivo)"),
                Pair("Muestra 5: Clip de Cámara Frontal Auténtico", "Video grabado directamente con cámara frontal. Categoria de referencia: RIESGO BAJO (inferencia TFLite en vivo)")
            )
        }

        samples.forEachIndexed { idx, item ->
            val isHighRisk = item.second.contains("ALTO RIESGO")
            val isMediumRisk = !isHighRisk && item.second.contains("RIESGO MEDIO")
            val sampleBadgeColor = when {
                isHighRisk -> RiskHighRed
                isMediumRisk -> RiskMediumYellow
                else -> RiskLowGreen
            }
            val sampleBadgeText = when {
                isHighRisk -> "AMENAZA"
                isMediumRisk -> "SOSPECHOSO"
                else -> "AUTÉNTICO"
            }

            Card(
                colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .border(1.dp, CardBorder, RoundedCornerShape(12.dp))
                    .clickable { onAnalyzeSample(selectedCategory, idx + 1) }
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .background(sampleBadgeColor.copy(alpha = 0.15f), RoundedCornerShape(4.dp))
                                    .border(1.dp, sampleBadgeColor.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = sampleBadgeText,
                                    color = sampleBadgeColor,
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = item.first,
                                color = TextPrimary,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = item.second,
                            color = TextSecondary,
                            fontSize = 11.sp
                        )
                    }

                    Spacer(modifier = Modifier.width(10.dp))

                    Button(
                        onClick = { onAnalyzeSample(selectedCategory, idx + 1) },
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.PlayArrow,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Analizar", fontSize = 11.sp, color = Color.White)
                    }
                }
            }
        }
    }
}

@Composable
fun SystemInfoView() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, PrimaryIndigo.copy(alpha = 0.5f), RoundedCornerShape(14.dp))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "UNIVERSIDAD MAYOR REAL Y PONTIFICIA DE SAN FRANCISCO XAVIER DE CHUQUISACA",
                    color = AccentCyan,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    lineHeight = 15.sp
                )
                Text(
                    text = "Facultad de Ciencias y Tecnología",
                    color = TextPrimary,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "Carrera de Ingeniería en Ciencias de la Computación",
                    color = TextSecondary,
                    fontSize = 12.sp
                )

                Spacer(modifier = Modifier.height(10.dp))
                HorizontalDivider(color = PrimaryIndigo.copy(alpha = 0.3f), thickness = 1.dp)
                Spacer(modifier = Modifier.height(10.dp))

                Text(
                    text = "Detector Móvil Multimodal de Estafas Digitales mediante IA Explicable (XAI) y Procesamiento en el Borde (Edge AI)",
                    color = TextPrimary,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    lineHeight = 20.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("• Materia: SIS 330 (Desarrollo de Aplicaciones Inteligentes) - Grupo 1", fontSize = 11.sp, color = TextSecondary)
                Text("• Universitario: Romero Morales Jhojan Erick", fontSize = 11.sp, color = AccentCyan, fontWeight = FontWeight.SemiBold)
                Text("• Docente: Ing. Pacheco Lora Carlos Walter", fontSize = 11.sp, color = TextSecondary)
                Text("• Gestión: 2026 | Sucre – Bolivia", fontSize = 11.sp, color = TextSecondary)
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Memory,
                        contentDescription = null,
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Enjambre de Modelos Expertos (Edge AI TFLite FP16 + XNNPACK)",
                        color = PrimaryIndigo,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Experto Audio: EfficientNetV2-B0 14x14 FP16 (11.27 MB)", fontSize = 12.sp, color = TextPrimary, fontWeight = FontWeight.SemiBold)
                Text("  Entrada: Espectrograma Mel multiescala RGB 224x224x3 (RMS-VAD + 1024/256 FFT + CMVN).", fontSize = 11.sp, color = TextSecondary)
                Text("  Dataset: 47,200 clips train / 29,741 externos (ASVspoof19/21/5, BRSpeech, Opus, Unseen).", fontSize = 11.sp, color = TextSecondary)
                Text("  Rendimiento: 84.17% Acc | 95.95% Prec | 75.69% RecFake | 95.68% SpecReal | 0.9197 AUC.", fontSize = 11.sp, color = AccentCyan)

                Spacer(modifier = Modifier.height(8.dp))
                Text("• Experto Visión: EfficientNet-B0 FP16 (7.74 MB)", fontSize = 12.sp, color = TextPrimary, fontWeight = FontWeight.SemiBold)
                Text("  Entrada: Keyframe facial BlazeFace con margen pericial del 15% (224x224x3).", fontSize = 11.sp, color = TextSecondary)
                Text("  Dataset: 30,750 muestras externas (StyleGAN 140k, SDXL vs FFHQ, FaceForensics++).", fontSize = 11.sp, color = TextSecondary)
                Text("  Rendimiento: 97.66% Acc | 97.55% Prec | 97.76% Recall | 97.66% F1 | 0.9934 AUC.", fontSize = 11.sp, color = AccentCyan)

                Spacer(modifier = Modifier.height(8.dp))
                Text("• Orquestador Multimodal: BlazeFace + Extracción Mel paralela (Fusión 97.47% Acc / 0.9953 AUC).", fontSize = 11.sp, color = TextSecondary)
                Text("• Capa XAI: Grad-CAM 2D desde audio_gradcam_weights.bin (14x14x576) y gradcam_weights.bin (1280).", fontSize = 11.sp, color = TextSecondary)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Analytics,
                        contentDescription = null,
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Fusión Tardía (Score-Level Fusion) & Reglas",
                        color = PrimaryIndigo,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Ponderación Multimodal: 55% Audio (w_a = 0.55) + 45% Visión (w_v = 0.45)", fontSize = 12.sp, color = TextPrimary)
                Text("• Regla de 3/5 en Video: Alerta crítica ante clonación de audio o mayoría de frames manipulados.", fontSize = 11.sp, color = TextSecondary)
                Text("• Rebalanceo Unimodal: Adaptación dinámica al 100% cuando solo existe un canal.", fontSize = 11.sp, color = TextSecondary)
                Text("• Descarte Automático sin Rostro: Neutralización visual a 0.0% ante ausencia facial.", fontSize = 11.sp, color = TextSecondary)
                Text("• Umbrales de Riesgo: Bajo (<30%), Medio (30%-69%), Alto (>=70%).", fontSize = 11.sp, color = TextSecondary)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Shield,
                        contentDescription = null,
                        tint = AccentCyan,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Métricas de Perfilado Edge (Android Profiler)",
                        color = AccentCyan,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Delta de Memoria RAM TFLite + XNNPACK: +103.4 MB (pico proceso medido 470.2 MB)", fontSize = 11.sp, color = TextPrimary)
                Text("• Latencia Media TFLite CPU (XNNPACK 4 hilos): Audio 1.8 ms | Visión 7.5 ms | Par 9.3 ms", fontSize = 11.sp, color = TextPrimary)
                Text("• Uso de CPU Medido: 1.45% promedio / 8.68% pico con delegado XNNPACK estable (4 hilos)", fontSize = 11.sp, color = TextPrimary)
                Text("• Impacto Energético: Ligero / Medio, previniendo sobrecalentamiento", fontSize = 11.sp, color = TextPrimary)
                Text("• Privacidad por Diseño: Procesamiento 100% local sin servidores externos (ISO/IEC 27400 y NIST)", fontSize = 11.sp, color = TextPrimary)
            }
        }
    }
}

@Composable
fun NotificationsView(
    onInspectNotification: (InterceptedNotification) -> Unit,
    onSimulateNotification: (MediaType, Boolean, String, String) -> Unit = { _, _, _, _ -> }
) {
    val context = LocalContext.current
    val notifications by NotificationRepository.notifications.collectAsState()
    val isMonitoringActive by NotificationRepository.isMonitoringActive.collectAsState()
    val isServiceConnected by NotificationRepository.isServiceConnected.collectAsState()
    var isPermissionGranted by remember { mutableStateOf(NotificationRepository.isPermissionGranted(context)) }

    LaunchedEffect(Unit) {
        NotificationRepository.autoResolvePendingNotifications(context)
    }

    DisposableEffect(Unit) {
        isPermissionGranted = NotificationRepository.isPermissionGranted(context)
        onDispose { }
    }

    val (statusLabel, statusColor) = when {
        !isPermissionGranted -> Pair("PERMISO REQUERIDO", RiskHighRed)
        !isMonitoringActive -> Pair("MONITOREO EN PAUSA", Color(0xFF94A3B8))
        isServiceConnected -> Pair("ACTIVO Y ESCUCHANDO", RiskLowGreen)
        else -> Pair("LISTO PARA INTERCEPTAR", AccentCyan)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Monitoreo WhatsApp",
                    color = TextPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Interceptacion y Analisis Automatico de Mensajeria",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Box(
                modifier = Modifier
                    .background(
                        statusColor.copy(alpha = 0.2f),
                        RoundedCornerShape(20.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                statusColor,
                                CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = statusLabel,
                        color = statusColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Escucha de Notificaciones en Segundo Plano",
                            color = TextPrimary,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Intercepta notas de voz (.opus, .mp3) y fotos de WhatsApp o Telegram para verificarlas con Edge AI.",
                            color = TextSecondary,
                            fontSize = 11.sp,
                            lineHeight = 15.sp
                        )
                    }

                    Switch(
                        checked = isMonitoringActive,
                        onCheckedChange = { NotificationRepository.setMonitoringActive(it) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = PrimaryIndigo
                        )
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                if (!isPermissionGranted) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(RiskHighRed.copy(alpha = 0.15f), RoundedCornerShape(8.dp))
                            .border(1.dp, RiskHighRed.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                            .padding(10.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, contentDescription = null, tint = RiskHighRed, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Permiso de Android no otorgado. La aplicacion no puede leer notificaciones hasta habilitarlo.",
                                color = TextPrimary,
                                fontSize = 11.sp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                } else if (!isServiceConnected) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFEAB308).copy(alpha = 0.15f), RoundedCornerShape(8.dp))
                            .border(1.dp, Color(0xFFEAB308).copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                            .padding(10.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFEAB308), modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "El sistema Android desvinculo el servicio en reposo. Pulsa 'Reconectar Servicio' para reactivarlo.",
                                color = TextPrimary,
                                fontSize = 11.sp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                }

                Button(
                    onClick = {
                        NotificationMonitorService.ensureServiceBound(context)
                        isPermissionGranted = NotificationRepository.isPermissionGranted(context)
                        Toast.makeText(context, "Re-vinculando servicio de escucha con el SO...", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Reconectar Servicio con el SO", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedButton(
                    onClick = {
                        try {
                            val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                            context.startActivity(intent)
                        } catch (e: Exception) {
                            Toast.makeText(context, "Abre Ajustes > Acceso a notificaciones", Toast.LENGTH_SHORT).show()
                        }
                    },
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = null,
                        tint = AccentCyan,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Abrir Ajustes de Acceso a Notificaciones", fontSize = 12.sp, color = AccentCyan)
                }

                Spacer(modifier = Modifier.height(10.dp))

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                        .padding(10.dp)
                ) {
                    Column {
                        Text(
                            text = "Ajuste para Xiaomi / HyperOS / Samsung:",
                            color = AccentCyan,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = "En Ajustes > Apps > Detector Preventor > Bateria, elige 'Sin restricciones' e 'Inicio automatico' para impedir que el sistema suspenda el escuchador en reposo.",
                            color = TextSecondary,
                            fontSize = 10.sp,
                            lineHeight = 14.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Security,
                        contentDescription = null,
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Simulador de Amenazas de Fraude",
                        color = PrimaryIndigo,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Prueba la reacción del sistema simulando la llegada de mensajes multimedia:",
                    color = TextSecondary,
                    fontSize = 11.sp
                )

                Spacer(modifier = Modifier.height(12.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.AUDIO_ONLY,
                            true,
                            "Mamá (Urgente)",
                            "Hijo, perdí mi tarjeta, hazme un depósito rápido por favor (Nota de voz 0:14)"
                        )
                        Toast.makeText(context, "Audio sospechoso interceptado y procesado por TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = RiskHighRed.copy(alpha = 0.85f)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Warning, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Audio de Voz Manipulado (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.IMAGE_ONLY,
                            true,
                            "Número Desconocido (+591 ...)",
                            "Mira esta foto tuya que encontré en redes (foto_comprometedora.jpg)"
                        )
                        Toast.makeText(context, "Imagen alterada interceptada y procesada por TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Image, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Imagen Facial Alterada (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.VIDEO_MULTIMODAL,
                            true,
                            "Contacto Sospechoso (+34 ...)",
                            "Mira el video que me enviaron, urge que lo veas (video_01.mp4)"
                        )
                        Toast.makeText(context, "Video multimodal interceptado y procesado por TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDC2626)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Videocam, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Video Multimodal Inconsistente (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.AUDIO_ONLY,
                            false,
                            "Carlos Amigo",
                            "Hola hermano, nos vemos a las 5pm en la facultad (Nota de voz 0:09)"
                        )
                        Toast.makeText(context, "Audio legítimo autenticado por modelo de audio TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF059669)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Audio de Voz Auténtico (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.IMAGE_ONLY,
                            false,
                            "Hermana",
                            "Foto familiar del almuerzo (foto_original.jpg)"
                        )
                        Toast.makeText(context, "Fotografía legítima autenticada por modelo de visión TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0D9488)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Fotografía Facial Auténtica (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.IMAGE_ONLY,
                            false,
                            "Contacto WhatsApp",
                            "Paisaje natural de montaña sin personas (descarte_blazeface.jpg)"
                        )
                        Toast.makeText(context, "Foto sin rostro enviada a BlazeFace (Descarte automático a 0.0% de riesgo)", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0284C7)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.FilterVintage, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Foto Sin Rostro Humano (Descarte BlazeFace)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSimulateNotification(
                            MediaType.VIDEO_MULTIMODAL,
                            false,
                            "Primo",
                            "Video familiar del cumpleaños (video_autentico.mp4)"
                        )
                        Toast.makeText(context, "Video auténtico autenticado por modelos de visión y audio TFLite", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF047857)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Simular Video Multimodal Auténtico (Inferencia TFLite)", fontSize = 12.sp, color = Color.White)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Historial Interceptado (${notifications.size})",
                color = TextPrimary,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )

            Row(verticalAlignment = Alignment.CenterVertically) {
                val hasPending = notifications.any { it.isPendingDownload() }
                if (hasPending) {
                    TextButton(
                        onClick = {
                            NotificationRepository.autoResolvePendingNotifications(context)
                            Toast.makeText(context, "Resolviendo descargas con Edge AI TFLite...", Toast.LENGTH_SHORT).show()
                        }
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null, tint = AccentCyan, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Resolver Descargas", color = AccentCyan, fontSize = 12.sp)
                    }
                    Spacer(modifier = Modifier.width(4.dp))
                }

                if (notifications.isNotEmpty()) {
                    TextButton(
                        onClick = { NotificationRepository.clearNotifications() }
                    ) {
                        Text("Limpiar", color = TextSecondary, fontSize = 12.sp)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        val blockedSet by NotificationRepository.blockedSenders.collectAsState()
        if (blockedSet.isNotEmpty()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1018)),
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(1.dp, RiskHighRed.copy(alpha = 0.5f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Shield,
                            contentDescription = null,
                            tint = RiskHighRed,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Lista Negra Local: ${blockedSet.size} contacto(s) aislado(s)",
                            color = RiskHighRed,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    TextButton(
                        onClick = {
                            NotificationRepository.clearBlockedSenders()
                            Toast.makeText(context, "Lista negra local restablecida.", Toast.LENGTH_SHORT).show()
                        },
                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text("Restablecer", color = AccentCyan, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        if (notifications.isEmpty()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.MarkChatRead,
                        contentDescription = null,
                        tint = TextSecondary,
                        modifier = Modifier.size(40.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Sin alertas pendientes",
                        color = TextPrimary,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Usa el simulador superior o envía una nota de voz por WhatsApp para ver la intercepción en vivo.",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                }
            }
        } else {
            notifications.forEach { notif ->
                val isNotifBlocked = NotificationRepository.isSenderBlocked(notif.sender)
                val badgeColor = if (notif.riskScore >= 70f) RiskHighRed else if (notif.riskScore >= 30f) RiskMediumYellow else RiskLowGreen
                Card(
                    colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                        .border(1.dp, badgeColor.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                        .clickable { onInspectNotification(notif) }
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .background(PrimaryIndigo.copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = notif.appName,
                                        color = AccentCyan,
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = notif.sender,
                                    color = TextPrimary,
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                if (isNotifBlocked) {
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Box(
                                        modifier = Modifier
                                            .background(RiskHighRed.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
                                            .border(1.dp, RiskHighRed.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
                                            .padding(horizontal = 5.dp, vertical = 2.dp)
                                    ) {
                                        Text(
                                            text = "BLOQUEADO",
                                            color = RiskHighRed,
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Black
                                        )
                                    }
                                }
                            }

                            Text(
                                text = notif.formattedTime,
                                color = TextSecondary,
                                fontSize = 10.sp
                            )
                        }

                        if (notif.isFaceDiscarded()) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Box(
                                modifier = Modifier
                                    .background(AccentCyan.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                    .border(1.dp, AccentCyan.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 7.dp, vertical = 3.dp)
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.Face,
                                        contentDescription = null,
                                        tint = AccentCyan,
                                        modifier = Modifier.size(13.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = "BLAZEFACE: DESCARTE FACIAL (0.0% RIESGO)",
                                        color = AccentCyan,
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        } else if (notif.mediaType == MediaType.IMAGE_ONLY || notif.mediaType == MediaType.VIDEO_MULTIMODAL) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Box(
                                modifier = Modifier
                                    .background(PrimaryIndigo.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                    .border(1.dp, PrimaryIndigo.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 7.dp, vertical = 3.dp)
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.Face,
                                        contentDescription = null,
                                        tint = Color(0xFFA5B4FC),
                                        modifier = Modifier.size(13.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = "BLAZEFACE: ROSTRO DETECTADO (ROI)",
                                        color = Color(0xFFA5B4FC),
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        } else if (notif.mediaType == MediaType.AUDIO_ONLY) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Box(
                                modifier = Modifier
                                    .background(Color(0xFF8B5CF6).copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                    .border(1.dp, Color(0xFF8B5CF6).copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 7.dp, vertical = 3.dp)
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.GraphicEq,
                                        contentDescription = null,
                                        tint = Color(0xFFC4B5FD),
                                        modifier = Modifier.size(13.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = "AUDIO PCM (VOZ)",
                                        color = Color(0xFFC4B5FD),
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(6.dp))

                        Text(
                            text = notif.text,
                            color = TextSecondary,
                            fontSize = 12.sp,
                            lineHeight = 16.sp
                        )

                        Spacer(modifier = Modifier.height(10.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    text = "Riesgo Deepfake:",
                                    color = TextSecondary,
                                    fontSize = 11.sp
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = String.format("%.1f%%", notif.riskScore),
                                    color = badgeColor,
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                IconButton(
                                    onClick = {
                                        NotificationRepository.removeNotification(notif.id)
                                        Toast.makeText(context, "Notificación descartada del historial.", Toast.LENGTH_SHORT).show()
                                    },
                                    modifier = Modifier.size(30.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.DeleteOutline,
                                        contentDescription = "Descartar del historial",
                                        tint = TextSecondary,
                                        modifier = Modifier.size(16.dp)
                                    )
                                }

                                IconButton(
                                    onClick = {
                                        if (isNotifBlocked) {
                                            NotificationRepository.unblockSender(notif.sender)
                                            Toast.makeText(context, "Remitente '${notif.sender}' desbloqueado.", Toast.LENGTH_SHORT).show()
                                        } else {
                                            NotificationRepository.blockSender(notif.sender)
                                            Toast.makeText(context, "Remitente '${notif.sender}' bloqueado en lista negra local.", Toast.LENGTH_SHORT).show()
                                        }
                                    },
                                    modifier = Modifier.size(30.dp)
                                ) {
                                    Icon(
                                        imageVector = if (isNotifBlocked) Icons.Default.LockOpen else Icons.Default.Block,
                                        contentDescription = "Bloquear o Desbloquear",
                                        tint = if (isNotifBlocked) AccentCyan else RiskHighRed,
                                        modifier = Modifier.size(16.dp)
                                    )
                                }

                                val isPendingDownload = notif.isPendingDownload()
                                Button(
                                    onClick = { onInspectNotification(notif) },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = if (isPendingDownload) Color(0xFF2563EB) else PrimaryIndigo
                                    ),
                                    shape = RoundedCornerShape(6.dp),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                    modifier = Modifier.height(30.dp)
                                ) {
                                    Icon(
                                        imageVector = if (isPendingDownload) Icons.Default.Download else Icons.Default.Visibility,
                                        contentDescription = null,
                                        tint = Color.White,
                                        modifier = Modifier.size(13.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = if (isPendingDownload) "Descargar y Analizar" else "Inspeccionar XAI",
                                        fontSize = 10.sp,
                                        color = Color.White
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

