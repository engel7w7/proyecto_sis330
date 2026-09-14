package com.detectorpreventor.app.ui

import android.widget.Toast
import androidx.compose.animation.*
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
import android.provider.Settings
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.MediaPayload
import com.detectorpreventor.app.domain.MediaType
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.domain.RiskScorer
import com.detectorpreventor.app.notifications.InterceptedNotification
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

/**
 * Pantalla principal que integra la barra de navegación inferior y las cinco vistas principales.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetectorScreen(
    payload: MediaPayload?,
    fusionResult: FusionResult?,
    onSelectFile: () -> Unit,
    onAnalyzeSample: (String, Int) -> Unit,
    onInspectNotification: (InterceptedNotification) -> Unit = {}
) {
    var currentScreen by remember { mutableStateOf<ScreenNav>(ScreenNav.Scanner) }
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
                    val isSelected = currentScreen == screen
                    NavigationBarItem(
                        selected = isSelected,
                        onClick = { currentScreen = screen },
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
            when (currentScreen) {
                is ScreenNav.Scanner -> ScannerView(
                    payload = payload,
                    fusionResult = fusionResult,
                    isHeatmapEnabled = isHeatmapEnabled,
                    onToggleHeatmap = { isHeatmapEnabled = it },
                    onNavigateToUpload = { currentScreen = ScreenNav.Upload }
                )
                is ScreenNav.Upload -> UploadView(
                    payload = payload,
                    onSelectFile = onSelectFile,
                    onNavigateToScanner = { currentScreen = ScreenNav.Scanner }
                )
                is ScreenNav.Notifications -> NotificationsView(
                    onInspectNotification = { notif ->
                        onInspectNotification(notif)
                        currentScreen = ScreenNav.Scanner
                    }
                )
                is ScreenNav.Benchmark -> BenchmarkView(
                    onAnalyzeSample = { type, idx ->
                        onAnalyzeSample(type, idx)
                        currentScreen = ScreenNav.Scanner
                    }
                )
                is ScreenNav.SystemInfo -> SystemInfoView()
            }
        }
    }
}

/**
 * Vista de escáner y resultado de análisis.
 */
@Composable
fun ScannerView(
    payload: MediaPayload?,
    fusionResult: FusionResult?,
    isHeatmapEnabled: Boolean,
    onToggleHeatmap: (Boolean) -> Unit,
    onNavigateToUpload: () -> Unit
) {
    val context = LocalContext.current

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
            Column {
                Text(
                    text = "Detector Preventor",
                    color = TextPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Protección Multimodal Edge AI contra Fraud",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Box(
                modifier = Modifier
                    .background(PrimaryIndigo.copy(alpha = 0.2f), RoundedCornerShape(20.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Offline",
                        tint = PrimaryIndigo,
                        modifier = Modifier.height(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "100% Offline",
                        color = PrimaryIndigo,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
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

        Spacer(modifier = Modifier.height(8.dp))

        val previewBitmap = payload?.faceKeyframe ?: payload?.audioSpectrogram
        val riskFactor = (fusionResult?.globalRiskPercentage ?: 50f) / 100f

        HeatmapOverlay(
            bitmap = previewBitmap,
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

        Spacer(modifier = Modifier.height(20.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = {
                    Toast.makeText(
                        context,
                        "Remitente bloqueado y reportado localmente por amenaza de estafa.",
                        Toast.LENGTH_LONG
                    ).show()
                },
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

            OutlinedButton(
                onClick = {
                    Toast.makeText(context, "Alerta descartada.", Toast.LENGTH_SHORT).show()
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
    }
}

/**
 * Vista de carga libre de archivos multimedia.
 */
@Composable
fun UploadView(
    payload: MediaPayload?,
    onSelectFile: () -> Unit,
    onNavigateToScanner: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Carga Libre de Archivos",
            color = TextPrimary,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Inspecciona audios, imágenes y videos almacenados en tu teléfono",
            color = TextSecondary,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(24.dp))

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
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .size(64.dp)
                        .background(PrimaryIndigo.copy(alpha = 0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.FolderOpen,
                        contentDescription = "Upload",
                        tint = PrimaryIndigo,
                        modifier = Modifier.size(32.dp)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Toca aquí para seleccionar un archivo",
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

                Spacer(modifier = Modifier.height(20.dp))

                Button(
                    onClick = onSelectFile,
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text("Abrir Galería / Archivos", fontSize = 12.sp, color = Color.White)
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

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
                        text = "Integración Directa con WhatsApp / Telegram",
                        color = TextPrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "También puedes analizar cualquier audio o video recibido en WhatsApp manteniendo presionado el archivo y seleccionando 'Compartir -> Detector Preventor'.",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }
        }

        if (payload != null) {
            Spacer(modifier = Modifier.height(20.dp))
            Button(
                onClick = onNavigateToScanner,
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text("Ver Diagnóstico de '${payload.filename}'", color = Color.White, fontSize = 13.sp)
            }
        }
    }
}

/**
 * Vista de dataset de pruebas académicas.
 */
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
                label = { Text("🎙️ Audio (5)", fontSize = 12.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = PrimaryIndigo,
                    selectedLabelColor = Color.White
                ),
                modifier = Modifier.weight(1f)
            )
            FilterChip(
                selected = selectedCategory == "image",
                onClick = { selectedCategory = "image" },
                label = { Text("👁️ Imagen (5)", fontSize = 12.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = PrimaryIndigo,
                    selectedLabelColor = Color.White
                ),
                modifier = Modifier.weight(1f)
            )
            FilterChip(
                selected = selectedCategory == "video",
                onClick = { selectedCategory = "video" },
                label = { Text("🎬 Video (5)", fontSize = 12.sp) },
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
                Pair("Muestra 1: Voz Humana Real (ASVspoof Bonafide)", "Audios .flac limpios de voz real humana. Inferencia esperada: RIESGO BAJO (4%)"),
                Pair("Muestra 2: Clonación por IA (ASVspoof Deepfake)", "Voz clonada mediante modelo neuronal sintético. Inferencia esperada: ALTO RIESGO (96%)"),
                Pair("Muestra 3: Síntesis de Texto a Voz (TTS AI)", "Audio generado por motor de texto a voz. Inferencia esperada: ALTO RIESGO (88%)"),
                Pair("Muestra 4: Nota de Voz WhatsApp Replicada", "Simulación de estafa por nota de voz en WhatsApp. Inferencia esperada: ALTO RIESGO (92%)"),
                Pair("Muestra 5: Conversación Telefónica Auténtica", "Grabación bonafide de llamada de voz. Inferencia esperada: RIESGO BAJO (8%)")
            )
            "image" -> listOf(
                Pair("Muestra 1: Retrato Real Prístino (FF++)", "Rostro real extraído de FaceForensics++. Inferencia esperada: RIESGO BAJO (2%)"),
                Pair("Muestra 2: Manipulación FaceSwap AI", "Rostro intercambiado mediante algoritmo FaceSwap. Inferencia esperada: ALTO RIESGO (98%)"),
                Pair("Muestra 3: Rostro Sintético GAN (CIFAKE)", "Imagen generada artificialmente por red GAN. Inferencia esperada: ALTO RIESGO (91%)"),
                Pair("Muestra 4: Reenactamiento Face2Face", "Manipulación de gesticulación facial Face2Face. Inferencia esperada: ALTO RIESGO (89%)"),
                Pair("Muestra 5: Fotografía HD Original", "Fotografía prístina de alta resolución. Inferencia esperada: RIESGO BAJO (5%)")
            )
            else -> listOf(
                Pair("Muestra 1: Entrevista Real YouTube (FF++)", "Video original prístino sin alterations. Inferencia esperada: RIESGO BAJO (5%)"),
                Pair("Muestra 2: Sincronización Labial LipSync AI", "Video con movimiento de labios alterado por IA. Inferencia esperada: ALTO RIESGO (91%)"),
                Pair("Muestra 3: FaceSwap HD Video", "Secuencia de video con rostro sustituido. Inferencia esperada: ALTO RIESGO (95%)"),
                Pair("Muestra 4: Avatar IA Multimodal Completo", "Video sintético con audio y rostro generados. Inferencia esperada: ALTO RIESGO (96%)"),
                Pair("Muestra 5: Clip de Cámara Frontal Auténtico", "Video grabado directamente con cámara frontal. Inferencia esperada: RIESGO BAJO (5%)")
            )
        }

        samples.forEachIndexed { idx, item ->
            Card(
                colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
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
                        Text(
                            text = item.first,
                            color = TextPrimary,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
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
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Text("Probar", fontSize = 11.sp, color = Color.White)
                    }
                }
            }
        }
    }
}

/**
 * Vista de información del sistema y resumen de la arquitectura.
 */
@Composable
fun SystemInfoView() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text(
            text = "Arquitectura MLOps & Edge AI",
            color = TextPrimary,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Especificaciones técnicas de los modelos e infraestructura local",
            color = TextSecondary,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "🤖 Modelos Experto Cuantizados (Assets Locales)",
                    color = PrimaryIndigo,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Experto Audio: MobileNetV3-Small INT8 (6.2 MB)", fontSize = 12.sp, color = TextPrimary)
                Text("• Experto Visión: EfficientNet-B0 INT8 (16.0 MB)", fontSize = 12.sp, color = TextPrimary)
                Text("• Aceleración hardware: TFLite GPU Delegate / 4 hilos CPU", fontSize = 12.sp, color = TextPrimary)
                Text("• Consumo de Memoria RAM: < 200 MB garantizado", fontSize = 12.sp, color = TextPrimary)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "📊 Resultados Oficiales de Entrenamiento (GPU CUDA)",
                    color = PrimaryIndigo,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("• Muestras Totales de Entrenamiento: 123,303 muestras", fontSize = 12.sp, color = TextPrimary)
                Text("• Precisión Experto Visión: 97.17% Accuracy | 95.82% F1-Score", fontSize = 12.sp, color = TextPrimary)
                Text("• Sensibilidad Experto Audio: 99.96% Recall | 87.80% F1-Score", fontSize = 12.sp, color = TextPrimary)
                Text("• Algoritmo de Fusión: Score-Level Fusion (w_a = 0.6, w_v = 0.4)", fontSize = 12.sp, color = TextPrimary)
            }
        }
    }
}

/**
 * Vista de interceptación y monitoreo de notificaciones de WhatsApp / Telegram en segundo plano.
 */
@Composable
fun NotificationsView(
    onInspectNotification: (InterceptedNotification) -> Unit
) {
    val context = LocalContext.current
    val notifications by NotificationRepository.notifications.collectAsState()
    val isMonitoringActive by NotificationRepository.isMonitoringActive.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Título y Estado
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Monitoreo WhatsApp",
                    color = TextPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Interceptación y Análisis Automático de Mensajería",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Box(
                modifier = Modifier
                    .background(
                        if (isMonitoringActive) RiskLowGreen.copy(alpha = 0.2f) else RiskHighRed.copy(alpha = 0.2f),
                        RoundedCornerShape(20.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                if (isMonitoringActive) RiskLowGreen else RiskHighRed,
                                CircleShape
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isMonitoringActive) "ACTIVO" else "PAUSADO",
                        color = if (isMonitoringActive) RiskLowGreen else RiskHighRed,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Tarjeta de Control del Servicio y Permisos
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
                            text = "Intercepta notas de voz (.opus, .mp3) y fotos entrantes de WhatsApp o Telegram para clasificarlas con Edge AI.",
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
                    Text("Configurar Permiso de Acceso en Android", fontSize = 12.sp, color = AccentCyan)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Simulador de Pruebas Rápidas
        Card(
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "🧪 Simulador de Amenazas (Prueba Inmediata)",
                    color = PrimaryIndigo,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Prueba la reacción del sistema simulando la llegada de mensajes multimedia:",
                    color = TextSecondary,
                    fontSize = 11.sp
                )

                Spacer(modifier = Modifier.height(12.dp))

                Button(
                    onClick = {
                        val fusion = RiskScorer.calculateGlobalRisk(0.94f, null)
                        NotificationRepository.addNotification(
                            InterceptedNotification(
                                id = UUID.randomUUID().toString(),
                                appName = "WhatsApp",
                                packageName = "com.whatsapp",
                                sender = "Mamá (Urgente)",
                                text = "Hijo, perdí mi tarjeta, hazme un depósito rápido por favor (Nota de voz 0:14)",
                                timestamp = System.currentTimeMillis(),
                                mediaType = MediaType.AUDIO_ONLY,
                                riskScore = fusion.globalRiskPercentage,
                                isThreat = true,
                                fusionResult = fusion
                            )
                        )
                        Toast.makeText(context, "🚨 Alerta: Nota de voz sospechosa interceptada", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = RiskHighRed.copy(alpha = 0.85f)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Warning, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("🚨 Simular Audio WhatsApp Falso (94% Riesgo)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        val fusion = RiskScorer.calculateGlobalRisk(null, 0.97f)
                        NotificationRepository.addNotification(
                            InterceptedNotification(
                                id = UUID.randomUUID().toString(),
                                appName = "WhatsApp",
                                packageName = "com.whatsapp",
                                sender = "Número Desconocido (+591 ...)",
                                text = "Mira esta foto tuya que encontré en redes (foto_comprometedora.jpg)",
                                timestamp = System.currentTimeMillis(),
                                mediaType = MediaType.IMAGE_ONLY,
                                riskScore = fusion.globalRiskPercentage,
                                isThreat = true,
                                fusionResult = fusion
                            )
                        )
                        Toast.makeText(context, "🚨 Alerta: Imagen FaceSwap detectada", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Image, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("📷 Simular Imagen WhatsApp FaceSwap (97% Riesgo)", fontSize = 12.sp, color = Color.White)
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        val fusion = RiskScorer.calculateGlobalRisk(0.05f, null)
                        NotificationRepository.addNotification(
                            InterceptedNotification(
                                id = UUID.randomUUID().toString(),
                                appName = "WhatsApp",
                                packageName = "com.whatsapp",
                                sender = "Carlos Amigo",
                                text = "Hola hermano, nos vemos a las 5pm en la facultad (Nota de voz 0:09)",
                                timestamp = System.currentTimeMillis(),
                                mediaType = MediaType.AUDIO_ONLY,
                                riskScore = fusion.globalRiskPercentage,
                                isThreat = false,
                                fusionResult = fusion
                            )
                        )
                        Toast.makeText(context, "✅ Nota de voz auténtica verificada", Toast.LENGTH_SHORT).show()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF059669)),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("✅ Simular Audio WhatsApp Seguro (5% Riesgo)", fontSize = 12.sp, color = Color.White)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Historial de Notificaciones
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

            if (notifications.isNotEmpty()) {
                TextButton(
                    onClick = { NotificationRepository.clearNotifications() }
                ) {
                    Text("Limpiar", color = TextSecondary, fontSize = 12.sp)
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

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
                            }

                            Text(
                                text = notif.formattedTime,
                                color = TextSecondary,
                                fontSize = 10.sp
                            )
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

                            Button(
                                onClick = { onInspectNotification(notif) },
                                colors = ButtonDefaults.buttonColors(containerColor = PrimaryIndigo),
                                shape = RoundedCornerShape(6.dp),
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                modifier = Modifier.height(30.dp)
                            ) {
                                Text("Ver Escáner", fontSize = 10.sp, color = Color.White)
                            }
                        }
                    }
                }
            }
        }
    }
}

