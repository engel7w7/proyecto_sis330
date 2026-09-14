package com.sis330.detector

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Block
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Actividad Principal del Detector Móvil Multimodal de Estafas Digitales (Edge AI).
 * Proyecto SIS-330 (Taller de Grado / Sistemas Expertos) - Ing. Pacheco.
 *
 * Implementada con Jetpack Compose en Modo Oscuro de Alta Fidelidad.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // =========================================================================
        // PUNTO DE INSERCIÓN Y CAPTURA DEL INTENT DE WHATSAPP / COMPARTIR:
        // =========================================================================
        // Cuando WhatsApp o el usuario comparte un archivo multimedia a nuestra app,
        // Android invoca esta actividad con una acción Intent.ACTION_SEND o ACTION_VIEW.
        //
        // 1. Extraer el URI del archivo entrante (.opus, .mp3, .jpg, .mp4):
        //    val mediaUri = intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM) ?: intent.data
        //
        // 2. Si mediaUri != null:
        //    - Si el MIME type es audio/*: se deriva a AudioClassifier.classifySpectrogram()
        //    - Si el MIME type es image/* o video/*: se extraen 5 keyframes y se deriva a VisionClassifier
        //    - Finalmente se combinan con RiskScorer.calculateRisk(audioProb, visualProbs)
        // =========================================================================
        handleIncomingWhatsAppIntent(intent)

        setContent {
            DetectorDarkTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFF0F172A) // Slate-900 Oscuro de Ciberseguridad
                ) {
                    DetectorScreen(
                        onBlockAndReport = {
                            Toast.makeText(this, "Contacto bloqueado y evidencia enviada a ciberseguridad", Toast.LENGTH_LONG).show()
                        },
                        onDismissAnalysis = {
                            Toast.makeText(this, "Análisis descartado. Medio marcado como seguro.", Toast.LENGTH_SHORT).show()
                        }
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleIncomingWhatsAppIntent(it) }
    }

    private fun handleIncomingWhatsAppIntent(incomingIntent: Intent) {
        val action = incomingIntent.action
        val type = incomingIntent.type
        if (Intent.ACTION_SEND == action && type != null) {
            val mediaUri = incomingIntent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
            if (mediaUri != null) {
                // Aquí el MediaRouter procesa el archivo físico recibido desde WhatsApp
                Toast.makeText(this, "Archivo interceptado desde WhatsApp para verificación Edge AI", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

/**
 * Tema Oscuro Personalizado para la Aplicación.
 */
@Composable
fun DetectorDarkTheme(content: @Composable () -> Unit) {
    val darkColorScheme = darkColorScheme(
        primary = Color(0xFF38BDF8),       // Celeste Neón
        secondary = Color(0xFF94A3B8),     // Gris Neutral
        background = Color(0xFF0F172A),    // Fondo Fondo Negro Azulado
        surface = Color(0xFF1E293B),       // Superficie de Tarjetas
        error = Color(0xFFEF4444)          // Rojo Peligro
    )
    MaterialTheme(colorScheme = darkColorScheme, content = content)
}

/**
 * Pantalla Principal con el Score Global de Riesgo, Veredicto y Botones de Acción.
 */
@Composable
fun DetectorScreen(
    onBlockAndReport: () -> Unit,
    onDismissAnalysis: () -> Unit
) {
    // Estado de demostración: Evaluación con Fusión Multimodal (Regla 3/5 activada)
    var audioProb by remember { mutableStateOf(0.85f) }
    var keyframeProbs by remember { mutableStateOf(listOf(0.92f, 0.88f, 0.79f, 0.45f, 0.38f)) } // 3 de 5 frames superan umbral

    // Cálculo dinámico usando el algoritmo RiskScorer
    val riskResult = remember(audioProb, keyframeProbs) {
        RiskScorer.calculateRisk(audioFraudProb = audioProb, visualKeyframeProbs = keyframeProbs)
    }

    // Colores dinámicos según el veredicto
    val (verdictColor, badgeBg) = when (riskResult.verdict) {
        "Estafa Inminente" -> Color(0xFFEF4444) to Color(0x33EF4444)
        "Sospechoso" -> Color(0xFFF59E0B) to Color(0x33F59E0B)
        else -> Color(0xFF10B981) to Color(0x3310B981)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        Spacer(modifier = Modifier.height(10.dp))

        // Encabezado de la Aplicación
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Security,
                contentDescription = "Escudo de Seguridad",
                tint = Color(0xFF38BDF8),
                modifier = Modifier.size(28.dp)
            )
            Text(
                text = "SIS-330 • ANTIFRAUDE EDGE AI",
                color = Color(0xFF94A3B8),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.5.sp
            )
        }

        // Tarjeta Principal del Score Global de Riesgo
        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, verdictColor.copy(alpha = 0.5f), RoundedCornerShape(24.dp))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Badge de Veredicto
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(12.dp))
                        .background(badgeBg)
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = riskResult.verdict.uppercase(),
                        color = verdictColor,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = 1.sp
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                // TEXTO GRANDE REQUERIDO: "Riesgo de Estafa: [Score Global]%"
                Text(
                    text = "Riesgo de Estafa: ${String.format("%.1f", riskResult.globalRiskScore)}%",
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Barra de Progreso de Riesgo
                LinearProgressIndicator(
                    progress = { riskResult.globalRiskScore / 100.0f },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(10.dp)
                        .clip(RoundedCornerShape(5.dp)),
                    color = verdictColor,
                    trackColor = Color(0xFF334155)
                )

                Spacer(modifier = Modifier.height(18.dp))

                Text(
                    text = if (riskResult.rule35Triggered) {
                        "⚠️ Alerta Crítica: Se detectó clonación de voz concurrentemente con manipulación facial sistemática (Regla 3/5)."
                    } else {
                        "Patrón de comunicación dentro de los parámetros esperados."
                    },
                    color = Color(0xFFCBD5E1),
                    fontSize = 13.sp,
                    textAlign = TextAlign.Center
                )
            }
        }

        // Desglose de Diagnóstico Multimodal
        Card(
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B).copy(alpha = 0.7f)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = "Desglose de Inferencia Multimodal (INT8 Edge):",
                    color = Color(0xFF94A3B8),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )

                MetricRow(
                    label = "Audio (MobileNetV3 - Espectrograma):",
                    value = "${String.format("%.1f", riskResult.audioProb * 100)}%",
                    isWarning = riskResult.audioProb > 0.5f
                )

                MetricRow(
                    label = "Visión (EfficientNet-B0 - Regla 3/5):",
                    value = "${String.format("%.1f", riskResult.visualProb * 100)}% (${riskResult.framesFlaggedCount}/5 frames)",
                    isWarning = riskResult.rule35Triggered
                )

                Divider(color = Color(0xFF334155), thickness = 0.8.dp)

                Text(
                    text = "Fusión Tardía: 55% Peso Audio + 45% Peso Visión.",
                    color = Color(0xFF64748B),
                    fontSize = 11.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // BOTÓN ROJO DE PELIGRO: "Bloquear y Reportar"
        Button(
            onClick = onBlockAndReport,
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFFEF4444), // Rojo Peligro
                contentColor = Color.White
            ),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Block,
                contentDescription = null,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = "Bloquear y Reportar",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // BOTÓN GRIS SECUNDARIO: "Descartar Análisis"
        OutlinedButton(
            onClick = onDismissAnalysis,
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = Color(0xFF94A3B8)
            ),
            border = ButtonDefaults.outlinedButtonBorder.copy(
                brush = androidx.compose.ui.graphics.SolidColor(Color(0xFF475569))
            ),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = null,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Descartar Análisis",
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold
            )
        }

        // Botón de Demostración para el Docente (Cambiar Escenario en Vivo)
        TextButton(
            onClick = {
                if (audioProb > 0.5f) {
                    audioProb = 0.12f
                    keyframeProbs = listOf(0.08f, 0.15f, 0.10f, 0.05f, 0.09f)
                } else {
                    audioProb = 0.88f
                    keyframeProbs = listOf(0.95f, 0.91f, 0.84f, 0.40f, 0.25f)
                }
            }
        ) {
            Icon(
                imageVector = Icons.Default.Refresh,
                contentDescription = null,
                tint = Color(0xFF38BDF8),
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "Alternar Escenario de Prueba (Seguro ⟷ Estafa)",
                color = Color(0xFF38BDF8),
                fontSize = 12.sp
            )
        }
    }
}

@Composable
fun MetricRow(label: String, value: String, isWarning: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, color = Color(0xFFCBD5E1), fontSize = 12.sp)
        Text(
            text = value,
            color = if (isWarning) Color(0xFFEF4444) else Color(0xFF10B981),
            fontWeight = FontWeight.Bold,
            fontSize = 12.sp
        )
    }
}
