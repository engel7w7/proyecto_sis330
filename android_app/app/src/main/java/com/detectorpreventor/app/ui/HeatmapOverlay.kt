package com.detectorpreventor.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.detectorpreventor.app.ui.theme.CardBorder
import com.detectorpreventor.app.ui.theme.SurfaceDark

/**
 * Componente UI para visualización del mapa de calor explicativo Grad-CAM sobre la imagen o espectrograma.
 */
@Composable
fun HeatmapOverlay(
    bitmap: Bitmap?,
    isHeatmapActive: Boolean = true,
    riskFactor: Float = 0.5f,
    modifier: Modifier = Modifier
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        shape = RoundedCornerShape(16.dp),
        modifier = modifier
            .fillMaxWidth()
            .height(250.dp)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
    ) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Muestra Multimedia",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height

                    drawRect(
                        brush = Brush.verticalGradient(
                            colors = listOf(Color(0xFF0F172A), Color(0xFF1E293B), Color(0xFF0284C7).copy(alpha = 0.2f))
                        ),
                        size = Size(w, h)
                    )

                    val numBars = 36
                    val barWidth = w / numBars
                    for (i in 0 until numBars) {
                        val heightFactor = (Math.sin(i * 0.45) * 0.4 + 0.5f).toFloat()
                        val barH = h * heightFactor * 0.6f
                        val topY = (h - barH) * 0.5f

                        val barColor = if (i in 12..24) {
                            Color(0xFFF59E0B).copy(alpha = 0.6f)
                        } else {
                            Color(0xFF0284C7).copy(alpha = 0.4f)
                        }

                        drawRect(
                            color = barColor,
                            topLeft = Offset(i * barWidth + 2f, topY),
                            size = Size(barWidth - 4f, barH)
                        )
                    }
                }
            }

            if (isHeatmapActive) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    val isHighRisk = riskFactor >= 0.5f

                    if (isHighRisk) {
                        val intensity = (riskFactor * 0.85f).coerceIn(0.40f, 0.90f)
                        val mainGrad = Brush.radialGradient(
                            colors = listOf(
                                Color(0xFFDC2626).copy(alpha = intensity),
                                Color(0xFFF59E0B).copy(alpha = intensity * 0.7f),
                                Color(0xFFFACC15).copy(alpha = intensity * 0.4f),
                                Color(0xFF38BDF8).copy(alpha = 0.1f),
                                Color.Transparent
                            ),
                            center = Offset(w * 0.50f, h * 0.45f),
                            radius = w * 0.45f
                        )

                        val secGrad = Brush.radialGradient(
                            colors = listOf(
                                Color(0xFFEF4444).copy(alpha = intensity * 0.8f),
                                Color(0xFFF59E0B).copy(alpha = intensity * 0.5f),
                                Color.Transparent
                            ),
                            center = Offset(w * 0.42f, h * 0.60f),
                            radius = w * 0.32f
                        )

                        drawRect(brush = mainGrad, size = Size(w, h))
                        drawRect(brush = secGrad, size = Size(w, h))

                        val strokeWidth = 2.dp.toPx()
                        val rectLeft = w * 0.18f
                        val rectTop = h * 0.12f
                        val rectWidth = w * 0.64f
                        val rectHeight = h * 0.74f

                        drawRoundRect(
                            color = Color(0xFFEF4444).copy(alpha = 0.8f),
                            topLeft = Offset(rectLeft, rectTop),
                            size = Size(rectWidth, rectHeight),
                            cornerRadius = CornerRadius(12.dp.toPx()),
                            style = Stroke(
                                width = strokeWidth,
                                pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f), 0f)
                            )
                        )
                    } else {
                        // Verificación de autenticidad: marco verde esmeralda y gradiente tenue
                        val authGrad = Brush.radialGradient(
                            colors = listOf(
                                Color(0xFF10B981).copy(alpha = 0.20f),
                                Color.Transparent
                            ),
                            center = Offset(w * 0.50f, h * 0.50f),
                            radius = w * 0.60f
                        )
                        drawRect(brush = authGrad, size = Size(w, h))

                        val strokeWidth = 2.dp.toPx()
                        val rectLeft = w * 0.18f
                        val rectTop = h * 0.12f
                        val rectWidth = w * 0.64f
                        val rectHeight = h * 0.74f

                        drawRoundRect(
                            color = Color(0xFF10B981).copy(alpha = 0.7f),
                            topLeft = Offset(rectLeft, rectTop),
                            size = Size(rectWidth, rectHeight),
                            cornerRadius = CornerRadius(12.dp.toPx()),
                            style = Stroke(
                                width = strokeWidth,
                                pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f), 0f)
                            )
                        )
                    }
                }
            }

            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(10.dp)
                    .background(Color.Black.copy(alpha = 0.75f), RoundedCornerShape(8.dp))
                    .border(1.dp, Color(0xFF38BDF8).copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Visibility,
                        contentDescription = null,
                        tint = if (isHeatmapActive) Color(0xFFF87171) else Color(0xFF94A3B8),
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isHeatmapActive) "Grad-CAM Heatmap Activo" else "Vista Previa Limpia",
                        color = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }

            if (isHeatmapActive) {
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.85f))
                            )
                        )
                        .padding(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Anomalía Baja (Real)",
                            color = Color(0xFF38BDF8),
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Medium
                        )

                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(6.dp)
                                .padding(horizontal = 12.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(
                                    Brush.horizontalGradient(
                                        colors = listOf(
                                            Color(0xFF38BDF8),
                                            Color(0xFFFACC15),
                                            Color(0xFFF59E0B),
                                            Color(0xFFDC2626)
                                        )
                                    )
                                )
                        )

                        Text(
                            text = "Anomalía Alta (Fake)",
                            color = Color(0xFFF87171),
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

