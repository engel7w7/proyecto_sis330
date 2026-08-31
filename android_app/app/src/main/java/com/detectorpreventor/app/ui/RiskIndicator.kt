package com.detectorpreventor.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.detectorpreventor.app.domain.FusionResult
import com.detectorpreventor.app.domain.RiskLevel
import com.detectorpreventor.app.ui.theme.*

/**
 * Componente UI para presentar la tarjeta del veredicto final de riesgo.
 */
@Composable
fun RiskIndicatorCard(
    fusionResult: FusionResult,
    modifier: Modifier = Modifier
) {
    val (badgeColor, badgeText) = when (fusionResult.riskLevel) {
        RiskLevel.BAJO -> Pair(RiskLowGreen, "RIESGO BAJO - CONTENIDO AUTÉNTICO")
        RiskLevel.MEDIO -> Pair(RiskMediumYellow, "RIESGO MEDIO - ALERTA MODERADA")
        RiskLevel.ALTO -> Pair(RiskHighRed, "RIESGO ALTO - POSIBLE DEEPFAKE / ESTAFA")
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceDark)
            .border(1.dp, CardBorder, RoundedCornerShape(16.dp))
            .padding(18.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(badgeColor.copy(alpha = 0.15f))
                .border(1.dp, badgeColor.copy(alpha = 0.6f), RoundedCornerShape(10.dp))
                .padding(vertical = 10.dp, horizontal = 14.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = badgeText,
                color = badgeColor,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Probabilidad Global de Estafa",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    text = String.format("%.1f%%", fusionResult.globalRiskPercentage),
                    color = badgeColor,
                    fontSize = 34.sp,
                    fontWeight = FontWeight.Black
                )
            }

            Column(
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .background(PrimaryIndigo.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = String.format("Audio (STFT): %.0f%%", fusionResult.audioFraudProb * 100),
                        color = TextPrimary,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Box(
                    modifier = Modifier
                        .background(PrimaryIndigo.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = String.format("Visión (Face): %.0f%%", fusionResult.visionFraudProb * 100),
                        color = TextPrimary,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        LinearProgressIndicator(
            progress = { (fusionResult.globalRiskPercentage / 100f).coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp)
                .clip(RoundedCornerShape(5.dp)),
            color = badgeColor,
            trackColor = Color(0xFF1E293B)
        )

        Spacer(modifier = Modifier.height(14.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                .padding(10.dp)
        ) {
            Text(
                text = fusionResult.diagnosticSummary,
                color = TextSecondary,
                fontSize = 11.sp,
                lineHeight = 16.sp
            )
        }
    }
}

