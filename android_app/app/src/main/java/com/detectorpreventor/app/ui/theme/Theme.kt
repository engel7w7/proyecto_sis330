package com.detectorpreventor.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Paleta Modo Oscuro Premium (Dark Glassmorphism / Cybersecurity Aesthetic)
val BackgroundDark = Color(0xFF0B0F19)
val SurfaceDark = Color(0xFF161E2E)
val CardBorder = Color(0xFF2D3748)

val PrimaryIndigo = Color(0xFF6366F1)
val AccentCyan = Color(0xFF06B6D4)

val RiskLowGreen = Color(0xFF10B981)
val RiskMediumYellow = Color(0xFFF59E0B)
val RiskHighRed = Color(0xFFEF4444)

val TextPrimary = Color(0xFFF8FAFC)
val TextSecondary = Color(0xFF94A3B8)
val ButtonGray = Color(0xFF334155)

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryIndigo,
    secondary = AccentCyan,
    background = BackgroundDark,
    surface = SurfaceDark,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = TextPrimary,
    onSurface = TextPrimary
)

@Composable
fun DetectorPreventorTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}
