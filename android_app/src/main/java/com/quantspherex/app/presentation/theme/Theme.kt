package com.quantspherex.app.presentation.theme

import androidx.compose.material3.DarkColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val DarkBackground = Color(0xFF0E1117)
val DarkSurface = Color(0xFF161B22)
val DarkCardBackground = Color(0xFF21262D)
val AccentCyan = Color(0xFF00E5FF)
val AccentBlue = Color(0xFF2979FF)
val PositiveGreen = Color(0xFF00E676)
val NegativeRed = Color(0xFFFF5252)
val TextPrimary = Color(0xFFE6EDF3)
val TextSecondary = Color(0xFF8B949E)

private val QuantColorScheme = DarkColorScheme(
    primary = AccentCyan,
    secondary = AccentBlue,
    background = DarkBackground,
    surface = DarkSurface,
    onPrimary = Color.Black,
    onBackground = TextPrimary,
    onSurface = TextPrimary
)

val QuantTypography = Typography(
    headlineLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Bold,
        fontSize = 28.sp,
        color = TextPrimary
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.SemiBold,
        fontSize = 20.sp,
        color = TextPrimary
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        color = TextSecondary
    )
)

@Composable
fun QuantSphereXTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = QuantColorScheme,
        typography = QuantTypography,
        content = content
    )
}
