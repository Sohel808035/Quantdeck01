package com.quantspherex.app.presentation.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.quantspherex.app.data.model.ChartPoint
import com.quantspherex.app.presentation.theme.AccentBlue
import com.quantspherex.app.presentation.theme.AccentCyan

@Composable
fun EquityCurveChart(
    points: List<ChartPoint>,
    modifier: Modifier = Modifier.fillMaxWidth().height(220.dp)
) {
    if (points.isEmpty()) return

    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height

        val minY = points.minOf { it.value }
        val maxY = points.maxOf { it.value }
        val rangeY = if (maxY - minY == 0.0) 1.0 else maxY - minY

        val path = Path()
        val fillPath = Path()

        points.forEachIndexed { index, point ->
            val x = (index.toFloat() / (points.size - 1)) * width
            val y = height - (((point.value - minY) / rangeY).toFloat() * (height * 0.8f) + (height * 0.1f))

            if (index == 0) {
                path.moveTo(x, y)
                fillPath.moveTo(x, height)
                fillPath.lineTo(x, y)
            } else {
                path.lineTo(x, y)
                fillPath.lineTo(x, y)
            }
        }

        fillPath.lineTo(width, height)
        fillPath.close()

        // Draw gradient area under equity curve
        drawPath(
            path = fillPath,
            brush = Brush.verticalGradient(
                colors = listOf(AccentCyan.copy(alpha = 0.35f), Color.Transparent)
            )
        )

        // Draw equity curve line
        drawPath(
            path = path,
            color = AccentCyan,
            style = Stroke(width = 3.dp.toPx())
        )
    }
}
