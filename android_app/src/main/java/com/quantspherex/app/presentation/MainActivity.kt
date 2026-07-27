package com.quantspherex.app.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quantspherex.app.data.model.ChartPoint
import com.quantspherex.app.presentation.components.EquityCurveChart
import com.quantspherex.app.presentation.portfolio.PortfolioUiState
import com.quantspherex.app.presentation.portfolio.PortfolioViewModel
import com.quantspherex.app.presentation.research.ResearchUiState
import com.quantspherex.app.presentation.research.ResearchViewModel
import com.quantspherex.app.presentation.theme.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            QuantSphereXTheme {
                MainAppScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainAppScreen(
    portfolioViewModel: PortfolioViewModel = PortfolioViewModel(),
    researchViewModel: ResearchViewModel = ResearchViewModel()
) {
    var selectedTab by remember { mutableStateOf(0) }
    val portfolioState by portfolioViewModel.uiState.collectAsState()
    val researchState by researchViewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "QuantSphereX Mobile",
                        style = MaterialTheme.typography.titleLarge,
                        color = AccentCyan
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = DarkSurface)
            )
        },
        bottomBar = {
            NavigationBar(containerColor = DarkSurface) {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    label = { Text("Portfolio") },
                    icon = { Text("📊") }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    label = { Text("Research") },
                    icon = { Text("🔬") }
                )
            }
        },
        containerColor = DarkBackground
    ) { innerPadding ->
        Box(modifier = Modifier.padding(innerPadding).fillMaxSize()) {
            when (selectedTab) {
                0 -> PortfolioDashboardScreen(portfolioState)
                1 -> ResearchDashboardScreen(researchState)
            }
        }
    }
}

@Composable
fun PortfolioDashboardScreen(state: PortfolioUiState) {
    val dummyPoints = listOf(
        ChartPoint(1, "Jan", 10_000_000.0),
        ChartPoint(2, "Feb", 10_400_000.0),
        ChartPoint(3, "Mar", 10_850_000.0),
        ChartPoint(4, "Apr", 11_200_000.0),
        ChartPoint(5, "May", 11_850_000.0)
    )

    when (state) {
        is PortfolioUiState.Loading -> {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AccentCyan)
            }
        }
        is PortfolioUiState.Success -> {
            val summary = state.summary
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = DarkCardBackground),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text("Total AUM", color = TextSecondary, fontSize = 14.sp)
                            Text("₹${String.format("%.2f", summary.totalAum / 1e7)} Cr", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                            Spacer(Modifier.height(8.dp))
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("CAGR: ${String.format("%.1f%%", summary.cagrPct * 100)}", color = PositiveGreen, fontWeight = FontWeight.SemiBold)
                                Text("Sharpe: ${String.format("%.2f", summary.sharpeRatio)}", color = AccentCyan, fontWeight = FontWeight.SemiBold)
                                Text("Max DD: ${String.format("%.1f%%", summary.maxDrawdownPct * 100)}", color = NegativeRed, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                }

                item {
                    Text("Net Equity Curve", style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(8.dp))
                    EquityCurveChart(points = dummyPoints)
                }

                item {
                    Text("Active Positions (${state.positions.size})", style = MaterialTheme.typography.titleLarge)
                }

                items(state.positions) { pos ->
                    Card(
                        colors = CardDefaults.cardColors(containerColor = DarkSurface),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(pos.symbol, fontWeight = FontWeight.Bold, color = TextPrimary)
                                Text("${pos.shares} shares @ ₹${pos.currentPrice}", color = TextSecondary, fontSize = 12.sp)
                            }
                            Text(
                                "${String.format("%+.1f%%", pos.pnlPct * 100)}",
                                color = if (pos.pnlPct >= 0) PositiveGreen else NegativeRed,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
        }
        is PortfolioUiState.Error -> {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(state.message, color = NegativeRed)
            }
        }
    }
}

@Composable
fun ResearchDashboardScreen(state: ResearchUiState) {
    when (state) {
        is ResearchUiState.Loading -> {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AccentCyan)
            }
        }
        is ResearchUiState.Success -> {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    Text("AI Quant Alpha Signals", style = MaterialTheme.typography.titleLarge)
                }
                items(state.items) { item ->
                    Card(
                        colors = CardDefaults.cardColors(containerColor = DarkCardBackground),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text(item.symbol, fontWeight = FontWeight.Bold, fontSize = 18.sp, color = TextPrimary)
                                Text(item.direction, color = PositiveGreen, fontWeight = FontWeight.Bold)
                            }
                            Spacer(Modifier.height(4.dp))
                            Text(item.executiveSummary, color = TextSecondary, fontSize = 13.sp)
                            Spacer(Modifier.height(8.dp))
                            Text("Top Positive: ${item.topPositiveDriver}", color = PositiveGreen, fontSize = 12.sp)
                            Text("Top Negative: ${item.topNegativeDriver}", color = NegativeRed, fontSize = 12.sp)
                        }
                    }
                }
            }
        }
        is ResearchUiState.Error -> {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(state.message, color = NegativeRed)
            }
        }
    }
}
