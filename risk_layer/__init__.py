"""
QuantSphereX Risk Layer Domain Package.
Preserves existing regime models while providing full institutional risk suite.
"""

from risk_layer.base import RiskMetricsReport
from risk_layer.config import RiskConfig
from risk_layer.engine import InstitutionalRiskEngine
from risk_layer.var_cvar import VaRCVaREngine
from risk_layer.stress_testing import StressTestingEngine
from risk_layer.liquidity_risk import LiquidityRiskEngine
from risk_layer.factor_risk import FactorRiskEngine
from risk_layer.sector_country_exposure import ExposureRiskEngine
from risk_layer.correlation_analysis import CorrelationAnalysisEngine
from risk_layer.tail_risk import TailRiskEngine
from risk_layer.scenario_analysis import ScenarioAnalysisEngine
from risk_layer.limits import LimitsAuditEngine

# Preserved legacy functions & classes
from risk_layer.regime_model import compute_regime_exposure
from risk_layer.vol_targeting import compute_vol_target_scalar
from risk_layer.regime_robustness import run_regime_robustness
from risk_layer.filters import RiskManager

__all__ = [
    "RiskMetricsReport",
    "RiskConfig",
    "InstitutionalRiskEngine",
    "VaRCVaREngine",
    "StressTestingEngine",
    "LiquidityRiskEngine",
    "FactorRiskEngine",
    "ExposureRiskEngine",
    "CorrelationAnalysisEngine",
    "TailRiskEngine",
    "ScenarioAnalysisEngine",
    "LimitsAuditEngine",
    "compute_regime_exposure",
    "compute_vol_target_scalar",
    "run_regime_robustness",
    "RiskManager",
]
