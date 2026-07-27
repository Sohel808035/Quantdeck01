"""
ai_quant_analyst/config.py
─────────────────────────
Configuration DTO for the QuantSphereX AI Quant Analyst.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AIAnalystConfig:
    """Master configuration for the AI Quant Analyst."""
    analyst_name: str = "QuantSphereX AI Quant Analyst"
    style: str = "institutional"  # 'institutional', 'executive', 'quantitative'
    max_top_features: int = 5      # Top N features to highlight in SHAP interpretations
    anomaly_z_threshold: float = 3.0 # Z-score threshold for anomaly detection
    confidence_level: float = 0.95  # Standard confidence level for risk interpretations
    enable_rule_engine: bool = True # Rule-based deterministic fallback (offline operation)
    report_format: str = "markdown" # 'markdown', 'json', 'text'
