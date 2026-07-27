"""
execution_layer/simulator/config.py
───────────────────────────────────
Configuration settings for the QuantSphereX Execution Simulator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionSimulatorConfig:
    """Master configuration for execution simulation."""
    # Latency
    latency_ms: float = 50.0                # Simulated network/order entry latency in ms

    # Costs
    commission_bps: float = 10.0            # Broker fee in basis points
    fixed_commission_per_order: float = 10.0 # Fixed broker fee per order (INR)
    bid_ask_spread_bps: float = 5.0          # Default bid-ask spread in bps
    slippage_bps: float = 2.0               # Base slippage in bps

    # Liquidity & Partial Fills
    max_volume_participation: float = 0.10  # Max 10% of candle volume per slice
    enable_partial_fills: bool = True

    # Algorithmic Execution Defaults
    vwap_slices: int = 10                   # Default slices for VWAP execution
    twap_slices: int = 10                   # Default slices for TWAP execution
    twap_interval_seconds: int = 60         # 1-minute TWAP slices
