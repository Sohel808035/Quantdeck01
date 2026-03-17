import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Applies institutional risk constraints strictly over portfolio allocations."""
    
    def __init__(self, max_drawdown_limit: float = 0.20):
        self.max_drawdown_limit = max_drawdown_limit
        
    def check_drawdown_stop(self, equity_curve: pd.Series) -> bool:
        """
        Calculates the current drawdown.
        Returns True if trading should be halted (drawdown > limit).
        """
        if equity_curve.empty:
            return False
            
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        current_dd = drawdown.iloc[-1]
        
        if current_dd < -self.max_drawdown_limit:
            logger.warning(f"CRITICAL: Max Drawdown breached ({current_dd:.2%} limit: {self.max_drawdown_limit:.2%}). Halting trading.")
            return True
            
        return False
        
    def adjust_for_regime(self, target_weights: pd.Series, regime_features: pd.DataFrame) -> pd.Series:
        """
        Adjusts portfolio leverage based on the current market regime.
        If in bear market or high volatility regime, reduce total exposure.
        """
        if target_weights.empty or regime_features.empty:
            return target_weights
            
        # Get latest regime indicators
        latest_regime = regime_features.iloc[-1]
        
        exposure_multiplier = 1.0
        
        if latest_regime.get('market_bull_regime', 1) == 0:
            logger.info("Bear Regime detected: Reducing exposure by 50%")
            exposure_multiplier *= 0.5
            
        if latest_regime.get('high_vol_regime', 0) == 1:
            logger.info("High Volatility Regime detected: Reducing exposure by 50%")
            exposure_multiplier *= 0.5
            
        adjusted_weights = target_weights * exposure_multiplier
        adjusted_weights.name = 'adjusted_weight'
        return adjusted_weights
