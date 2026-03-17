import pandas as pd  # type: ignore
import logging

logger = logging.getLogger(__name__)

class AlphaBaseModel:
    """Base framework for Machine Learning models predicting 60D/120D relative alpha."""
    
    def __init__(self, target_horizon: int = 60):
        self.target_horizon = target_horizon
        
    def create_target(self, stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the target variable: Future N-Day Excess Return vs NIFTY.
        alpha_target = (Stock_Close_{t+N} / Stock_Close_t) - (Nifty_Close_{t+N} / Nifty_Close_t)
        """
        logger.info(f"Creating {self.target_horizon}D target alpha...")
        # TODO: Implement shift operations to calculate N-day forward returns and subtract index return
        return stock_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit model to training data."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict cross-sectional alpha."""
        raise NotImplementedError
