"""
portfolio_layer/ranking.py  (v2 — Top 25 + Turnover Buffer)
─────────────────────────────────────────────────────────────
Cross-sectional ranking with a turnover buffer:

  SELECT TOP 25 by predicted score.
  RETAIN   current holding if its rank falls within TOP 30.
  This prevents unnecessary churn when a stock slips from rank 25 → 28.
"""

from __future__ import annotations

import logging
from typing import Set

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


class CrossSectionalRanker:
    """
    V3 Selection Logic (Turnover Reduction Optimized):
    - Hysteresis: Buy Top N (45), Sell if > Buffer (60).
    - Minimum Holding: 20 days (unless rank > Emergency 80).
    - Signal Smoothing: 5-day rolling average.
    """

    def __init__(self, top_n: int = 45, buffer_n: int = 60, min_hold: int = 20, emergency_rank: int = 80):
        self.top_n = top_n
        self.buffer_n = buffer_n
        self.min_hold = min_hold
        self.emergency_rank = emergency_rank
        # Track holding duration: {ticker: days_held}
        self.holding_since: dict[str, int] = {}

    def select_portfolio(
        self,
        scores: pd.Series,
        current_holdings: set[str],
        rebalance_date: pd.Timestamp
    ) -> set[str]:
        """
        Institutional selection with Hysteresis and Minimum Holding constraints.
        """
        if scores.empty:
            return set()

        # 1. Rank 1 = Highest Score
        ranks = scores.rank(ascending=False, method="first").astype(int)
        
        # 2. Update holding durations (increment for current survivors)
        # Note: We assume this is called only on monthly rebalance dates, 
        # so we increment by approx 21 days or just use date diff.
        # But for simplification, we'll use a simple count and let logic be 'days since entry'.
        
        # 3. Identify Candidates to SELL
        to_sell = set()
        for stock in current_holdings:
            if stock not in ranks.index:
                to_sell.add(stock)
                continue
                
            rank = ranks[stock]
            days_held = self.holding_since.get(stock, 0)
            
            # Rule: Sell if rank > buffer (60) AND (held > 20 days OR rank > 80)
            if rank > self.buffer_n:
                if days_held >= self.min_hold or rank > self.emergency_rank:
                    to_sell.add(stock)
        
        # 4. Identify Candidates to BUY
        # Filter available universe (not currently held)
        candidates = ranks[ranks <= self.top_n].index.tolist()
        potential_buys = [c for c in candidates if c not in (current_holdings - to_sell)]
        
        # 5. Core Portfolio: (Holdings - Sells)
        survivors = current_holdings - to_sell
        
        # 6. Fill remaining slots up to top_n
        new_holdings = list(survivors)
        slots_available = self.top_n - len(new_holdings)
        
        if slots_available > 0:
            # Sort potential buys by rank and pick best
            top_potentials = sorted(potential_buys, key=lambda x: int(ranks[x]))
            top_potentials = top_potentials[:int(slots_available)]
            new_holdings.extend(top_potentials)
            
            # Initialize entry for new buys
            for b in top_potentials:
                self.holding_since[b] = 0
        
        # 7. Update holding durations for the next round
        # In a real system we'd use exact date diffs, here we increment by 21 days (std month)
        for h in new_holdings:
            self.holding_since[h] = self.holding_since.get(h, 0) + 21
            
        # Clean up history for exited stocks
        final_set = set(new_holdings)
        stocks_exited = set(self.holding_since.keys()) - final_set
        for ex in stocks_exited:
            self.holding_since.pop(ex, None)

        logger.info(
            f"  [Portfolio] Hold: {len(survivors)} | Buy: {len(final_set - current_holdings)} | "
            f"Sell: {len(current_holdings - final_set)}"
        )
        return final_set
