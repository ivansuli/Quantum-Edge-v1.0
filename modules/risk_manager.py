"""
╔══════════════════════════════════════════════════════════════╗
║  Risk Manager — Position Sizing & Risk Control Engine        ║
║  Kelly Criterion · ATR-Based Stops · Correlation Guard       ║
╚══════════════════════════════════════════════════════════════╝
"""

import math
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("RiskManager")


@dataclass
class Position:
    ticker: str
    action: str
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    dollar_risk: float
    timestamp: str = ""


class RiskManager:
    """
    Professional-grade risk management engine.

    Rules:
    - Max risk per trade: 1-2% of total capital
    - Min risk/reward ratio: 1:2
    - Max portfolio risk: 6%
    - Max correlation between positions: 0.85
    - ATR-based stop loss (2x ATR default)
    - Kelly Criterion position sizing with half-Kelly safety
    """

    def __init__(
        self,
        total_capital: float = 100_000.0,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.06,
        max_correlation: float = 0.85,
        min_risk_reward: float = 2.0,
        atr_multiplier: float = 2.0,
    ):
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.min_risk_reward = min_risk_reward
        self.atr_multiplier = atr_multiplier
        self.active_positions: List[Position] = []

    def calculate_position(
        self,
        ticker: str,
        entry_price: float,
        action: str,
        atr: Optional[float] = None,
        support: Optional[float] = None,
        resistance: Optional[float] = None,
        win_probability: float = 0.65,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size, stop loss, and take profit.

        Uses a hierarchy for stop loss:
        1. ATR-based (preferred - 2x ATR)
        2. Support/Resistance based
        3. Percentage fallback (2%)
        """
        action = action.upper()

        # ─── STOP LOSS CALCULATION ─────────────────────────
        stop_loss = self._calculate_stop_loss(entry_price, action, atr, support, resistance)
        stop_distance = abs(entry_price - stop_loss)
        stop_distance = max(stop_distance, entry_price * 0.005)  # Min 0.5% stop

        # ─── TAKE PROFIT CALCULATION ───────────────────────
        take_profit = self._calculate_take_profit(
            entry_price, action, stop_distance, resistance, support
        )
        profit_distance = abs(take_profit - entry_price)

        # ─── RISK/REWARD RATIO ─────────────────────────────
        risk_reward = profit_distance / stop_distance if stop_distance > 0 else 0

        # Enforce minimum R:R
        if risk_reward < self.min_risk_reward:
            if action == "BUY":
                take_profit = entry_price + (stop_distance * self.min_risk_reward)
            else:
                take_profit = entry_price - (stop_distance * self.min_risk_reward)
            profit_distance = abs(take_profit - entry_price)
            risk_reward = self.min_risk_reward

        # ─── POSITION SIZING ──────────────────────────────
        max_dollar_risk = self.total_capital * self.max_risk_per_trade
        
        # Kelly Criterion (half-Kelly for safety)
        kelly_fraction = self._kelly_criterion(win_probability, risk_reward)
        kelly_dollar_risk = self.total_capital * kelly_fraction
        
        # Use the more conservative of the two
        dollar_risk = min(max_dollar_risk, kelly_dollar_risk)
        
        # Check portfolio-level risk
        current_portfolio_risk = sum(p.dollar_risk for p in self.active_positions)
        remaining_risk_budget = (self.total_capital * self.max_portfolio_risk) - current_portfolio_risk
        dollar_risk = min(dollar_risk, max(0, remaining_risk_budget))

        # Shares calculation
        if stop_distance > 0:
            position_size = int(dollar_risk / stop_distance)
        else:
            position_size = 0

        # Sanity checks
        position_value = position_size * entry_price
        max_position_value = self.total_capital * 0.25  # Max 25% in single position
        if position_value > max_position_value:
            position_size = int(max_position_value / entry_price)
            dollar_risk = position_size * stop_distance

        return {
            "ticker": ticker,
            "action": action,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "stop_distance": round(stop_distance, 2),
            "position_size": position_size,
            "position_value": round(position_size * entry_price, 2),
            "dollar_risk": round(dollar_risk, 2),
            "risk_percent": round((dollar_risk / self.total_capital) * 100, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "kelly_fraction": round(kelly_fraction, 4),
            "stop_method": self._get_stop_method(atr, support, resistance, action),
            "portfolio_risk_used": round(
                ((current_portfolio_risk + dollar_risk) / self.total_capital) * 100, 2
            ),
        }

    def _calculate_stop_loss(
        self,
        entry_price: float,
        action: str,
        atr: Optional[float],
        support: Optional[float],
        resistance: Optional[float],
    ) -> float:
        """
        Stop loss hierarchy:
        1. ATR-based (2x ATR below/above entry)
        2. Below support (BUY) / Above resistance (SELL)
        3. Percentage fallback (2%)
        """
        if action == "BUY":
            if atr and atr > 0:
                atr_stop = entry_price - (atr * self.atr_multiplier)
                if support and support > 0:
                    # Use the tighter of ATR stop and just below support
                    support_stop = support - (entry_price * 0.002)
                    return max(atr_stop, support_stop)
                return atr_stop
            elif support and support > 0:
                return support - (entry_price * 0.002)
            else:
                return entry_price * 0.98  # 2% fallback
        else:  # SELL
            if atr and atr > 0:
                atr_stop = entry_price + (atr * self.atr_multiplier)
                if resistance and resistance > 0:
                    resistance_stop = resistance + (entry_price * 0.002)
                    return min(atr_stop, resistance_stop)
                return atr_stop
            elif resistance and resistance > 0:
                return resistance + (entry_price * 0.002)
            else:
                return entry_price * 1.02

    def _calculate_take_profit(
        self,
        entry_price: float,
        action: str,
        stop_distance: float,
        resistance: Optional[float],
        support: Optional[float],
    ) -> float:
        """Take profit at minimum 2x risk distance or next S/R level"""
        min_tp_distance = stop_distance * self.min_risk_reward

        if action == "BUY":
            min_tp = entry_price + min_tp_distance
            if resistance and resistance > entry_price:
                return max(min_tp, resistance)
            return min_tp
        else:
            min_tp = entry_price - min_tp_distance
            if support and support < entry_price:
                return min(min_tp, support)
            return min_tp

    def _kelly_criterion(self, win_prob: float, risk_reward: float) -> float:
        """
        Half-Kelly Criterion for position sizing.
        f* = (p * b - q) / b, then halved for safety.
        """
        p = min(max(win_prob, 0.01), 0.99)
        q = 1 - p
        b = risk_reward

        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(kelly, 0)

        # Half-Kelly (conservative)
        half_kelly = kelly / 2

        # Cap at max risk per trade
        return min(half_kelly, self.max_risk_per_trade)

    def _get_stop_method(
        self, atr, support, resistance, action
    ) -> str:
        if atr and atr > 0:
            return "ATR-based"
        if action == "BUY" and support:
            return "Support-based"
        if action == "SELL" and resistance:
            return "Resistance-based"
        return "Percentage fallback"

    def check_correlation(
        self, ticker: str, portfolio_tickers: List[str]
    ) -> Dict[str, Any]:
        """
        Check if new ticker is too correlated with existing portfolio.
        Returns correlation status and max correlation found.
        """
        # Sector-based correlation approximation
        sector_map = {
            "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
            "META": "tech", "NVDA": "tech", "AMD": "tech", "TSLA": "auto",
            "JPM": "finance", "BAC": "finance", "GS": "finance",
            "XOM": "energy", "CVX": "energy", "COP": "energy",
            "JNJ": "health", "PFE": "health", "UNH": "health",
            "SPY": "index", "QQQ": "tech_index", "IWM": "index",
        }

        ticker_sector = sector_map.get(ticker, "unknown")
        same_sector_count = sum(
            1 for t in portfolio_tickers
            if sector_map.get(t, "other") == ticker_sector and ticker_sector != "unknown"
        )

        # Approximate correlation based on sector overlap
        if same_sector_count >= 3:
            approx_correlation = 0.90
        elif same_sector_count >= 2:
            approx_correlation = 0.80
        elif same_sector_count >= 1:
            approx_correlation = 0.65
        else:
            approx_correlation = 0.30

        return {
            "ticker": ticker,
            "sector": ticker_sector,
            "same_sector_positions": same_sector_count,
            "approximate_correlation": approx_correlation,
            "exceeds_limit": approx_correlation > self.max_correlation,
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        total_risk = sum(p.dollar_risk for p in self.active_positions)
        total_value = sum(p.shares * p.entry_price for p in self.active_positions)

        return {
            "total_capital": self.total_capital,
            "total_invested": round(total_value, 2),
            "total_risk": round(total_risk, 2),
            "risk_percentage": round((total_risk / self.total_capital) * 100, 2),
            "available_capital": round(self.total_capital - total_value, 2),
            "available_risk_budget": round(
                (self.total_capital * self.max_portfolio_risk) - total_risk, 2
            ),
            "positions_count": len(self.active_positions),
            "positions": [
                {
                    "ticker": p.ticker,
                    "action": p.action,
                    "shares": p.shares,
                    "entry": p.entry_price,
                    "stop": p.stop_loss,
                    "target": p.take_profit,
                    "risk": p.dollar_risk,
                }
                for p in self.active_positions
            ],
        }
