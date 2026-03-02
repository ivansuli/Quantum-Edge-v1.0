"""
╔══════════════════════════════════════════════════════════════╗
║  Macro Analyzer — Macroeconomic Context Engine               ║
║  Fed · CPI · NFP · DXY · Risk Country · Fiscal Policy       ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("MacroAnalyzer")


class MacroAnalyzer:
    """
    Analyzes macroeconomic context for trade signal filtering.
    
    Data Sources (when API keys available):
    - FRED API: Interest rates, CPI, PCE, unemployment
    - Alpha Vantage: DXY, commodities
    - Manual overrides via dashboard
    
    Without APIs, uses manually configurable defaults.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        
        # Default macro state (configurable via API)
        self._macro_state = {
            "fed_rate": 5.25,
            "fed_direction": "hold",        # "hiking", "hold", "cutting"
            "cpi_yoy": 3.2,
            "cpi_trend": "declining",       # "rising", "stable", "declining"
            "pce_yoy": 2.8,
            "unemployment": 3.9,
            "nfp_last": 175_000,
            "nfp_trend": "stable",
            "dxy_level": 104.5,
            "dxy_trend": 0.1,              # positive = strengthening
            "vix_level": 16.5,
            "risk_sentiment": "neutral",    # "risk_on", "neutral", "risk_off"
            "ar_risk_country": 1800,        # Argentina EMBI spread
            "fiscal_policy": "neutral",     # "expansionary", "neutral", "contractionary"
            "earnings_season": False,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def get_macro_context(self) -> Dict[str, Any]:
        """Return current macro context with derived sentiment."""
        context = dict(self._macro_state)
        context["overall_sentiment"] = self._compute_overall_sentiment()
        context["rate_environment"] = self._classify_rate_environment()
        context["inflation_risk"] = self._assess_inflation_risk()
        context["market_regime"] = self._classify_market_regime()
        return context

    def update_macro(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update macro parameters manually."""
        for key, value in updates.items():
            if key in self._macro_state:
                self._macro_state[key] = value
        self._macro_state["last_updated"] = datetime.utcnow().isoformat()
        return self.get_macro_context()

    def _compute_overall_sentiment(self) -> float:
        """
        Compute overall macro sentiment score (-1 to 1).
        Positive = favorable for risk assets, Negative = unfavorable.
        """
        score = 0.0

        # Fed direction impact
        fed_dir = self._macro_state.get("fed_direction", "hold")
        if fed_dir == "cutting":
            score += 0.25
        elif fed_dir == "hiking":
            score -= 0.30
        # hold is neutral

        # Inflation trend
        cpi_trend = self._macro_state.get("cpi_trend", "stable")
        if cpi_trend == "declining":
            score += 0.15
        elif cpi_trend == "rising":
            score -= 0.20

        # Employment
        nfp = self._macro_state.get("nfp_last", 150000)
        if nfp > 200000:
            score += 0.10
        elif nfp < 100000:
            score -= 0.15

        # DXY (strong dollar = headwind for stocks)
        dxy_trend = self._macro_state.get("dxy_trend", 0)
        score -= dxy_trend * 0.1

        # VIX
        vix = self._macro_state.get("vix_level", 20)
        if vix < 15:
            score += 0.10
        elif vix > 25:
            score -= 0.20
        elif vix > 30:
            score -= 0.35

        return round(max(-1, min(1, score)), 2)

    def _classify_rate_environment(self) -> str:
        fed_dir = self._macro_state.get("fed_direction", "hold")
        rate = self._macro_state.get("fed_rate", 5.0)
        if fed_dir == "cutting" or rate < 2.0:
            return "dovish"
        if fed_dir == "hiking" or rate > 5.5:
            return "hawkish"
        return "neutral"

    def _assess_inflation_risk(self) -> str:
        cpi = self._macro_state.get("cpi_yoy", 3.0)
        trend = self._macro_state.get("cpi_trend", "stable")
        if cpi > 4.0 and trend == "rising":
            return "high"
        if cpi > 3.0:
            return "moderate"
        return "low"

    def _classify_market_regime(self) -> str:
        vix = self._macro_state.get("vix_level", 20)
        sentiment = self._macro_state.get("risk_sentiment", "neutral")
        if vix > 30 or sentiment == "risk_off":
            return "crisis"
        if vix > 22:
            return "cautious"
        if vix < 15 and sentiment == "risk_on":
            return "euphoric"
        return "normal"
