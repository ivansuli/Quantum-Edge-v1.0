"""
╔══════════════════════════════════════════════════════════════╗
║  Signal Processor — AI Decision Rules Engine                 ║
║  Probability Gate · Correlation Filter · Volume Validation   ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("SignalProcessor")


class SignalProcessor:
    """
    AI Rules Engine that filters signals based on strict criteria.

    Rejection Rules:
    1. Win probability < 65% → REJECT
    2. Correlation with portfolio > 0.85 → REJECT
    3. Price at resistance without breakout volume → REJECT
    4. Extreme volatility without confirmation → REJECT
    5. Against macro sentiment alignment → PENALIZE
    """

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence

    def apply_ai_rules(
        self,
        win_probability: float,
        features: Dict[str, Any],
        signal: Dict[str, Any],
        portfolio: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply all AI rules and return pass/fail with detailed explanations.
        """
        violations = []
        warnings = []
        checks_passed = []

        # ── RULE 1: Minimum Win Probability ──
        if win_probability < self.min_confidence:
            violations.append({
                "rule": "MIN_WIN_PROBABILITY",
                "message": f"Win probability ({win_probability:.1%}) is below minimum threshold ({self.min_confidence:.0%})",
                "severity": "CRITICAL",
                "value": round(win_probability, 4),
                "threshold": self.min_confidence,
            })
        else:
            checks_passed.append({
                "rule": "MIN_WIN_PROBABILITY",
                "message": f"Win probability ({win_probability:.1%}) meets threshold",
            })

        # ── RULE 2: Portfolio Correlation ──
        correlation_result = self._check_portfolio_correlation(
            signal.get("ticker", ""), portfolio
        )
        if correlation_result["exceeds_limit"]:
            violations.append({
                "rule": "MAX_CORRELATION",
                "message": f"Ticker {signal.get('ticker')} has high correlation ({correlation_result['correlation']:.2f}) with existing portfolio",
                "severity": "HIGH",
                "value": correlation_result["correlation"],
                "threshold": 0.85,
            })
        else:
            checks_passed.append({
                "rule": "MAX_CORRELATION",
                "message": "Portfolio correlation within limits",
            })

        # ── RULE 3: Resistance Zone Without Volume ──
        in_resistance = features.get("in_resistance_zone", False)
        volume_breakout = features.get("volume_breakout", False)
        action = signal.get("action", "").upper()

        if action == "BUY" and in_resistance and not volume_breakout:
            violations.append({
                "rule": "RESISTANCE_NO_VOLUME",
                "message": "Price is in resistance zone without breakout volume. High risk of rejection.",
                "severity": "HIGH",
                "details": {
                    "resistance_distance": features.get("distance_to_resistance", 0),
                    "volume_ratio": features.get("volume_ratio", 0),
                },
            })
        elif action == "BUY" and in_resistance and volume_breakout:
            checks_passed.append({
                "rule": "RESISTANCE_VOLUME_CONFIRMED",
                "message": "Price at resistance WITH breakout volume — potential breakout",
            })
        else:
            checks_passed.append({
                "rule": "RESISTANCE_CHECK",
                "message": "Not in critical resistance zone",
            })

        # ── RULE 4: Extreme Volatility Guard ──
        volatility_regime = features.get("volatility_regime", "normal")
        if volatility_regime == "extreme":
            # Not a hard rejection but requires higher confidence
            if win_probability < 0.75:
                violations.append({
                    "rule": "EXTREME_VOLATILITY",
                    "message": f"Extreme volatility regime requires higher confidence (need 75%, have {win_probability:.1%})",
                    "severity": "MEDIUM",
                    "value": volatility_regime,
                })
            else:
                warnings.append({
                    "rule": "EXTREME_VOLATILITY",
                    "message": "Extreme volatility — confidence is high enough to proceed",
                })

        # ── RULE 5: RSI Extremes ──
        rsi = features.get("rsi")
        if rsi is not None:
            if action == "BUY" and rsi > 80:
                violations.append({
                    "rule": "RSI_EXTREME_OVERBOUGHT",
                    "message": f"RSI ({rsi:.1f}) is extremely overbought for a BUY signal",
                    "severity": "MEDIUM",
                })
            elif action == "SELL" and rsi < 20:
                violations.append({
                    "rule": "RSI_EXTREME_OVERSOLD",
                    "message": f"RSI ({rsi:.1f}) is extremely oversold for a SELL signal",
                    "severity": "MEDIUM",
                })
            else:
                checks_passed.append({
                    "rule": "RSI_CHECK",
                    "message": f"RSI ({rsi:.1f}) is acceptable for {action}",
                })

        # ── RULE 6: Macro Alignment ──
        macro_sentiment = features.get("macro_sentiment", 0)
        if action == "BUY" and macro_sentiment < -0.5:
            warnings.append({
                "rule": "MACRO_MISALIGNMENT",
                "message": "Buying against strong negative macro sentiment — elevated risk",
                "value": macro_sentiment,
            })
        elif action == "SELL" and macro_sentiment > 0.5:
            warnings.append({
                "rule": "MACRO_MISALIGNMENT",
                "message": "Selling against strong positive macro sentiment — elevated risk",
                "value": macro_sentiment,
            })

        # ── RULE 7: Near Psychological Level ──
        if features.get("near_psychological_level", False) and action == "BUY":
            warnings.append({
                "rule": "PSYCHOLOGICAL_LEVEL",
                "message": "Price near major psychological level — may act as resistance",
            })

        # ── RULE 8: Volume Confirmation ──
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio < 0.5:
            warnings.append({
                "rule": "LOW_VOLUME",
                "message": f"Volume ratio ({volume_ratio:.2f}) is very low — weak signal conviction",
            })

        # ── AGGREGATE DECISION ──
        critical_violations = [v for v in violations if v.get("severity") == "CRITICAL"]
        high_violations = [v for v in violations if v.get("severity") == "HIGH"]

        # Reject if any CRITICAL violation, or 2+ HIGH violations
        passed = len(critical_violations) == 0 and len(high_violations) < 2

        return {
            "passed": passed,
            "violations": violations,
            "warnings": warnings,
            "checks_passed": checks_passed,
            "summary": {
                "total_checks": len(violations) + len(warnings) + len(checks_passed),
                "violations_count": len(violations),
                "warnings_count": len(warnings),
                "passed_count": len(checks_passed),
                "critical_count": len(critical_violations),
            },
        }

    def _check_portfolio_correlation(
        self, ticker: str, portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Approximate correlation check based on sector."""
        positions = portfolio.get("positions", [])
        if not positions:
            return {"exceeds_limit": False, "correlation": 0.0}

        sector_map = {
            "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
            "META": "tech", "NVDA": "tech", "AMD": "tech", "INTC": "tech",
            "TSLA": "auto_tech", "JPM": "finance", "BAC": "finance",
            "GS": "finance", "XOM": "energy", "CVX": "energy",
            "SPY": "broad_market", "QQQ": "tech_market",
        }

        ticker_sector = sector_map.get(ticker, "unknown")
        portfolio_tickers = [p.get("ticker", "") for p in positions]
        same_sector = sum(
            1 for t in portfolio_tickers
            if sector_map.get(t, "other") == ticker_sector and ticker_sector != "unknown"
        )

        if same_sector >= 3:
            corr = 0.92
        elif same_sector >= 2:
            corr = 0.82
        elif same_sector >= 1:
            corr = 0.65
        else:
            corr = 0.25

        return {
            "exceeds_limit": corr > 0.85,
            "correlation": corr,
            "same_sector_count": same_sector,
            "sector": ticker_sector,
        }
