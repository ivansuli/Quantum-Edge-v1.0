"""
╔══════════════════════════════════════════════════════════════╗
║  Feature Engine — Technical Indicator Feature Engineering     ║
║  Trend · Momentum · Volume · Volatility Regime Analysis      ║
╚══════════════════════════════════════════════════════════════╝
"""

import math
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("FeatureEngine")


class FeatureEngine:
    """
    Transforms raw market data and indicators into ML-ready features.

    Feature Categories:
    1. Trend: MA alignment, higher highs/lows, price vs MAs
    2. Momentum: RSI zones, MACD divergence, rate of change
    3. Volume: VWAP deviation, volume delta, price-volume correlation
    4. Volatility: ATR regime, Bollinger bandwidth proxy
    5. Support/Resistance: Distance to key levels, psychological levels
    """

    def compute_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Master feature computation from enriched signal data."""
        features = {}

        # ─── RAW INDICATORS (pass through) ─────────────
        features["price"] = data.get("price", 0)
        features["rsi"] = data.get("rsi")
        features["macd"] = data.get("macd")
        features["macd_signal"] = data.get("macd_signal")
        features["macd_histogram"] = data.get("macd_histogram")
        features["ema_20"] = data.get("ema_20")
        features["ema_50"] = data.get("ema_50")
        features["ema_200"] = data.get("ema_200")
        features["atr"] = data.get("atr")
        features["vwap"] = data.get("vwap")
        features["volume"] = data.get("volume")
        features["avg_volume"] = data.get("avg_volume")
        features["support"] = data.get("support")
        features["resistance"] = data.get("resistance")
        features["action"] = data.get("action", "BUY")

        # ─── TREND FEATURES ───────────────────────────
        features["trend_score"] = self._compute_trend_score(data)
        features["ma_alignment"] = self._compute_ma_alignment(data)
        features["price_vs_ema200"] = self._price_vs_ma(data, "ema_200")
        features["price_vs_ema50"] = self._price_vs_ma(data, "ema_50")
        features["price_vs_ema20"] = self._price_vs_ma(data, "ema_20")
        features["ma_spread"] = self._compute_ma_spread(data)

        # ─── MOMENTUM FEATURES ────────────────────────
        features["momentum_score"] = self._compute_momentum_score(data)
        features["rsi_zone"] = self._rsi_zone(data.get("rsi"))
        features["rsi_divergence"] = self._detect_rsi_divergence(data)
        features["macd_crossover"] = self._detect_macd_crossover(data)
        features["macd_momentum"] = self._macd_momentum(data)

        # ─── VOLUME FEATURES ─────────────────────────
        features["volume_score"] = self._compute_volume_score(data)
        features["volume_ratio"] = self._volume_ratio(data)
        features["vwap_deviation"] = self._vwap_deviation(data)
        features["volume_breakout"] = self._is_volume_breakout(data)

        # ─── VOLATILITY FEATURES ──────────────────────
        features["volatility_regime"] = self._classify_volatility(data)
        features["atr_percent"] = self._atr_percent(data)
        features["volatility_score"] = self._compute_volatility_score(data)

        # ─── SUPPORT/RESISTANCE FEATURES ──────────────
        features["sr_analysis"] = self._analyze_sr_levels(data)
        features["distance_to_support"] = self._distance_to_level(data, "support")
        features["distance_to_resistance"] = self._distance_to_level(data, "resistance")
        features["near_psychological_level"] = self._near_psychological_level(data.get("price", 0))
        features["in_resistance_zone"] = self._in_resistance_zone(data)

        # ─── COMPOSITE SCORE ──────────────────────────
        features["composite_technical_score"] = round(
            (features["trend_score"] * 0.30) +
            (features["momentum_score"] * 0.25) +
            (features["volume_score"] * 0.25) +
            (features["volatility_score"] * 0.20), 4
        )

        return features

    # ═══════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _compute_trend_score(self, data: Dict) -> float:
        """Score 0-1: Overall trend strength and alignment."""
        score = 0.5  # Neutral start
        price = data.get("price", 0)
        if not price:
            return score

        # Price above/below key MAs
        ema_20 = data.get("ema_20")
        ema_50 = data.get("ema_50")
        ema_200 = data.get("ema_200")

        if ema_200 and price > ema_200:
            score += 0.15
        elif ema_200 and price < ema_200:
            score -= 0.15

        if ema_50 and price > ema_50:
            score += 0.10
        elif ema_50 and price < ema_50:
            score -= 0.10

        if ema_20 and price > ema_20:
            score += 0.10
        elif ema_20 and price < ema_20:
            score -= 0.10

        # MA alignment (golden cross / death cross)
        if ema_20 and ema_50 and ema_200:
            if ema_20 > ema_50 > ema_200:
                score += 0.15  # Perfect bullish alignment
            elif ema_20 < ema_50 < ema_200:
                score -= 0.15  # Perfect bearish alignment

        return round(max(0, min(1, score)), 4)

    def _compute_ma_alignment(self, data: Dict) -> str:
        ema_20 = data.get("ema_20")
        ema_50 = data.get("ema_50")
        ema_200 = data.get("ema_200")

        if not all([ema_20, ema_50, ema_200]):
            return "unknown"
        if ema_20 > ema_50 > ema_200:
            return "bullish_aligned"
        if ema_20 < ema_50 < ema_200:
            return "bearish_aligned"
        if ema_20 > ema_50 and ema_50 < ema_200:
            return "bullish_crossover"
        if ema_20 < ema_50 and ema_50 > ema_200:
            return "bearish_crossover"
        return "mixed"

    def _price_vs_ma(self, data: Dict, ma_key: str) -> float:
        price = data.get("price", 0)
        ma = data.get(ma_key)
        if not price or not ma or ma == 0:
            return 0
        return round((price - ma) / ma, 4)

    def _compute_ma_spread(self, data: Dict) -> float:
        ema_20 = data.get("ema_20")
        ema_200 = data.get("ema_200")
        if not ema_20 or not ema_200 or ema_200 == 0:
            return 0
        return round((ema_20 - ema_200) / ema_200, 4)

    # ═══════════════════════════════════════════════════════
    # MOMENTUM ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _compute_momentum_score(self, data: Dict) -> float:
        score = 0.5
        rsi = data.get("rsi")
        macd_hist = data.get("macd_histogram")
        action = data.get("action", "BUY")

        if rsi is not None:
            if action == "BUY":
                if 40 <= rsi <= 60:
                    score += 0.15  # Neutral zone, good for entries
                elif 30 <= rsi < 40:
                    score += 0.20  # Oversold bounce potential
                elif rsi < 30:
                    score += 0.10  # Deep oversold (risky but opportunity)
                elif rsi > 70:
                    score -= 0.20  # Overbought, bad for long entry
            else:
                if rsi > 70:
                    score += 0.20
                elif rsi < 30:
                    score -= 0.20

        if macd_hist is not None:
            if action == "BUY" and macd_hist > 0:
                score += 0.15
            elif action == "SELL" and macd_hist < 0:
                score += 0.15
            elif action == "BUY" and macd_hist < 0:
                score -= 0.10
            elif action == "SELL" and macd_hist > 0:
                score -= 0.10

        return round(max(0, min(1, score)), 4)

    def _rsi_zone(self, rsi: Optional[float]) -> str:
        if rsi is None:
            return "unknown"
        if rsi > 80:
            return "extreme_overbought"
        if rsi > 70:
            return "overbought"
        if rsi > 60:
            return "bullish"
        if rsi > 40:
            return "neutral"
        if rsi > 30:
            return "oversold"
        return "extreme_oversold"

    def _detect_rsi_divergence(self, data: Dict) -> str:
        """Simplified divergence detection from available data."""
        rsi = data.get("rsi")
        if rsi is None:
            return "none"
        # Would need historical data for proper divergence
        # This is a placeholder for the architecture
        return "none"

    def _detect_macd_crossover(self, data: Dict) -> str:
        macd = data.get("macd")
        signal = data.get("macd_signal")
        if macd is None or signal is None:
            return "unknown"
        if macd > signal:
            return "bullish"
        return "bearish"

    def _macd_momentum(self, data: Dict) -> float:
        hist = data.get("macd_histogram")
        if hist is None:
            return 0
        return round(hist, 4)

    # ═══════════════════════════════════════════════════════
    # VOLUME ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _compute_volume_score(self, data: Dict) -> float:
        score = 0.5
        volume = data.get("volume")
        avg_volume = data.get("avg_volume")
        vwap = data.get("vwap")
        price = data.get("price", 0)

        if volume and avg_volume and avg_volume > 0:
            ratio = volume / avg_volume
            if ratio > 2.0:
                score += 0.25  # Significant volume
            elif ratio > 1.5:
                score += 0.15
            elif ratio > 1.0:
                score += 0.05
            elif ratio < 0.5:
                score -= 0.15  # Low volume, weak signal

        if vwap and price:
            action = data.get("action", "BUY")
            if action == "BUY" and price > vwap:
                score += 0.10  # Buying above VWAP = institutional support
            elif action == "SELL" and price < vwap:
                score += 0.10

        return round(max(0, min(1, score)), 4)

    def _volume_ratio(self, data: Dict) -> float:
        volume = data.get("volume")
        avg = data.get("avg_volume")
        if not volume or not avg or avg == 0:
            return 1.0
        return round(volume / avg, 2)

    def _vwap_deviation(self, data: Dict) -> float:
        price = data.get("price", 0)
        vwap = data.get("vwap")
        if not price or not vwap or vwap == 0:
            return 0
        return round((price - vwap) / vwap, 4)

    def _is_volume_breakout(self, data: Dict) -> bool:
        volume = data.get("volume")
        avg = data.get("avg_volume")
        if not volume or not avg:
            return False
        return volume > avg * 1.5

    # ═══════════════════════════════════════════════════════
    # VOLATILITY ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _classify_volatility(self, data: Dict) -> str:
        atr_pct = self._atr_percent(data)
        if atr_pct == 0:
            return "unknown"
        if atr_pct > 4:
            return "extreme"
        if atr_pct > 2.5:
            return "high"
        if atr_pct > 1.0:
            return "normal"
        return "low"

    def _atr_percent(self, data: Dict) -> float:
        atr = data.get("atr")
        price = data.get("price", 0)
        if not atr or not price or price == 0:
            return 0
        return round((atr / price) * 100, 2)

    def _compute_volatility_score(self, data: Dict) -> float:
        """Score 0-1: Optimal volatility for trading (not too high, not too low)."""
        atr_pct = self._atr_percent(data)
        if atr_pct == 0:
            return 0.5
        if 1.0 <= atr_pct <= 3.0:
            return 0.8  # Ideal trading range
        if 0.5 <= atr_pct < 1.0:
            return 0.6  # Low but tradeable
        if 3.0 < atr_pct <= 5.0:
            return 0.5  # High, risky but opportunity
        if atr_pct > 5.0:
            return 0.3  # Too volatile
        return 0.4

    # ═══════════════════════════════════════════════════════
    # SUPPORT / RESISTANCE ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _analyze_sr_levels(self, data: Dict) -> Dict[str, Any]:
        price = data.get("price", 0)
        support = data.get("support")
        resistance = data.get("resistance")

        return {
            "has_support": support is not None,
            "has_resistance": resistance is not None,
            "support_distance": self._distance_to_level(data, "support"),
            "resistance_distance": self._distance_to_level(data, "resistance"),
            "in_compression": self._in_compression_zone(data),
        }

    def _distance_to_level(self, data: Dict, level_key: str) -> float:
        price = data.get("price", 0)
        level = data.get(level_key)
        if not price or not level or price == 0:
            return 0
        return round((price - level) / price, 4)

    def _near_psychological_level(self, price: float) -> bool:
        if price <= 0:
            return False
        # Check proximity to round numbers
        for base in [10, 50, 100, 500, 1000]:
            nearest = round(price / base) * base
            if abs(price - nearest) / price < 0.01:  # Within 1%
                return True
        return False

    def _in_resistance_zone(self, data: Dict) -> bool:
        price = data.get("price", 0)
        resistance = data.get("resistance")
        if not price or not resistance:
            return False
        distance = abs(price - resistance) / price
        return distance < 0.015  # Within 1.5% of resistance

    def _in_compression_zone(self, data: Dict) -> bool:
        support = data.get("support")
        resistance = data.get("resistance")
        price = data.get("price", 0)
        if not support or not resistance or not price or price == 0:
            return False
        range_pct = (resistance - support) / price
        return range_pct < 0.03  # Less than 3% range

    def get_feature_names(self) -> List[str]:
        """Return list of numeric feature names for ML model."""
        return [
            "trend_score", "momentum_score", "volume_score", "volatility_score",
            "price_vs_ema200", "price_vs_ema50", "price_vs_ema20",
            "ma_spread", "rsi", "macd_histogram", "volume_ratio",
            "vwap_deviation", "atr_percent", "distance_to_support",
            "distance_to_resistance", "composite_technical_score",
            "macro_sentiment", "dxy_trend",
        ]
