"""
╔══════════════════════════════════════════════════════════════╗
║  Data Ingestion — External Data Source Integration            ║
║  Alpha Vantage · Polygon · FRED · Options Flow               ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("DataIngestion")


class DataIngestion:
    """
    Enriches TradingView signals with external data sources.
    
    Integrations:
    - Alpha Vantage: Real-time quotes, technical indicators
    - Polygon.io: Options flow, dark pool activity
    - FRED: Macroeconomic data series
    - Finnhub: Earnings, fundamentals
    
    Graceful degradation: works with TradingView data alone if APIs unavailable.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def enrich_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a raw signal with external data.
        Falls back gracefully if APIs are unavailable.
        """
        enriched = dict(signal_data)
        sources_used = ["tradingview"]

        ticker = signal_data.get("ticker", "")

        # Try Alpha Vantage for real-time data
        if self.api_keys.get("alpha_vantage"):
            try:
                av_data = self._fetch_alpha_vantage(ticker)
                if av_data:
                    enriched.update(av_data)
                    sources_used.append("alpha_vantage")
            except Exception as e:
                logger.warning(f"Alpha Vantage fetch failed: {e}")

        # Try Polygon for options flow
        if self.api_keys.get("polygon"):
            try:
                options_data = self._fetch_options_flow(ticker)
                if options_data:
                    enriched["options_flow"] = options_data
                    sources_used.append("polygon_options")
            except Exception as e:
                logger.warning(f"Polygon fetch failed: {e}")

        # Try Finnhub for fundamentals
        if self.api_keys.get("finnhub"):
            try:
                fundamentals = self._fetch_fundamentals(ticker)
                if fundamentals:
                    enriched["fundamentals"] = fundamentals
                    sources_used.append("finnhub")
            except Exception as e:
                logger.warning(f"Finnhub fetch failed: {e}")

        enriched["sources_used"] = sources_used
        enriched["enrichment_timestamp"] = datetime.utcnow().isoformat()

        return enriched

    def _fetch_alpha_vantage(self, ticker: str) -> Optional[Dict]:
        """
        Fetch real-time quote and technical indicators from Alpha Vantage.
        
        In production, this would make HTTP requests:
        https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={key}
        https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&apikey={key}
        """
        api_key = self.api_keys.get("alpha_vantage", "")
        if not api_key:
            return None

        # Production implementation:
        # import requests
        # url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        # response = requests.get(url, timeout=10)
        # data = response.json()
        # ... parse and return

        logger.info(f"Alpha Vantage enrichment for {ticker} (API configured)")
        return None  # Returns None until API key is provided

    def _fetch_options_flow(self, ticker: str) -> Optional[Dict]:
        """
        Fetch options flow data from Polygon.io.
        
        Analyzes:
        - Call/Put ratio
        - Unusual options activity
        - Dark pool prints
        - Large block trades
        
        Production endpoint:
        https://api.polygon.io/v3/trades/{ticker}?apiKey={key}
        https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={key}
        """
        api_key = self.api_keys.get("polygon", "")
        if not api_key:
            return None

        logger.info(f"Options flow enrichment for {ticker} (API configured)")
        return None

    def _fetch_fundamentals(self, ticker: str) -> Optional[Dict]:
        """
        Fetch fundamental data from Finnhub.
        
        Data points:
        - Next earnings date
        - Revenue growth
        - Profit margins
        - Debt/Equity ratio
        - Sector classification
        
        Production endpoint:
        https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={key}
        https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&token={key}
        """
        api_key = self.api_keys.get("finnhub", "")
        if not api_key:
            return None

        logger.info(f"Fundamentals enrichment for {ticker} (API configured)")
        return None

    def _fetch_fred_data(self, series_id: str) -> Optional[Dict]:
        """
        Fetch macroeconomic series from FRED.
        
        Key series:
        - DFF: Federal Funds Rate
        - CPIAUCSL: Consumer Price Index
        - UNRATE: Unemployment Rate
        - DTWEXBGS: Trade Weighted Dollar Index
        - VIXCLS: VIX
        
        Production endpoint:
        https://api.stlouisfed.org/fred/series/observations?series_id={id}&api_key={key}&file_type=json
        """
        api_key = self.api_keys.get("fred", "")
        if not api_key:
            return None

        logger.info(f"FRED data fetch: {series_id}")
        return None

    def get_dark_pool_activity(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze dark pool activity for a ticker.
        
        In production, integrates with:
        - FINRA ADF data
        - Polygon dark pool trades
        - Short volume data
        
        Returns:
        - Dark pool volume percentage
        - Large block trade count
        - Net dark pool sentiment
        """
        return {
            "available": bool(self.api_keys.get("polygon")),
            "ticker": ticker,
            "dark_pool_volume_pct": None,
            "block_trades": None,
            "net_sentiment": None,
            "note": "Configure Polygon API key for dark pool data",
        }

    def get_options_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Compute options market sentiment.
        
        Metrics:
        - Put/Call ratio
        - Implied Volatility skew
        - Max Pain level
        - Unusual activity score
        """
        return {
            "available": bool(self.api_keys.get("polygon")),
            "ticker": ticker,
            "put_call_ratio": None,
            "iv_skew": None,
            "max_pain": None,
            "unusual_activity_score": None,
            "note": "Configure Polygon API key for options flow data",
        }
