"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANTUM EDGE — Algorithmic Trading Intelligence Platform       ║
║  Senior Quant Architecture · ML-Powered Signal Processing       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import hmac
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from modules.risk_manager import RiskManager
from modules.feature_engine import FeatureEngine
from modules.ml_predictor import MLPredictor
from modules.signal_processor import SignalProcessor
from modules.macro_analyzer import MacroAnalyzer
from modules.data_ingestion import DataIngestion

# ─── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("QuantumEdge")

# ─── Configuration ─────────────────────────────────────────────
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-secret-key-change-me")
API_KEYS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY", ""),
    "polygon": os.getenv("POLYGON_KEY", ""),
    "fred": os.getenv("FRED_KEY", ""),
    "finnhub": os.getenv("FINNHUB_KEY", ""),
}

# ─── App Init ──────────────────────────────────────────────────
app = FastAPI(
    title="Quantum Edge Trading Platform",
    description="ML-Powered Algorithmic Trading Signal Processor",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Core Modules ──────────────────────────────────────────────
risk_manager = RiskManager(
    total_capital=100_000.0,
    max_risk_per_trade=0.02,
    max_portfolio_risk=0.06,
    max_correlation=0.85
)
feature_engine = FeatureEngine()
ml_predictor = MLPredictor(model_path="models/")
signal_processor = SignalProcessor(min_confidence=0.65)
macro_analyzer = MacroAnalyzer(api_keys=API_KEYS)
data_ingestion = DataIngestion(api_keys=API_KEYS)

# ─── In-Memory State ──────────────────────────────────────────
portfolio_state = {
    "capital": 100_000.0,
    "positions": [],
    "signals_history": [],
    "trades_executed": [],
    "daily_pnl": 0.0,
    "total_pnl": 0.0,
}


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class TradingViewSignal(BaseModel):
    ticker: str = Field(..., description="Asset symbol e.g. AAPL")
    action: str = Field(..., description="BUY or SELL")
    price: float = Field(..., description="Current price at signal")
    timeframe: str = Field(default="1H", description="Chart timeframe")
    strategy: Optional[str] = Field(default="custom", description="Strategy name")
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    volume: Optional[float] = None
    avg_volume: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    notes: Optional[str] = None


class ManualSignal(BaseModel):
    ticker: str
    action: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1H"
    notes: Optional[str] = None


class PortfolioUpdate(BaseModel):
    capital: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    max_portfolio_risk: Optional[float] = None
    max_correlation: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_signal_pipeline(signal: TradingViewSignal) -> Dict[str, Any]:
    """
    Master pipeline: Signal → Enrichment → Features → ML → Risk → Decision
    """
    logger.info(f"═══ Processing: {signal.ticker} {signal.action} @ ${signal.price:.2f} ═══")

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": signal.ticker,
        "action": signal.action,
        "entry_price": signal.price,
        "status": "PENDING",
        "stages": {}
    }

    # STAGE 1: Data Enrichment
    try:
        enriched_data = data_ingestion.enrich_signal(signal.dict())
        result["stages"]["data_ingestion"] = {"status": "OK", "sources": enriched_data.get("sources_used", [])}
    except Exception as e:
        logger.warning(f"Data ingestion partial failure: {e}")
        enriched_data = signal.dict()
        result["stages"]["data_ingestion"] = {"status": "PARTIAL", "error": str(e)}

    # STAGE 2: Feature Engineering
    features = feature_engine.compute_features(enriched_data)
    result["stages"]["feature_engineering"] = {
        "status": "OK",
        "features_count": len(features),
        "key_features": {
            "trend_score": features.get("trend_score", 0),
            "momentum_score": features.get("momentum_score", 0),
            "volume_score": features.get("volume_score", 0),
            "volatility_regime": features.get("volatility_regime", "unknown"),
        }
    }

    # STAGE 3: Macro Analysis
    macro_context = macro_analyzer.get_macro_context()
    features["macro_sentiment"] = macro_context.get("overall_sentiment", 0)
    features["dxy_trend"] = macro_context.get("dxy_trend", 0)
    features["rate_environment"] = macro_context.get("rate_environment", "neutral")
    result["stages"]["macro_analysis"] = {
        "status": "OK",
        "sentiment": macro_context.get("overall_sentiment", 0),
        "environment": macro_context.get("rate_environment", "neutral")
    }

    # STAGE 4: ML Prediction
    prediction = ml_predictor.predict(features, signal.ticker, signal.action)
    win_probability = prediction["win_probability"]
    result["stages"]["ml_prediction"] = {
        "status": "OK",
        "win_probability": round(win_probability, 4),
        "confidence_band": prediction.get("confidence_band", [0, 0]),
        "model_version": prediction.get("model_version", "1.0"),
        "feature_importance": prediction.get("top_features", {})
    }

    # STAGE 5: AI Rules Check
    ai_rules_result = signal_processor.apply_ai_rules(
        win_probability=win_probability,
        features=features,
        signal=signal.dict(),
        portfolio=portfolio_state
    )
    result["stages"]["ai_rules"] = ai_rules_result

    if not ai_rules_result["passed"]:
        result["status"] = "REJECTED"
        result["rejection_reasons"] = ai_rules_result["violations"]
        result["win_probability"] = round(win_probability, 4)
        logger.warning(f"Signal REJECTED: {ai_rules_result['violations']}")
        portfolio_state["signals_history"].append(result)
        return result

    # STAGE 6: Risk Management
    risk_calc = risk_manager.calculate_position(
        ticker=signal.ticker,
        entry_price=signal.price,
        action=signal.action,
        atr=features.get("atr", signal.atr),
        support=signal.support,
        resistance=signal.resistance,
        win_probability=win_probability
    )
    result["stages"]["risk_management"] = risk_calc
    result["position_size"] = risk_calc["position_size"]
    result["stop_loss"] = risk_calc["stop_loss"]
    result["take_profit"] = risk_calc["take_profit"]
    result["risk_reward_ratio"] = risk_calc["risk_reward_ratio"]
    result["dollar_risk"] = risk_calc["dollar_risk"]

    # FINAL DECISION
    result["status"] = "APPROVED"
    result["win_probability"] = round(win_probability, 4)
    result["overall_score"] = round(
        (win_probability * 0.4) +
        (features.get("trend_score", 0) * 0.2) +
        (features.get("momentum_score", 0) * 0.2) +
        (features.get("volume_score", 0) * 0.2), 4
    )

    portfolio_state["signals_history"].append(result)
    if len(portfolio_state["signals_history"]) > 500:
        portfolio_state["signals_history"] = portfolio_state["signals_history"][-500:]

    logger.info(f"Signal APPROVED: {signal.ticker} | Win: {win_probability:.2%} | "
                f"Size: {risk_calc['position_size']} shares | R:R {risk_calc['risk_reward_ratio']:.2f}")
    return result


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "portfolio": portfolio_state})


@app.post("/webhook/tradingview")
async def receive_tradingview_webhook(request: Request):
    try:
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        secret = payload.pop("secret", None)
        if secret and not hmac.compare_digest(str(secret), WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Invalid secret")

        signal = TradingViewSignal(**payload)
        result = process_signal_pipeline(signal)
        return JSONResponse(content=result, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/signal/manual")
async def manual_signal(signal: ManualSignal):
    tv_signal = TradingViewSignal(
        ticker=signal.ticker, action=signal.action.upper(),
        price=signal.price, timeframe=signal.timeframe, notes=signal.notes
    )
    return process_signal_pipeline(tv_signal)


@app.get("/api/portfolio")
async def get_portfolio():
    return {
        "capital": portfolio_state["capital"],
        "positions": portfolio_state["positions"],
        "daily_pnl": portfolio_state["daily_pnl"],
        "total_pnl": portfolio_state["total_pnl"],
        "active_positions_count": len(portfolio_state["positions"]),
        "signals_today": len([
            s for s in portfolio_state["signals_history"]
            if s.get("timestamp", "")[:10] == datetime.utcnow().strftime("%Y-%m-%d")
        ])
    }


@app.get("/api/signals/history")
async def get_signals_history(limit: int = 50):
    return {"signals": portfolio_state["signals_history"][-limit:][::-1], "total": len(portfolio_state["signals_history"])}


@app.post("/api/portfolio/update")
async def update_portfolio(update: PortfolioUpdate):
    if update.capital is not None:
        portfolio_state["capital"] = update.capital
        risk_manager.total_capital = update.capital
    if update.max_risk_per_trade is not None:
        risk_manager.max_risk_per_trade = update.max_risk_per_trade
    if update.max_portfolio_risk is not None:
        risk_manager.max_portfolio_risk = update.max_portfolio_risk
    if update.max_correlation is not None:
        risk_manager.max_correlation = update.max_correlation
    return {"status": "updated", "portfolio": portfolio_state}


@app.get("/api/model/status")
async def model_status():
    return ml_predictor.get_model_status()


@app.post("/api/model/retrain")
async def retrain_model():
    try:
        result = ml_predictor.retrain()
        return {"status": "success", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/macro")
async def get_macro_context():
    return macro_analyzer.get_macro_context()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": ml_predictor.is_loaded(), "version": "2.0.0"}


@app.on_event("startup")
async def startup():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   QUANTUM EDGE Trading Platform v2.0        ║")
    logger.info("║   ML-Powered Signal Processing Active       ║")
    logger.info("╚══════════════════════════════════════════════╝")
    try:
        ml_predictor.load_or_train()
        logger.info("ML Model loaded successfully")
    except Exception as e:
        logger.warning(f"ML Model init note: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
