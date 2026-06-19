"""Load the baked-in XGBoost model and run inference on pre-computed features."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from forex_predict_mcp.tickers import make_ticker

MODELS_DIR = Path(__file__).parent / "models"


def get_forex_signal(ticker_symbol: str, features: dict[str, float]) -> dict[str, Any]:
    """Run the XGBoost model for the given ticker with pre-computed features.

    Returns {ticker, as_of, signal, prob_up, confidence} on success,
    or {error: reason} on failure.
    """
    try:
        ticker = make_ticker(ticker_symbol)
    except ValueError as exc:
        return {"error": str(exc)}

    missing = [col for col in ticker.feature_cols if col not in features]
    if missing:
        return {"error": f"Missing required features: {missing}"}

    slug = ticker_symbol.lower().replace("=x", "").replace(".", "")
    model_path = MODELS_DIR / slug / "model.json"

    if not model_path.exists():
        return {"error": f"Model file not found for {ticker_symbol} at {model_path}"}

    bst = xgb.Booster()
    bst.load_model(str(model_path))

    values = np.array([[features[col] for col in ticker.feature_cols]], dtype=float)
    dmatrix = xgb.DMatrix(values, feature_names=ticker.feature_cols)
    prob_up: float = float(bst.predict(dmatrix)[0])
    confidence = max(prob_up, 1.0 - prob_up)

    return {
        "ticker": ticker_symbol,
        "as_of": str(date.today()),
        "signal": "UP" if prob_up >= 0.5 else "DOWN",
        "prob_up": round(prob_up, 4),
        "confidence": round(confidence, 4),
    }
