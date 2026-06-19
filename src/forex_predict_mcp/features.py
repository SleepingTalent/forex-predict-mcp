"""Fetch market data and compute the feature vector for a forex ticker."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from forex_predict_mcp.tickers import make_ticker


def _fetch_data(symbol: str, lookback_years: int = 1) -> pd.DataFrame:
    """Fetch daily OHLCV data for the given symbol via yfinance."""
    start = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    t = yf.Ticker(symbol)
    df: pd.DataFrame = t.history(start=start, end=end)
    return df


def get_market_features(ticker_symbol: str) -> dict[str, Any]:
    """Fetch and compute the full feature vector for the given forex ticker.

    Returns {ticker, as_of, features: {name: value}} on success,
    or {error: reason} on failure.
    """
    try:
        ticker = make_ticker(ticker_symbol)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        df = _fetch_data(ticker_symbol)
    except Exception as exc:
        return {"error": f"Failed to fetch market data: {exc}"}

    df = ticker.features(df)
    df = df.dropna(subset=ticker.feature_cols)

    if df.empty:
        return {"error": "No data available after feature computation"}

    latest = df[ticker.feature_cols].iloc[-1]
    as_of = str(df.index[-1].date())

    return {
        "ticker": ticker_symbol,
        "as_of": as_of,
        "features": {col: float(latest[col]) for col in ticker.feature_cols},
    }
