"""FastMCP server — tool registration and stdio entrypoint."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

import forex_predict_mcp.features as _features
import forex_predict_mcp.predict as _predict

mcp = FastMCP("Forex Predict MCP")


@mcp.tool
def get_market_features(ticker: str) -> dict[str, Any]:
    """Fetch and compute the feature vector for a forex pair.

    ticker: 'EURUSD=X' or 'GBPUSD=X'

    Returns {ticker, as_of, features: {feature_name: value}} on success,
    or {error: reason} if the ticker is unsupported or data cannot be fetched.
    Call this first, inspect the features, then pass them to get_forex_signal.
    """
    return _features.get_market_features(ticker)


@mcp.tool
def get_forex_signal(ticker: str, features: dict[str, float]) -> dict[str, Any]:
    """Run the XGBoost model with pre-computed features from get_market_features.

    ticker: 'EURUSD=X' or 'GBPUSD=X'
    features: the features dict returned by get_market_features()

    Returns {ticker, as_of, signal (UP/DOWN), prob_up, confidence} on success,
    or {error: reason} if features are missing or the ticker is unsupported.
    """
    return _predict.get_forex_signal(ticker, features)


def main() -> None:
    """Run the MCP server using stdio transport (default)."""
    mcp.run()


if __name__ == "__main__":
    main()
