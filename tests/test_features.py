"""Tests for features.py — get_market_features tool."""

from __future__ import annotations

import pytest

from forex_predict_mcp.features import get_market_features
from forex_predict_mcp.tickers import EURUSDTicker, GBPUSDTicker


def test_eurusd_returns_correct_structure(mock_yf_and_fetch: None) -> None:
    result = get_market_features("EURUSD=X")

    assert "error" not in result
    assert result["ticker"] == "EURUSD=X"
    assert "as_of" in result
    assert "features" in result


def test_eurusd_returns_eleven_features(mock_yf_and_fetch: None) -> None:
    result = get_market_features("EURUSD=X")

    assert set(result["features"].keys()) == set(EURUSDTicker("EURUSD=X").feature_cols)
    assert len(result["features"]) == 11


def test_gbpusd_returns_thirteen_features(mock_yf_and_fetch: None) -> None:
    result = get_market_features("GBPUSD=X")

    assert set(result["features"].keys()) == set(GBPUSDTicker("GBPUSD=X").feature_cols)
    assert len(result["features"]) == 13


def test_all_feature_values_are_floats(mock_yf_and_fetch: None) -> None:
    result = get_market_features("EURUSD=X")

    for name, value in result["features"].items():
        assert isinstance(value, float), f"{name} is not a float: {type(value)}"


def test_unsupported_ticker_returns_error_dict(mock_yf_and_fetch: None) -> None:
    result = get_market_features("UNKNOWN")

    assert "error" in result
    assert "UNKNOWN" in result["error"]


def test_yfinance_failure_returns_error_dict(mocker: pytest.MockerFixture) -> None:
    mocker.patch(
        "forex_predict_mcp.features.yf.Ticker",
        side_effect=RuntimeError("network error"),
    )

    result = get_market_features("EURUSD=X")

    assert "error" in result
    assert "market data" in result["error"]
