"""Tests for predict.py — get_forex_signal tool."""

from __future__ import annotations

from forex_predict_mcp.predict import get_forex_signal
from forex_predict_mcp.tickers import EURUSDTicker, GBPUSDTicker

EURUSD_FEATURES = {col: 0.01 for col in EURUSDTicker("EURUSD=X").feature_cols}
GBPUSD_FEATURES = {col: 0.01 for col in GBPUSDTicker("GBPUSD=X").feature_cols}


def test_eurusd_signal_returns_correct_structure() -> None:
    result = get_forex_signal("EURUSD=X", EURUSD_FEATURES)

    assert "error" not in result
    assert result["ticker"] == "EURUSD=X"
    assert "as_of" in result
    assert result["signal"] in ("UP", "DOWN")
    assert "prob_up" in result
    assert "confidence" in result


def test_gbpusd_signal_returns_correct_structure() -> None:
    result = get_forex_signal("GBPUSD=X", GBPUSD_FEATURES)

    assert "error" not in result
    assert result["ticker"] == "GBPUSD=X"
    assert result["signal"] in ("UP", "DOWN")


def test_signal_is_up_when_prob_up_above_half() -> None:
    result = get_forex_signal("EURUSD=X", EURUSD_FEATURES)

    if result["prob_up"] >= 0.5:
        assert result["signal"] == "UP"
    else:
        assert result["signal"] == "DOWN"


def test_confidence_is_max_of_prob_and_complement() -> None:
    result = get_forex_signal("EURUSD=X", EURUSD_FEATURES)

    prob_up: float = result["prob_up"]
    expected = max(prob_up, 1.0 - prob_up)
    assert abs(result["confidence"] - round(expected, 4)) < 1e-6


def test_missing_features_returns_error_dict() -> None:
    incomplete = {"return_5d": 0.01}

    result = get_forex_signal("EURUSD=X", incomplete)

    assert "error" in result
    assert "Missing" in result["error"]


def test_unsupported_ticker_returns_error_dict() -> None:
    result = get_forex_signal("UNKNOWN", EURUSD_FEATURES)

    assert "error" in result
    assert "UNKNOWN" in result["error"]
