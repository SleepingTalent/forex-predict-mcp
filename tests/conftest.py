"""Shared pytest fixtures for forex-predict-mcp tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with a DatetimeIndex, no timezone."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 1.08 + rng.normal(0, 0.005, n).cumsum()
    high = close + rng.uniform(0.001, 0.005, n)
    low = close - rng.uniform(0.001, 0.005, n)
    open_ = close + rng.normal(0, 0.002, n)
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture()
def mock_yf(mocker: MockerFixture) -> None:
    """Mock yfinance.Ticker in tickers/base.py for all feature-fetch tests."""
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("forex_predict_mcp.tickers.base.yf.Ticker", return_value=ticker_mock)


@pytest.fixture()
def mock_yf_and_fetch(mocker: MockerFixture) -> None:
    """Mock both yfinance.Ticker (cross-asset) and the top-level fetch in features.py."""
    ticker_mock = mocker.MagicMock()
    ticker_mock.history.return_value = _make_ohlcv()
    mocker.patch("forex_predict_mcp.tickers.base.yf.Ticker", return_value=ticker_mock)
    mocker.patch("forex_predict_mcp.features.yf.Ticker", return_value=ticker_mock)
