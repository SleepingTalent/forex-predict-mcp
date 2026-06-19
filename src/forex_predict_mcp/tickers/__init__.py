"""Ticker subclasses — one per supported forex asset, each with hand-tuned features."""

from __future__ import annotations

from forex_predict_mcp.tickers.base import Ticker
from forex_predict_mcp.tickers.eurusd import EURUSDTicker
from forex_predict_mcp.tickers.gbpusd import GBPUSDTicker

_TICKER_MAP: dict[str, type[Ticker]] = {
    "EURUSD=X": EURUSDTicker,
    "GBPUSD=X": GBPUSDTicker,
}


def make_ticker(symbol: str) -> Ticker:
    """Return the asset-specific Ticker subclass for the given symbol.

    Raises ValueError for unrecognised symbols.
    """
    cls = _TICKER_MAP.get(symbol)
    if cls is None:
        supported = ", ".join(sorted(_TICKER_MAP))
        raise ValueError(f"Unsupported ticker '{symbol}'. Supported: {supported}")
    return cls(symbol)


__all__ = ["Ticker", "EURUSDTicker", "GBPUSDTicker", "make_ticker", "_TICKER_MAP"]
