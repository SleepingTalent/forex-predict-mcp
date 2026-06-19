# forex-predict-mcp

A FastMCP server that exposes XGBoost directional predictions for EURUSD=X and GBPUSD=X as MCP tools. Packaged as a Docker stdio image — no env vars required.

Designed to run alongside the [oanda-mcp-server](https://github.com/SleepingTalent/oanda-mcp-server) as a tool for a Claude trading agent.

## Tools

### `get_market_features(ticker)`

Fetches ~1 year of daily OHLCV data from Yahoo Finance plus cross-asset data (DXY, VIX, EURUSD), computes the full feature vector, and returns it for inspection.

**Parameters:**
- `ticker` — `"EURUSD=X"` or `"GBPUSD=X"`

**Response:**
```json
{
  "ticker": "EURUSD=X",
  "as_of": "2026-06-19",
  "features": {
    "return_5d": 0.0124,
    "return_10d": 0.0231,
    "return_20d": -0.0052,
    "rsi_14": 58.3,
    "atr_14": 0.00234,
    "macd_hist": 0.00012,
    "sma50_ratio": 1.003,
    "dxy_return_5d": -0.0081,
    "dxy_return_20d": -0.0152,
    "vix_return_5d": 0.122,
    "vix_vs_sma20": -0.048
  }
}
```

GBPUSD=X additionally includes `eurusd_return_5d` and `eurusd_return_20d`.

---

### `get_forex_signal(ticker, features)`

Runs the baked-in XGBoost model against pre-computed features and returns a directional signal.

**Parameters:**
- `ticker` — `"EURUSD=X"` or `"GBPUSD=X"`
- `features` — the `features` dict from `get_market_features()`

**Response:**
```json
{
  "ticker": "EURUSD=X",
  "as_of": "2026-06-19",
  "signal": "UP",
  "prob_up": 0.773,
  "confidence": 0.773
}
```

---

## Typical agent workflow

```
1. get_market_features("EURUSD=X")  →  inspect RSI, DXY, VIX values
2. get_forex_signal("EURUSD=X", features)  →  UP 77.3%
3. cross-reference with Oanda live price and open positions
4. place or skip order
```

## `.mcp.json` configuration

```json
"forex-predict-mcp": {
  "type": "stdio",
  "command": "docker",
  "args": ["run", "--rm", "-i", "sleepingtalent/forex-predict-mcp:latest"]
}
```

## Models

XGBoost binary classifiers trained on 5 years of daily data:

| Ticker | Features | Test accuracy |
|--------|----------|--------------|
| EURUSD=X | 11 (returns, RSI, ATR, MACD, SMA50, DXY, VIX) | 68.80% |
| GBPUSD=X | 13 (same + EURUSD cross-asset returns) | 67.60% |

Models are baked into the Docker image at build time. To update: retrain in [weights-biases-example](https://github.com/SleepingTalent/weights-and-biases-example), run `uv run task export_models`, copy the JSON files into `src/forex_predict_mcp/models/`, and push — CI publishes a new tagged image automatically.
