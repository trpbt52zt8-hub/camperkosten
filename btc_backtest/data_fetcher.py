import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-Backtest/1.0)"
}


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_exc = None
    for url in BINANCE_ENDPOINTS:
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            raw = response.json()
            break
        except Exception as exc:
            last_exc = exc
            continue
    else:
        raise ConnectionError(
            f"All Binance endpoints failed. Last error: {last_exc}\n"
            "Tip: run with --demo to use synthetic data, or place btcusdt_1h.csv in data/"
        )
    raw = response.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic_data(limit: int) -> pd.DataFrame:
    """Generates realistic synthetic BTC OHLCV data using geometric Brownian motion."""
    print("Generating synthetic BTCUSDT data for demo/testing...")
    np.random.seed(42)
    n = limit
    dt = 1 / 8760
    mu = 0.5
    sigma = 0.8

    prices = [45000.0]
    for _ in range(n - 1):
        ret = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn())
        prices.append(prices[-1] * ret)

    end = pd.Timestamp.now().floor("h")
    index = pd.date_range(end=end, periods=n, freq="1h")

    rows = []
    for i, price in enumerate(prices):
        noise = abs(np.random.randn()) * price * 0.005
        high = price + noise
        low = price - noise
        open_ = prices[i - 1] if i > 0 else price
        vol = np.random.uniform(100, 2000)
        rows.append([open_, high, low, price, vol])

    df = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])
    return df


def get_data(config: dict, demo: bool = False) -> pd.DataFrame:
    cfg = config["data"]
    cache = cfg["cache_file"]

    if demo:
        df = generate_synthetic_data(cfg["limit"])
        print(f"Demo data: {len(df)} synthetic candles from {df.index[0]} to {df.index[-1]}")
        return df

    if os.path.exists(cache):
        print(f"Loading cached data from {cache}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        print(f"Fetching {cfg['limit']} {cfg['interval']} candles for {cfg['symbol']} from Binance...")
        try:
            df = fetch_klines(cfg["symbol"], cfg["interval"], cfg["limit"])
        except ConnectionError as e:
            print(f"\nWaarschuwing: {e}")
            print("Fallback: synthetische data wordt gebruikt.\n")
            df = generate_synthetic_data(cfg["limit"])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache)
        print(f"Cached to {cache}")

    print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df
