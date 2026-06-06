"""Fetch BTC OHLCV candles from Binance (no key required)."""
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from agent.database.models import get_session, BTCCandle
import config

log = logging.getLogger(__name__)

BINANCE_KLINES = f"{config.BINANCE_API}/api/v3/klines"


def fetch_candles(symbol: str = "BTCUSDT", interval: str = None,
                  limit: int = None) -> pd.DataFrame:
    interval = interval or config.CANDLE_INTERVAL
    limit    = limit or config.CANDLE_LIMIT
    """Return DataFrame with columns: timestamp, open, high, low, close, volume."""
    try:
        resp = requests.get(
            BINANCE_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.set_index("timestamp", inplace=True)
        return _resample(df)
    except Exception as e:
        log.error(f"Binance fetch error: {e}")
        return _load_from_db()


def _resample(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fine candles into custom-second buckets (e.g. 20s)."""
    secs = getattr(config, "CANDLE_RESAMPLE_SEC", 0)
    if not secs or df.empty:
        return df
    rule = f"{secs}s"
    agg = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return agg


def _load_from_db() -> pd.DataFrame:
    session = get_session()
    candles = session.query(BTCCandle).order_by(BTCCandle.timestamp.desc()).limit(200).all()
    session.close()
    if not candles:
        return pd.DataFrame()
    rows = [{"timestamp": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume} for c in reversed(candles)]
    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def store_candles(df: pd.DataFrame):
    session = get_session()
    try:
        for ts, row in df.iterrows():
            exists = session.query(BTCCandle).filter_by(
                timestamp=ts.to_pydatetime().replace(tzinfo=None)).first()
            if not exists:
                c = BTCCandle(
                    timestamp=ts.to_pydatetime().replace(tzinfo=None),
                    open=row.open, high=row.high,
                    low=row.low, close=row.close, volume=row.volume
                )
                session.add(c)
        session.commit()
    except Exception as e:
        session.rollback()
        log.error(f"store_candles error: {e}")
    finally:
        session.close()


def get_current_btc_price() -> float:
    # 1. Prefer the live WebSocket price (sub-second freshness)
    try:
        from agent.websocket_feed import LIVE
        live_price = LIVE.get_btc_price()
        if live_price:
            return live_price
    except Exception:
        pass
    # 2. Fall back to REST if the WS feed is stale/unavailable
    try:
        resp = requests.get(
            f"{config.BINANCE_API}/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"}, timeout=5
        )
        return float(resp.json()["price"])
    except Exception as e:
        log.error(f"Price fetch error: {e}")
        return 0.0
