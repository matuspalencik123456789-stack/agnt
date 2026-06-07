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

# Short-lived candle cache so a fast (2s) polling loop doesn't hammer Binance
# REST and risk an IP ban. Live price stays sub-second via the WebSocket feed;
# candle-based indicators don't need to refresh more often than this.
_CANDLE_CACHE: dict = {"df": None, "ts": 0.0, "key": None}
_CANDLE_CACHE_SEC = int(getattr(config, "CANDLE_CACHE_SEC", 5))


def fetch_candles(symbol: str = "BTCUSDT", interval: str = None,
                  limit: int = None) -> pd.DataFrame:
    interval = interval or config.CANDLE_INTERVAL
    limit    = limit or config.CANDLE_LIMIT
    """Return DataFrame with columns: timestamp, open, high, low, close, volume."""
    key = (symbol, interval, limit)
    if (_CANDLE_CACHE["df"] is not None and _CANDLE_CACHE["key"] == key
            and (time.time() - _CANDLE_CACHE["ts"]) < _CANDLE_CACHE_SEC):
        return _CANDLE_CACHE["df"]
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
        out = _resample(df)
        _CANDLE_CACHE.update({"df": out, "ts": time.time(), "key": key})
        return out
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


def get_price_at(dt) -> float:
    """
    BTC close price AROUND a past timestamp `dt` (UTC), from the 1-minute kline
    covering it. Used to resolve a paper trade against the price AT the window's
    end rather than 'now' — important when resolution runs late (after a restart
    or a busy period). Returns 0.0 if unavailable.
    """
    try:
        import pandas as _pd
        ts = _pd.Timestamp(dt).tz_localize(None) if _pd.Timestamp(dt).tzinfo is None \
            else _pd.Timestamp(dt).tz_convert("UTC").tz_localize(None)
        start_ms = int(ts.timestamp() * 1000)
        resp = requests.get(
            BINANCE_KLINES,
            params={"symbol": "BTCUSDT", "interval": "1m",
                    "startTime": start_ms, "limit": 1},
            timeout=8,
        )
        resp.raise_for_status()
        raw = resp.json()
        if raw:
            return float(raw[0][4])   # close of the covering 1m candle
    except Exception as e:
        log.debug(f"get_price_at error: {e}")
    return 0.0


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
