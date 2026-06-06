"""
Live WebSocket feeds — replaces REST polling for real-time data.

Two independent feeds, each in its own background thread with auto-reconnect
(exponential backoff):

  • BinanceFeed   — live BTC/USDT price + 15m kline closes
  • PolymarketFeed — live CLOB order-book best bid/ask for tracked tokens

Both publish into a thread-safe LiveState singleton that the trader and
dashboard read without blocking. If a feed goes stale (no message within
WS_STALE_SECONDS) callers can fall back to REST.
"""
import json
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Set, Callable

import websocket  # websocket-client

import config

log = logging.getLogger(__name__)


# ── Shared thread-safe state ─────────────────────────────────────────────────

class LiveState:
    """In-memory snapshot updated by the WS threads, read by everyone else."""

    def __init__(self):
        self._lock = threading.RLock()
        self.btc_price: Optional[float] = None
        self.btc_price_ts: float = 0.0
        # token_id -> {"bid": float, "ask": float, "mid": float, "ts": float}
        self._books: Dict[str, Dict] = {}

    # BTC price
    def set_btc_price(self, price: float):
        with self._lock:
            self.btc_price = price
            self.btc_price_ts = time.time()

    def get_btc_price(self) -> Optional[float]:
        with self._lock:
            if self.btc_price is None:
                return None
            if time.time() - self.btc_price_ts > config.WS_STALE_SECONDS:
                return None
            return self.btc_price

    # Order books
    def set_book(self, token_id: str, bid: float, ask: float):
        with self._lock:
            mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
            self._books[token_id] = {"bid": bid, "ask": ask, "mid": mid, "ts": time.time()}

    def get_mid(self, token_id: str) -> Optional[float]:
        with self._lock:
            b = self._books.get(token_id)
            if not b:
                return None
            if time.time() - b["ts"] > config.WS_STALE_SECONDS:
                return None
            return b["mid"]

    def get_book(self, token_id: str) -> Optional[Dict]:
        with self._lock:
            return dict(self._books[token_id]) if token_id in self._books else None


# Module-level singleton
LIVE = LiveState()


# ── Base reconnecting WS thread ──────────────────────────────────────────────

class _ReconnectingWS(threading.Thread):
    def __init__(self, url: str, name: str):
        super().__init__(daemon=True, name=name)
        self.url = url
        self._stop = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._attempt = 0

    def stop(self):
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    # subclasses override these
    def on_open(self, ws):  ...
    def on_message(self, ws, message):  ...

    def _on_error(self, ws, error):
        log.warning(f"[{self.name}] WS error: {error}")

    def _on_close(self, ws, code, msg):
        log.info(f"[{self.name}] WS closed ({code}).")

    def run(self):
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._attempt = 0
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                log.warning(f"[{self.name}] run_forever crashed: {e}")

            if self._stop.is_set():
                break
            # exponential backoff, capped
            self._attempt = min(self._attempt + 1, config.WS_RECONNECT_MAX)
            delay = 2 ** self._attempt
            log.info(f"[{self.name}] reconnecting in {delay}s...")
            self._stop.wait(delay)


# ── Binance BTC feed ─────────────────────────────────────────────────────────

class BinanceFeed(_ReconnectingWS):
    def __init__(self, symbol: str = "btcusdt"):
        # combined stream: trade ticker + kline (interval matches CANDLE_INTERVAL)
        kline_iv = getattr(config, "WS_KLINE_INTERVAL", "1m")
        url = f"{config.BINANCE_WS}/{symbol}@trade/{symbol}@kline_{kline_iv}"
        super().__init__(url, name="binance-ws")
        self.symbol = symbol
        self.on_kline_close: Optional[Callable[[dict], None]] = None

    def on_open(self, ws):
        log.info("[binance-ws] connected — streaming BTC price + klines.")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            etype = data.get("e")
            if etype == "trade":
                LIVE.set_btc_price(float(data["p"]))
            elif etype == "kline":
                k = data["k"]
                LIVE.set_btc_price(float(k["c"]))
                if k.get("x") and self.on_kline_close:   # candle closed
                    self.on_kline_close({
                        "timestamp": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
                        "open": float(k["o"]), "high": float(k["h"]),
                        "low": float(k["l"]), "close": float(k["c"]),
                        "volume": float(k["v"]),
                    })
        except Exception as e:
            log.debug(f"[binance-ws] parse error: {e}")


# ── Polymarket CLOB order-book feed ──────────────────────────────────────────

class PolymarketFeed(_ReconnectingWS):
    """Subscribes to a dynamic set of asset (token) ids for live best bid/ask."""

    def __init__(self):
        super().__init__(config.POLYMARKET_WS, name="polymarket-ws")
        self._tokens: Set[str] = set()
        self._tokens_lock = threading.Lock()

    def set_tokens(self, token_ids):
        """Replace the subscription set; reconnects to apply (simple + robust)."""
        with self._tokens_lock:
            new = {t for t in token_ids if t}
            if new == self._tokens:
                return
            self._tokens = new
        # bounce the socket so the new subscription message is sent on_open
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def on_open(self, ws):
        with self._tokens_lock:
            tokens = list(self._tokens)
        if not tokens:
            log.info("[polymarket-ws] connected (no tokens subscribed yet).")
            return
        ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))
        log.info(f"[polymarket-ws] subscribed to {len(tokens)} token(s).")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            events = data if isinstance(data, list) else [data]
            for ev in events:
                token = ev.get("asset_id") or ev.get("market")
                if not token:
                    continue
                bids = ev.get("bids") or ev.get("buys") or []
                asks = ev.get("asks") or ev.get("sells") or []
                best_bid = max((float(b["price"]) for b in bids), default=0.0) if bids else 0.0
                best_ask = min((float(a["price"]) for a in asks), default=0.0) if asks else 0.0
                if best_bid or best_ask:
                    LIVE.set_book(token, best_bid, best_ask)
        except Exception as e:
            log.debug(f"[polymarket-ws] parse error: {e}")


# ── Feed manager ─────────────────────────────────────────────────────────────

class FeedManager:
    def __init__(self):
        self.binance = BinanceFeed()
        self.polymarket = PolymarketFeed()
        self._started = False

    def start(self, on_kline_close=None):
        if not config.ENABLE_WEBSOCKET or self._started:
            return
        self.binance.on_kline_close = on_kline_close
        self.binance.start()
        self.polymarket.start()
        self._started = True
        log.info("WebSocket feeds started.")

    def stop(self):
        self.binance.stop()
        self.polymarket.stop()
        self._started = False

    def subscribe_tokens(self, token_ids):
        self.polymarket.set_tokens(token_ids)
