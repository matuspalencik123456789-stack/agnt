"""Polymarket CLOB API client with BTC market discovery."""
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

import config

log = logging.getLogger(__name__)


class PolymarketClient:
    def __init__(self):
        self._gamma = config.GAMMA_API
        self._clob_host = config.CLOB_HOST
        self._client: Optional[ClobClient] = None
        self._init_client()

    def _init_client(self):
        if not config.POLYMARKET_PRIVATE_KEY:
            log.warning("No Polymarket private key — running in read-only mode.")
            return
        try:
            creds = ApiCreds(
                api_key=config.POLYMARKET_API_KEY,
                api_secret=config.POLYMARKET_SECRET,
                api_passphrase=config.POLYMARKET_PASSPHRASE,
            )
            self._client = ClobClient(
                host=self._clob_host,
                chain_id=137,           # Polygon mainnet
                key=config.POLYMARKET_PRIVATE_KEY,
                creds=creds,
                signature_type=2,       # POLY_GNOSIS_SAFE
            )
            log.info("Polymarket CLOB client initialized.")
        except Exception as e:
            log.error(f"CLOB client init error: {e}")

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_btc_markets(self, limit: int = 20) -> List[Dict]:
        """Return active BTC markets from Gamma API."""
        markets = []
        for kw in config.BTC_MARKET_KEYWORDS:
            try:
                resp = requests.get(
                    f"{self._gamma}/markets",
                    params={"q": kw, "active": "true", "closed": "false",
                            "limit": limit},
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("markets", [])
                for m in data:
                    q = m.get("question", "").lower()
                    if any(k in q for k in ["btc", "bitcoin"]) and \
                       ("above" in q or "below" in q or "exceed" in q):
                        markets.append(m)
            except Exception as e:
                log.error(f"Gamma API error ({kw}): {e}")

        # de-duplicate by conditionId
        seen = set()
        unique = []
        for m in markets:
            cid = m.get("conditionId", m.get("id", ""))
            if cid not in seen:
                seen.add(cid)
                unique.append(m)
        return unique

    def enrich_market(self, market: Dict) -> Dict:
        """Add strike_price and parsed metadata to market dict."""
        q = market.get("question", "")
        import re
        prices = re.findall(r"\$?([\d,]+(?:\.\d+)?)", q.replace(",", ""))
        strike = float(prices[0].replace(",", "")) if prices else 0.0
        market["strike_price"] = strike
        return market

    # ── Order book ───────────────────────────────────────────────────────────

    def get_book(self, token_id: str) -> Optional[Dict]:
        try:
            if self._client:
                book = self._client.get_order_book(token_id)
                return book
            # fallback: REST
            resp = requests.get(
                f"{self._clob_host}/book",
                params={"token_id": token_id}, timeout=8
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"get_book error: {e}")
            return None

    def get_mid_price(self, token_id: str) -> Optional[float]:
        """Return mid price — live WebSocket book first, REST as fallback."""
        try:
            from agent.websocket_feed import LIVE
            live_mid = LIVE.get_mid(token_id)
            if live_mid:
                return live_mid
        except Exception:
            pass
        book = self.get_book(token_id)
        if not book:
            return None
        try:
            bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
            asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
            if bids and asks:
                return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
        except Exception:
            pass
        return None

    def get_market_prices(self, market: Dict) -> tuple[float, float]:
        """Return (yes_price, no_price) for a market."""
        tokens = market.get("tokens", market.get("clobTokenIds", []))
        yes_token = no_token = None

        if isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict):
                    outcome = t.get("outcome", "").upper()
                    if outcome == "YES":
                        yes_token = t.get("token_id", t.get("tokenId", ""))
                    elif outcome == "NO":
                        no_token = t.get("token_id", t.get("tokenId", ""))

        yes_price = self.get_mid_price(yes_token) if yes_token else None
        no_price  = self.get_mid_price(no_token)  if no_token  else None

        # fallback to gamma prices
        if yes_price is None:
            yes_price = float(market.get("outcomePrices", [0.5, 0.5])[0])
        if no_price is None:
            no_price  = float(market.get("outcomePrices", [0.5, 0.5])[1])

        return yes_price, no_price

    # ── Order placement ──────────────────────────────────────────────────────

    def place_market_order(self, token_id: str, side: str,
                           size_usd: float, price: float) -> Optional[str]:
        """Place a market order. Returns order_id or None."""
        if not self._client:
            log.warning("No CLOB client — paper trade only.")
            return f"PAPER_{int(time.time())}"
        try:
            size = round(size_usd / price, 4)
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=size,
                side=side.lower(),
            )
            resp = self._client.create_and_post_order(order_args)
            order_id = resp.get("orderID", "")
            log.info(f"Order placed: {side} {size} @ {price} → {order_id}")
            return order_id
        except Exception as e:
            log.error(f"place_order error: {e}")
            return None

    def get_portfolio(self) -> List[Dict]:
        if not self._client:
            return []
        try:
            return self._client.get_positions() or []
        except Exception as e:
            log.error(f"get_portfolio error: {e}")
            return []

    def get_balance(self) -> float:
        if not self._client:
            return 0.0
        try:
            bal = self._client.get_balance()
            return float(bal) if bal else 0.0
        except Exception as e:
            log.error(f"get_balance error: {e}")
            return 0.0
