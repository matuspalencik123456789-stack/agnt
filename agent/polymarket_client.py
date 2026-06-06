"""Polymarket CLOB API client with BTC market discovery."""
import logging
import time
import re
from typing import List, Dict, Optional

import requests

import config

log = logging.getLogger(__name__)


class PolymarketClient:
    def __init__(self):
        self._gamma = config.GAMMA_API
        self._clob_host = config.CLOB_HOST
        self._client = None
        self._init_client()

    def _init_client(self):
        if not config.POLYMARKET_PRIVATE_KEY:
            log.warning("No Polymarket private key — paper mode (no real trades).")
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            creds = ApiCreds(
                api_key=config.POLYMARKET_API_KEY,
                api_secret=config.POLYMARKET_SECRET,
                api_passphrase=config.POLYMARKET_PASSPHRASE,
            )
            self._client = ClobClient(
                host=self._clob_host,
                chain_id=137,
                key=config.POLYMARKET_PRIVATE_KEY,
                creds=creds,
                signature_type=2,
            )
            log.info("Polymarket CLOB client initialized.")
        except ImportError:
            log.warning("py_clob_client not installed — paper mode only.")
        except Exception as e:
            log.error(f"CLOB client init error: {e}")

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_btc_markets(self, limit: int = 20) -> List[Dict]:
        markets = []
        for kw in config.BTC_MARKET_KEYWORDS:
            try:
                resp = requests.get(
                    f"{self._gamma}/markets",
                    params={"q": kw, "active": "true", "closed": "false", "limit": limit},
                    timeout=10,
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

        seen, unique = set(), []
        for m in markets:
            cid = m.get("conditionId", m.get("id", ""))
            if cid not in seen:
                seen.add(cid)
                unique.append(m)
        return unique

    def enrich_market(self, market: Dict) -> Dict:
        q = market.get("question", "")
        prices = re.findall(r"\$?([\d,]+(?:\.\d+)?)", q.replace(",", ""))
        market["strike_price"] = float(prices[0].replace(",", "")) if prices else 0.0
        return market

    # ── Order book ───────────────────────────────────────────────────────────

    def get_book(self, token_id: str) -> Optional[Dict]:
        try:
            if self._client:
                return self._client.get_order_book(token_id)
            resp = requests.get(
                f"{self._clob_host}/book",
                params={"token_id": token_id}, timeout=8,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"get_book error: {e}")
            return None

    def get_mid_price(self, token_id: str) -> Optional[float]:
        # prefer live WebSocket price
        try:
            from agent.websocket_feed import LIVE
            live_mid = LIVE.get_mid(token_id)
            if live_mid:
                return live_mid
        except Exception:
            pass
        # fallback to REST order book
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

    def get_buy_price(self, token_id: str, mid: float) -> float:
        """
        Realistic fill price for BUYING this token: cross the spread to the ask.
        Uses the live order book's best ask when available; otherwise models an
        assumed spread around the mid. Clamped to (0, 1).
        """
        # try the real ask from the order book
        try:
            book = self.get_book(token_id) if token_id else None
            if book:
                asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
                if asks:
                    return min(0.999, float(asks[0]["price"]))
        except Exception:
            pass
        # fallback: mid + half the assumed spread
        return min(0.999, max(0.001, mid + config.PAPER_SPREAD / 2.0))

    def get_sell_price(self, token_id: str, mid: float) -> float:
        """
        Realistic fill price for SELLING this token: cross the spread to the bid.
        Uses the live order book's best bid when available; otherwise models an
        assumed spread around the mid. Clamped to (0, 1).
        """
        try:
            book = self.get_book(token_id) if token_id else None
            if book:
                bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
                if bids:
                    return max(0.001, float(bids[0]["price"]))
        except Exception:
            pass
        return min(0.999, max(0.001, mid - config.PAPER_SPREAD / 2.0))

    def get_market_prices(self, market: Dict) -> tuple:
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

        if yes_price is None:
            yes_price = float(market.get("outcomePrices", [0.5, 0.5])[0])
        if no_price is None:
            no_price  = float(market.get("outcomePrices", [0.5, 0.5])[1])
        return yes_price, no_price

    # ── Order placement ──────────────────────────────────────────────────────

    def place_market_order(self, token_id: str, side: str,
                           size_usd: float, price: float) -> Optional[str]:
        if not self._client:
            log.info(f"[paper] {side} ${size_usd:.2f} @ {price:.3f} (token={token_id[:12]}...)")
            return f"PAPER_{int(time.time())}"
        try:
            from py_clob_client.clob_types import OrderArgs
            size = round(size_usd / price, 4)
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=size,
                side=side.lower(),
            )
            resp = self._client.create_and_post_order(order_args)
            order_id = resp.get("orderID", "")
            log.info(f"Order placed: {side} {size} @ {price} -> {order_id}")
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
