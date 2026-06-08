"""Polymarket CLOB API client with BTC market discovery + live execution.

PAPER mode (no POLYMARKET_PRIVATE_KEY): every write method is a no-op stub that
just logs, so the agent runs end-to-end without touching real money.

LIVE mode (key present): real orders via py_clob_client, real USDC balance via
the CLOB balance/allowance endpoint, automatic USDC allowance approval, and
on-chain redemption of winning positions via the Conditional Tokens Framework
(CTF) contract using web3. All live calls are wrapped so a failure logs and
returns safely instead of crashing the trading loop.
"""
import logging
import time
import re
from typing import List, Dict, Optional

import requests

import config

log = logging.getLogger(__name__)

# Polygon mainnet contract addresses (used for on-chain redemption).
DATA_API = "https://data-api.polymarket.com"

# Minimal ABI: only redeemPositions, which converts resolved outcome tokens
# back into USDC for the winning side.
_CTF_ABI = [{
    "constant": False,
    "inputs": [
        {"name": "collateralToken",    "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId",        "type": "bytes32"},
        {"name": "indexSets",          "type": "uint256[]"},
    ],
    "name": "redeemPositions",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function",
}]


class PolymarketClient:
    def __init__(self):
        self._gamma = config.GAMMA_API
        self._clob_host = config.CLOB_HOST
        self._client = None
        self._allowance_ok = False     # set once USDC approval is confirmed
        self._init_client()

    def _init_client(self):
        if not config.POLYMARKET_PRIVATE_KEY:
            log.warning("No Polymarket private key — paper mode (no real trades).")
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            # signature_type 2 = Polymarket proxy/email wallet; the USDC and the
            # outcome tokens live in the FUNDER (proxy) address, not the EOA
            # derived from the key — so it must be supplied or orders sign for
            # the wrong account. signature_type 0 = a plain EOA (funder unused).
            sig_type = int(getattr(config, "POLYMARKET_SIGNATURE_TYPE", 2))
            funder   = getattr(config, "POLYMARKET_FUNDER", "") or None

            kwargs = dict(host=self._clob_host, chain_id=137,
                          key=config.POLYMARKET_PRIVATE_KEY, signature_type=sig_type)
            if funder:
                kwargs["funder"] = funder

            # Use supplied API creds if present, else derive them from the key.
            if config.POLYMARKET_API_KEY and config.POLYMARKET_SECRET:
                kwargs["creds"] = ApiCreds(
                    api_key=config.POLYMARKET_API_KEY,
                    api_secret=config.POLYMARKET_SECRET,
                    api_passphrase=config.POLYMARKET_PASSPHRASE,
                )
                self._client = ClobClient(**kwargs)
            else:
                self._client = ClobClient(**kwargs)
                creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(creds)
                log.info("Derived Polymarket API credentials from private key.")

            log.info(f"Polymarket CLOB client initialized "
                     f"(sig_type={sig_type}, funder={'set' if funder else 'EOA'}).")
            # Make sure USDC is approved for trading before the first order.
            if getattr(config, "AUTO_APPROVE_USDC", True):
                self.ensure_allowance()
        except ImportError:
            log.warning("py_clob_client not installed — paper mode only. "
                        "Run: pip3 install py-clob-client web3")
        except Exception as e:
            log.error(f"CLOB client init error: {e}")
            self._client = None

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

    def _book_levels(self, book, side: str):
        """Return sorted (price, size) levels from a book, dict- or object-shaped."""
        raw = None
        if isinstance(book, dict):
            raw = book.get(side, [])
        else:
            raw = getattr(book, side, None)
        levels = []
        for lvl in (raw or []):
            try:
                if isinstance(lvl, dict):
                    levels.append((float(lvl["price"]), float(lvl.get("size", 0))))
                else:
                    levels.append((float(lvl.price), float(getattr(lvl, "size", 0))))
            except Exception:
                continue
        levels.sort(key=lambda x: x[0], reverse=(side == "bids"))
        return levels

    def get_mid_price(self, token_id: str) -> Optional[float]:
        # prefer live WebSocket price
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
        bids = self._book_levels(book, "bids")
        asks = self._book_levels(book, "asks")
        if bids and asks:
            return (bids[0][0] + asks[0][0]) / 2
        return None

    def get_buy_price(self, token_id: str, mid: float) -> float:
        """Realistic BUY fill: cross to the best ask (live book), else mid+spread/2."""
        try:
            book = self.get_book(token_id) if token_id else None
            if book:
                asks = self._book_levels(book, "asks")
                if asks:
                    return min(0.999, asks[0][0])
        except Exception:
            pass
        return min(0.999, max(0.001, mid + config.PAPER_SPREAD / 2.0))

    def get_sell_price(self, token_id: str, mid: float) -> float:
        """Realistic SELL fill: cross to the best bid (live book), else mid-spread/2."""
        try:
            book = self.get_book(token_id) if token_id else None
            if book:
                bids = self._book_levels(book, "bids")
                if bids:
                    return max(0.001, bids[0][0])
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
        """
        OPEN a position: BUY `size_usd` worth of the given outcome token.

        On Polymarket you take a position by BUYING the YES *or* the NO token —
        the order side is therefore ALWAYS 'BUY'; `token_id` already encodes
        which outcome. (The old code wrongly passed the YES/NO outcome as the
        order side, which the exchange rejects.) Returns the order id, or None.
        """
        if not self._client:
            log.info(f"[paper] BUY {side}-token ${size_usd:.2f} @ {price:.3f} "
                     f"(token={token_id[:12]}...)")
            return f"PAPER_{int(time.time())}"
        return self._market_order(token_id, "BUY", size_usd, price)

    def sell_position(self, token_id: str, shares: float,
                      price: float) -> Optional[str]:
        """CLOSE (early-exit) a position: SELL `shares` of the held outcome token."""
        if not self._client:
            log.info(f"[paper] SELL {shares:.4f} sh @ {price:.3f} "
                     f"(token={token_id[:12]}...)")
            return f"PAPER_SELL_{int(time.time())}"
        if shares <= 0:
            return None
        return self._market_order(token_id, "SELL", shares, price)

    def _market_order(self, token_id: str, action: str,
                      amount: float, price: float) -> Optional[str]:
        """
        Place a marketable Fill-or-Kill order.

        amount semantics (py_clob_client MarketOrderArgs):
          • BUY  → `amount` is USDC to spend
          • SELL → `amount` is the number of shares to sell
        Falls back to a crossing LIMIT order if the market-order API differs
        across py_clob_client versions.
        """
        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            self.ensure_allowance()
            side_const = BUY if action == "BUY" else SELL
            args = MarketOrderArgs(token_id=token_id, amount=round(amount, 4),
                                   side=side_const, price=round(price, 4))
            signed = self._client.create_market_order(args)
            resp = self._client.post_order(signed, OrderType.FOK)
            oid = (resp or {}).get("orderID", "") if isinstance(resp, dict) else ""
            log.info(f"LIVE {action} order filled: amount={amount:.4f} "
                     f"@~{price:.3f} → {oid or resp}")
            return oid or f"LIVE_{int(time.time())}"
        except Exception as e:
            log.warning(f"market order ({action}) failed ({e}); trying limit fallback.")
        # ── Fallback: crossing limit order (size always in SHARES) ────────────
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            shares = amount / price if action == "BUY" else amount
            args = OrderArgs(token_id=token_id, price=round(price, 4),
                             size=round(shares, 4),
                             side=(BUY if action == "BUY" else SELL))
            signed = self._client.create_order(args)
            resp = self._client.post_order(signed, OrderType.FAK)  # fill-and-kill
            oid = (resp or {}).get("orderID", "") if isinstance(resp, dict) else ""
            log.info(f"LIVE {action} limit-fallback posted: {oid or resp}")
            return oid or f"LIVE_{int(time.time())}"
        except Exception as e:
            log.error(f"place_order ({action}) error: {e}")
            return None

    # ── Balance / allowance ──────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Real USDC collateral balance (in dollars), or 0.0 in paper mode."""
        if not self._client:
            return 0.0
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = self._client.get_balance_allowance(params)
            raw = resp.get("balance") if isinstance(resp, dict) else getattr(resp, "balance", None)
            # USDC has 6 decimals; the endpoint returns base units as a string.
            return round(float(raw) / 1_000_000, 2) if raw is not None else 0.0
        except Exception as e:
            log.error(f"get_balance error: {e}")
            return 0.0

    def ensure_allowance(self) -> bool:
        """
        Make sure the CLOB exchange is approved to move our USDC. Without this the
        very first BUY reverts. Runs once (cached); safe to call repeatedly.
        """
        if not self._client or self._allowance_ok:
            return self._allowance_ok
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            cur = self._client.get_balance_allowance(params)
            allowance = cur.get("allowance") if isinstance(cur, dict) else getattr(cur, "allowance", 0)
            if allowance and float(allowance) > 0:
                self._allowance_ok = True
                return True
            log.info("Approving USDC allowance for the CLOB exchange (one-time)...")
            self._client.update_balance_allowance(params)
            self._allowance_ok = True
            log.info("USDC allowance approved.")
            return True
        except Exception as e:
            log.error(f"ensure_allowance error: {e}")
            return False

    def get_portfolio(self) -> List[Dict]:
        return self.get_positions()

    def get_positions(self) -> List[Dict]:
        """Open positions from the Polymarket Data API (keyed on the funder addr)."""
        if not self._client:
            return []
        try:
            addr = getattr(config, "POLYMARKET_FUNDER", "") or self._client.get_address()
            resp = requests.get(f"{DATA_API}/positions",
                                params={"user": addr}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("positions", [])
        except Exception as e:
            log.error(f"get_positions error: {e}")
            return []

    # ── Redemption of winnings (on-chain) ────────────────────────────────────

    def redeem_position(self, condition_id: str) -> Optional[str]:
        """
        Convert a RESOLVED market's winning outcome tokens back into USDC by
        calling redeemPositions on the CTF contract. Returns the tx hash.

        Note on proxy wallets: with signature_type 1/2 the tokens are held by the
        Polymarket proxy (funder) address, and Polymarket typically AUTO-REDEEMS
        resolved winnings to your balance — so manual redemption is usually
        unnecessary there and a direct EOA call would redeem nothing. This path
        is implemented for EOA-held tokens (signature_type 0) and as an explicit
        fallback; it no-ops cleanly when web3/key/RPC aren't available.
        """
        if not self._client or not getattr(config, "ENABLE_REDEEM", True):
            return None
        try:
            from web3 import Web3
        except ImportError:
            log.warning("web3 not installed — cannot redeem. Run: pip3 install web3")
            return None
        try:
            w3 = Web3(Web3.HTTPProvider(config.POLYGON_RPC))
            acct = w3.eth.account.from_key(config.POLYMARKET_PRIVATE_KEY)
            ctf = w3.eth.contract(
                address=Web3.to_checksum_address(config.CTF_ADDRESS), abi=_CTF_ABI)
            usdc = Web3.to_checksum_address(config.USDC_ADDRESS)
            parent = b"\x00" * 32
            cond = condition_id if condition_id.startswith("0x") else "0x" + condition_id
            cond_bytes = bytes.fromhex(cond[2:])
            fn = ctf.functions.redeemPositions(usdc, parent, cond_bytes, [1, 2])
            tx = fn.build_transaction({
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
                "gas": 200_000,
                "maxFeePerGas": w3.to_wei("100", "gwei"),
                "maxPriorityFeePerGas": w3.to_wei("30", "gwei"),
                "chainId": 137,
            })
            signed = acct.sign_transaction(tx)
            txh = w3.eth.send_raw_transaction(signed.raw_transaction)
            h = txh.hex()
            log.info(f"Redeem submitted for condition {cond[:14]}… → tx {h}")
            return h
        except Exception as e:
            log.error(f"redeem_position error: {e}")
            return None
