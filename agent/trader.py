"""Main trading orchestrator — runs every 15 minutes."""
import logging
import time
from datetime import datetime
from typing import List, Dict

from agent.polymarket_client import PolymarketClient
from agent.market_data import fetch_candles, store_candles, get_current_btc_price
from agent.market_tracker import MarketTracker
from agent.websocket_feed import FeedManager, LIVE
from agent.strategies.ensemble import EnsembleStrategy
from agent.ml.self_learner import SelfLearner
from agent.risk_manager import RiskManager
from agent.database.models import get_session, Trade, MarketSnapshot, AgentLog
import config

log = logging.getLogger(__name__)


def _market_token_ids(market: Dict) -> List[str]:
    """Extract YES/NO token ids from a market dict."""
    tokens = market.get("tokens", market.get("clobTokenIds", []))
    ids = []
    for t in (tokens if isinstance(tokens, list) else []):
        if isinstance(t, dict):
            ids.append(t.get("token_id", t.get("tokenId", "")))
        elif isinstance(t, str):
            ids.append(t)
    return [i for i in ids if i]


class Trader:
    def __init__(self):
        self.poly     = PolymarketClient()
        self.tracker  = MarketTracker()
        self.feeds    = FeedManager()
        self.ensemble = EnsembleStrategy()
        self.learner  = SelfLearner()
        self.risk     = RiskManager()
        self.cycle    = 0
        # slug currently being traded, so we only act once per new slug
        self._traded_slug: str = None
        # start live WebSocket feeds (Binance price + Polymarket books)
        self.feeds.start(on_kline_close=self._on_kline_close)

    def _on_kline_close(self, candle: dict):
        """Called by the Binance WS thread whenever a 15m candle closes."""
        log.info(f"[ws] 15m candle closed @ {candle['close']:.0f} — re-evaluating slug.")
        try:
            self.roll_check()
        except Exception as e:
            log.warning(f"on_kline_close roll error: {e}")

    # ── Main loop step ───────────────────────────────────────────────────────

    def run_cycle(self):
        self.cycle += 1
        log.info(f"=== Cycle #{self.cycle} @ {datetime.utcnow().isoformat()} ===")
        self._log_event("cycle_start", {"cycle": self.cycle})

        # 1. Refresh strategy weights from learning
        weights = self.learner.get_weights()
        self.ensemble.update_weights(weights)

        # 2. Get BTC candles
        candles = fetch_candles()
        if candles.empty:
            log.warning("No candle data — skipping cycle.")
            return
        store_candles(candles)

        btc_price = get_current_btc_price()
        log.info(f"BTC: ${btc_price:,.0f}  |  candles: {len(candles)}")

        # 3. Lock onto the current rolling 15-min BTC slug and trade it
        balance = self.poly.get_balance() or 1000.0  # fallback for paper trading
        self._trade_active_slug(candles, balance)

        # 4. Periodically retrain the meta-model
        if self.cycle % 10 == 0:
            log.info("Retraining meta-model...")
            self.learner.train_meta_model()
            self.learner.update_weights_from_history()

        log.info(f"Cycle #{self.cycle} complete.")

    # ── Rolling slug handling ────────────────────────────────────────────────

    def roll_check(self):
        """
        Fast hand-off check (runs every ROLL_CHECK_SECONDS). The moment the
        active 15-min slug ends, this locks onto the next one and trades it
        immediately — no waiting for the next 15-min cycle.
        """
        market = self.tracker.get_current_market()
        if not market:
            return

        slug = market.get("slug") or market.get("conditionId")
        secs = self.tracker.seconds_to_end(market)

        # New slug we haven't traded yet → jump on it right away
        if slug != self._traded_slug:
            log.info(f"[roll] New active slug detected: {slug} "
                     f"({secs:.0f}s to end)" if secs is not None else
                     f"[roll] New active slug detected: {slug}")
            candles = fetch_candles()
            if candles.empty:
                return
            balance = self.poly.get_balance() or 1000.0
            self._trade_active_slug(candles, balance, prefetched=market)

    def _trade_active_slug(self, candles, balance: float, prefetched: Dict = None):
        """Evaluate and (maybe) trade the current rolling 15-min BTC market."""
        market = prefetched or self.tracker.get_current_market()
        if not market:
            log.info("No active rolling 15-min BTC slug — nothing to trade.")
            return

        slug = market.get("slug") or market.get("conditionId")

        # Point the live WS order-book feed at this slug's tokens
        self.feeds.subscribe_tokens(_market_token_ids(market))

        # Don't open new positions in the final seconds before resolution
        if not self.tracker.is_tradeable(market):
            secs = self.tracker.seconds_to_end(market)
            log.info(f"[roll] Slug {slug} too close to end ({secs:.0f}s) — "
                     f"skipping new entries.")
            return

        self._snapshot_markets([market])
        self._evaluate_market(market, candles, balance)
        self._traded_slug = slug

    # ── Market evaluation ────────────────────────────────────────────────────

    def _evaluate_market(self, market: Dict, candles, balance: float):
        market = self.poly.enrich_market(market)
        yes_price, no_price = self.poly.get_market_prices(market)

        if not (0.02 < yes_price < 0.98):
            return   # near-resolved market, skip

        signal = self.ensemble.generate_signal(
            candles, yes_price, no_price, market
        )

        if signal.direction == "PASS":
            log.debug(f"  PASS {market.get('question','')[:60]}: {signal.details.get('reason','')}")
            return

        # Meta-model probability boost
        ml_prob = self.learner.predict_win_probability(
            signal.details, signal.direction, yes_price if signal.direction == "YES" else no_price
        )
        blended_conf = 0.7 * signal.confidence + 0.3 * ml_prob

        price = yes_price if signal.direction == "YES" else no_price
        size  = self.risk.compute_position_size(blended_conf, signal.edge, price, balance)

        can_trade, reason = self.risk.can_trade(size, balance)
        if not can_trade:
            log.info(f"  Blocked ({reason}): {market.get('question','')[:60]}")
            return

        log.info(
            f"  TRADE → {signal.direction} | {market.get('question','')[:60]}\n"
            f"    conf={blended_conf:.2%}  edge={signal.edge:.3f}  size=${size:.2f}"
        )
        self._execute_trade(market, signal, blended_conf, size, yes_price, no_price)

    # ── Trade execution ──────────────────────────────────────────────────────

    def _execute_trade(self, market: Dict, signal, confidence: float,
                       size_usd: float, yes_price: float, no_price: float):
        tokens = market.get("tokens", market.get("clobTokenIds", []))
        token_id = None
        for t in (tokens if isinstance(tokens, list) else []):
            if isinstance(t, dict):
                if t.get("outcome", "").upper() == signal.direction:
                    token_id = t.get("token_id", t.get("tokenId", ""))

        price = yes_price if signal.direction == "YES" else no_price
        order_id = self.poly.place_market_order(token_id or "", signal.direction, size_usd, price)

        session = get_session()
        try:
            trade = Trade(
                timestamp=datetime.utcnow(),
                market_id=market.get("conditionId", market.get("id", "")),
                market_slug=market.get("slug", ""),
                question=market.get("question", ""),
                side=signal.direction,
                price=price,
                size_usd=size_usd,
                shares=round(size_usd / price, 4) if price > 0 else 0,
                order_id=order_id,
                strategy_used="ensemble",
                signal_data=signal.details,
            )
            session.add(trade)
            session.commit()
            log.info(f"  Trade saved: id={trade.id}")
        except Exception as e:
            session.rollback()
            log.error(f"save trade error: {e}")
        finally:
            session.close()

    # ── Resolution checker ───────────────────────────────────────────────────

    def check_resolutions(self):
        """Check open trades and mark resolved if market has settled."""
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.resolved == False).all()
            for trade in open_trades:
                market = self._fetch_market_by_id(trade.market_id)
                if not market:
                    continue
                resolved_val = market.get("resolved")
                resolution   = market.get("resolution", "")

                if resolved_val or resolution in ("YES", "NO"):
                    trade.resolved   = True
                    trade.resolution = resolution or ("YES" if resolved_val else "NO")
                    exit_p = 1.0 if trade.resolution == trade.side else 0.0
                    trade.exit_price = exit_p
                    trade.pnl_usd    = round((exit_p - trade.price) * trade.shares, 4)
                    trade.roi_pct    = round((exit_p / trade.price - 1) * 100, 2) if trade.price else 0
                    trade.closed_at  = datetime.utcnow()
                    log.info(f"Resolved trade {trade.id}: {trade.resolution}  PnL=${trade.pnl_usd:.2f}")

            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"check_resolutions error: {e}")
        finally:
            session.close()

    def _fetch_market_by_id(self, market_id: str) -> Dict:
        import requests
        try:
            r = requests.get(f"{config.GAMMA_API}/markets/{market_id}", timeout=8)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return {}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _snapshot_markets(self, markets: List[Dict]):
        session = get_session()
        try:
            for m in markets:
                prices = m.get("outcomePrices", [0.5, 0.5])
                snap = MarketSnapshot(
                    market_id=m.get("conditionId", m.get("id", "")),
                    yes_price=float(prices[0]) if len(prices) > 0 else 0.5,
                    no_price=float(prices[1])  if len(prices) > 1 else 0.5,
                    volume_24h=float(m.get("volume24hr", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    question=m.get("question", ""),
                )
                session.add(snap)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def _log_event(self, level: str, data: dict):
        session = get_session()
        try:
            session.add(AgentLog(level=level, message=str(data), data=data))
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
