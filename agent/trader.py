"""Main trading orchestrator — runs every 15 minutes."""
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Optional

from sqlalchemy import func

from agent.polymarket_client import PolymarketClient
from agent.market_data import (
    fetch_candles, store_candles, get_current_btc_price, get_price_at,
)
from agent.strategies.outcome_model import window_open_price, analyze_trend
from agent.market_tracker import MarketTracker
from agent.websocket_feed import FeedManager, LIVE
from agent.strategies.ensemble import EnsembleStrategy
from agent.ml.self_learner import SelfLearner
from agent.risk_manager import RiskManager
from agent.database.models import get_session, Trade, MarketSnapshot, AgentLog
import config

log = logging.getLogger(__name__)


def _token_for_side(market: Dict, side: str) -> str:
    """Return the YES/NO token id for the given side.

    Handles both market shapes:
      • `tokens`: a list of dicts each carrying an `outcome` + `token_id`.
      • `clobTokenIds`: a bare list/JSON-string of ids in outcome order
        (index 0 = Yes, index 1 = No — the Gamma convention).
    """
    side = (side or "").upper()
    # 1) dict-shaped tokens with explicit outcomes
    tokens = market.get("tokens")
    if isinstance(tokens, list):
        for t in tokens:
            if isinstance(t, dict) and t.get("outcome", "").upper() == side:
                return t.get("token_id", t.get("tokenId", "")) or ""
    # 2) bare clobTokenIds in [Yes, No] order
    ids = market.get("clobTokenIds", [])
    if isinstance(ids, str):
        try:
            import json
            ids = json.loads(ids)
        except Exception:
            ids = []
    if isinstance(ids, list) and len(ids) >= 2:
        idx = 0 if side == "YES" else 1
        tok = ids[idx]
        if isinstance(tok, dict):
            return tok.get("token_id", tok.get("tokenId", "")) or ""
        return str(tok) if tok else ""
    return ""


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
        # track the active slug + how many trades we've opened on it (cap per slug)
        self._current_slug: str = None
        self._slug_trades: int  = 0
        self._last_entry_ts: float = 0.0   # for the per-slug entry cooldown
        # Track model direction across consecutive roll-checks to filter flip-flops.
        self._signal_history: deque = deque(maxlen=5)
        # Block new entries until this timestamp after a LOSING early exit.
        self._post_exit_until: float = 0.0
        # pending early-exit confirmations: {trade_id: {"reason": str, "count": int}}
        self._exit_pending: Dict[int, dict] = {}
        # One lock serialises every critical section. roll_check fires from BOTH
        # the APScheduler threads (5s) and the Binance WS thread (on candle close),
        # all mutating shared state (_slug_trades, _exit_pending, _last_entry_ts)
        # and writing the same DB rows — without this they race into double entries
        # and corrupted counters.
        self._lock = threading.RLock()
        # adaptive market-anchor weight for the outcome model (calibration-driven)
        try:
            anchor = self.learner.get_market_anchor()
            self.ensemble.strategies["stat_outcome"].market_weight = anchor
        except Exception:
            pass
        # startup time — the agent observes/analyses candles during the warm-up
        # before it's allowed to place its first trade (no blind immediate entry)
        self._started_at: float = time.time()
        # start live WebSocket feeds (Binance price + Polymarket books)
        self.feeds.start(on_kline_close=self._on_kline_close)

    def _on_kline_close(self, candle: dict):
        """Called by the Binance WS thread whenever a kline closes."""
        iv = getattr(config, "WS_KLINE_INTERVAL", "1m")
        log.info(f"[ws] {iv} candle closed @ {candle['close']:.0f} — re-evaluating slug.")
        try:
            self.roll_check()
        except Exception as e:
            log.warning(f"on_kline_close roll error: {e}")

    # ── Main loop step ───────────────────────────────────────────────────────

    def run_cycle(self):
      with self._lock:
        self.cycle += 1
        log.info(f"=== Cycle #{self.cycle} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
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
        balance = self._available_balance()
        self._trade_active_slug(candles, balance)

        # 4. Periodically retrain the meta-model + recalibrate the market anchor
        if self.cycle % 5 == 0:
            log.info("Retraining meta-model...")
            self.learner.train_meta_model()
            self.learner.update_weights_from_history()
            try:
                anchor = self.learner.update_market_anchor()
                self.ensemble.strategies["stat_outcome"].market_weight = anchor
                n, bm, bk = self.learner.calibration_stats()
                if n:
                    log.info(f"Calibration: n={n} Brier model={bm:.4f} "
                             f"market={bk:.4f} → market-anchor={anchor:.2f}")
            except Exception as e:
                log.debug(f"calibration update: {e}")

        log.info(f"Cycle #{self.cycle} complete.")

    # ── Balance ──────────────────────────────────────────────────────────────

    def _available_balance(self) -> float:
        """Live balance from the broker, or a virtual paper bankroll that actually
        moves: PAPER_START_BALANCE + realised P&L of all resolved trades."""
        if self.poly._client:
            return self.poly.get_balance() or 0.0
        session = get_session()
        try:
            realized = session.query(
                func.coalesce(func.sum(Trade.pnl_usd), 0.0)
            ).filter(Trade.resolved == True).scalar() or 0.0
        finally:
            session.close()
        return round(config.PAPER_START_BALANCE + float(realized), 2)

    # ── Rolling slug handling ────────────────────────────────────────────────

    def roll_check(self):
      with self._lock:
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

        # New slug → announce and reset the per-slug trade counter
        if slug != self._current_slug:
            log.info(f"[roll] New active slug: {slug}" +
                     (f" ({secs:.0f}s to end)" if secs is not None else ""))

        # Re-evaluate every roll-check; the per-slug cap limits how many entries
        candles = fetch_candles()
        if candles.empty:
            return
        # Manage existing positions first — maybe close one early.
        self.manage_open_positions(market, candles)
        balance = self._available_balance()
        self._trade_active_slug(candles, balance, prefetched=market)

    def _trade_active_slug(self, candles, balance: float, prefetched: Dict = None):
        """Evaluate and (maybe) trade the current rolling 15-min BTC market."""
        market = prefetched or self.tracker.get_current_market()
        if not market:
            log.info("No active rolling 15-min BTC slug — nothing to trade.")
            return

        slug = market.get("slug") or market.get("conditionId")

        # Reset the per-slug counter whenever we roll onto a new slug
        if slug != self._current_slug:
            self._current_slug = slug
            self._slug_trades  = 0
            self._last_entry_ts = 0.0
            self._signal_history.clear()
            self._post_exit_until = 0.0   # new slug = fresh start, no whipsaw memory

        # Point the live WS order-book feed at this slug's tokens
        self.feeds.subscribe_tokens(_market_token_ids(market))

        # Warm-up: observe & analyse the candle history before the first trade.
        elapsed = time.time() - self._started_at
        if elapsed < config.WARMUP_SECONDS:
            trend = analyze_trend(candles["close"], config.TREND_LOOKBACK,
                                  config.TREND_MIN_STRENGTH)
            log.info(f"[warm-up] analysing candles "
                     f"({elapsed:.0f}/{config.WARMUP_SECONDS}s) — "
                     f"trend={trend['direction']} strength={trend['strength']:.5f}; "
                     f"not trading yet.")
            return

        # Enforce the per-slug trade cap
        if self._slug_trades >= config.MAX_TRADES_PER_SLUG:
            return

        # Don't open new positions in the final seconds before resolution
        if not self.tracker.is_tradeable(market):
            secs = self.tracker.seconds_to_end(market)
            log.info(f"[roll] Slug {slug} too close to end ({secs:.0f}s) — "
                     f"skipping new entries.")
            return

        self._snapshot_markets([market])
        traded = self._evaluate_market(market, candles, balance)
        if traded:
            self._slug_trades += 1
            log.info(f"  [slug {slug}] trade {self._slug_trades}/{config.MAX_TRADES_PER_SLUG}")

    # ── Selectivity ("should I trade now, or wait for a better setup?") ──────

    def _is_worth_trading(self, confidence: float, edge: float) -> tuple[bool, str]:
        """
        Decide whether this setup justifies spending one of the limited per-slug
        trade slots *right now*. The bar rises with each entry already taken and
        a cooldown prevents firing several trades back-to-back.
        """
        quality = confidence * max(edge, 0.0)

        # cooldown since the last entry on this slug
        cd = config.ENTRY_COOLDOWN_SECONDS
        since = time.time() - self._last_entry_ts
        if self._last_entry_ts and since < cd:
            return False, f"cooldown {since:.0f}s/{cd}s, quality={quality:.4f}"

        # escalating quality bar: 1st entry cheapest, each extra one stricter
        bar = config.ENTRY_QUALITY_MIN * (config.ENTRY_QUALITY_ESCALATION ** self._slug_trades)
        if quality < bar:
            return False, f"quality {quality:.4f} < bar {bar:.4f} (entry #{self._slug_trades+1})"

        return True, f"quality {quality:.4f} ≥ bar {bar:.4f}"

    # ── Market evaluation ────────────────────────────────────────────────────

    def _evaluate_market(self, market: Dict, candles, balance: float) -> bool:
        """Returns True if a trade was executed."""
        market = self.poly.enrich_market(market)
        yes_price, no_price = self.poly.get_market_prices(market)

        if not (0.02 < yes_price < 0.98):
            return False   # near-resolved market, skip

        signal = self.ensemble.generate_signal(
            candles, yes_price, no_price, market
        )

        log.info(f"  Eval: {market.get('question','')[:55]} | YES={yes_price:.3f} NO={no_price:.3f}")

        if signal.direction == "PASS":
            log.info(f"  → PASS: {signal.details.get('reason', 'no edge / no consensus')}")
            return False

        # ── Post-exit cooldown gate ───────────────────────────────────────────
        # After a LOSING early exit (model flip / stop-loss), block new entries
        # for POST_EXIT_COOLDOWN_SECONDS to prevent immediately re-entering in
        # the opposite direction while the model is still oscillating.
        now = time.time()
        if now < self._post_exit_until:
            remaining = int(self._post_exit_until - now)
            log.info(f"  → HOLD: post-exit cooldown ({remaining}s left) — "
                     f"preventing whipsaw re-entry after loss")
            return False

        # ── Signal stability gate ─────────────────────────────────────────────
        # Require SIGNAL_STABILITY_COUNT consecutive same-direction signals before
        # entering. P(up) can swing wildly (0.08 → 0.82 → 0.08) within one slug;
        # this filters those flip-flops so we only trade a stable model conviction.
        n_stable = config.SIGNAL_STABILITY_COUNT
        if n_stable > 1:
            self._signal_history.append(signal.direction)
            recent = list(self._signal_history)
            if len(recent) < n_stable or not all(d == signal.direction for d in recent[-n_stable:]):
                log.info(f"  → HOLD: signal {signal.direction} not stable yet "
                         f"(history={recent[-n_stable:]}, need {n_stable}× same)")
                return False

        # Determine the recent price development and block clearly counter-trend
        # bets (YES = expecting UP, NO = expecting DOWN on these up/down markets).
        # Only block when the trend is GENUINELY STRONG — a micro-slope (e.g.
        # 0.00009 on a flat tape) is noise and must not veto a strong signal.
        trend = analyze_trend(candles["close"], config.TREND_LOOKBACK,
                              config.TREND_MIN_STRENGTH)
        log.info(f"  Trend: {trend['direction']} (slope={trend['slope']:.5f} "
                 f"strength={trend['strength']:.5f})")
        block_strength = config.TREND_BLOCK_MIN_STRENGTH
        if (config.REQUIRE_TREND_AGREEMENT and trend["direction"] != "FLAT"
                and trend["strength"] >= block_strength):
            expect_up = (signal.direction == "YES")
            trend_up  = (trend["direction"] == "UP")
            if expect_up != trend_up:
                log.info(f"  → PASS: {signal.direction} is counter-trend "
                         f"(price developing {trend['direction']}, "
                         f"strength {trend['strength']:.5f} ≥ {block_strength})")
                return False

        # Meta-model probability boost
        ml_prob = self.learner.predict_win_probability(
            signal.details, signal.direction, yes_price if signal.direction == "YES" else no_price
        )
        blended_conf = 0.7 * signal.confidence + 0.3 * ml_prob

        # Realistic entry: we BUY at the ask, not the mid. Recompute edge against
        # the price we'd actually pay — a model that only beats the mid but not
        # the spread is not a real edge.
        mid_price = yes_price if signal.direction == "YES" else no_price
        token_id  = _token_for_side(market, signal.direction)
        price     = self.poly.get_buy_price(token_id, mid_price)

        # Dead-zone: check against BOTH mid and fill. The mid is the truer signal
        # of "this is a coin-flip market" — the ask is always slightly above mid
        # so checking only the ask lets near-0.50 mids slip through.
        dz = config.PRICE_DEADZONE_HALF
        if dz > 0 and (abs(mid_price - 0.5) < dz or abs(price - 0.5) < dz):
            log.info(f"  → PASS: mid {mid_price:.3f} / fill {price:.3f} in dead-zone "
                     f"(0.50 ± {dz}) — max fee, near coin-flip")
            return False

        # ── Don't buy straight into a forced exit ─────────────────────────────
        # If the current best BID is already at/below the stop-loss, opening here
        # means an instant early-exit at a loss: we pay the ask, but the bid is
        # already under water. This is the exact loop that burned trades 226-230 —
        # buy YES @ 0.095 while the bid is 0.075 ≤ 0.25 stop-loss, get stopped out
        # 30s later, re-enter, repeat until the circuit breaker trips.
        if config.ENABLE_EARLY_EXIT and config.STOP_LOSS_PRICE > 0:
            bid_now = self.poly.get_sell_price(token_id, mid_price)
            if bid_now <= config.STOP_LOSS_PRICE:
                log.info(f"  → PASS: bid {bid_now:.3f} already ≤ stop-loss "
                         f"{config.STOP_LOSS_PRICE} — entry would force an immediate "
                         f"exit at a loss")
                return False

        # ── "Don't fade a decided market" gate ────────────────────────────────
        # edge = (vote-confidence − price) REWARDS buying cheap tokens: the more
        # certainly the book has priced our side as a loser (e.g. YES at 0.085),
        # the bigger the *illusory* edge a contrarian RSI/momentum vote makes. The
        # market price is the crowd's calibrated probability — fading it hard is
        # only justified when the principled outcome model AGREES with our side.
        floor = config.FADE_MARKET_PRICE_FLOOR
        if floor > 0 and mid_price < floor:
            fair_yes = self._model_prob_for_side(candles, yes_price, no_price, market)
            if fair_yes is None:
                model_side = None
            else:
                model_side = fair_yes if signal.direction == "YES" else 1.0 - fair_yes
            if model_side is None or model_side < config.FADE_MARKET_MODEL_MINPROB:
                detail = (f"model p={model_side:.2f} < {config.FADE_MARKET_MODEL_MINPROB}"
                          if model_side is not None else "model unavailable")
                log.info(f"  → PASS: fading a decided market — our side priced "
                         f"{mid_price:.3f} (< {floor}) and {detail}")
                return False

        # Fee-aware edge: subtract the expected taker fees from the edge.
        # If early exit is enabled we may pay fee TWICE (entry + sell), so we
        # use the conservative roundtrip cost when checking viability.
        entry_fee = config.taker_fee(1.0, price)
        if config.ENABLE_EARLY_EXIT:
            # Estimate exit fee at roughly the same price (conservative).
            expected_fees = entry_fee + config.taker_fee(1.0, price)
        else:
            expected_fees = entry_fee

        # The edge that JUSTIFIES a trade must come from the principled model
        # diverging from the market (fair_side − ask), NOT from the technical tilt
        # inflating our confidence. Otherwise momentum/RSI manufacture a fake edge
        # over a fairly-priced market. The technical refinement still feeds sizing
        # via `blended_conf`; it just can't conjure a reason to enter on its own.
        fair_yes  = signal.details.get("fair_yes")
        fair_side = (fair_yes if signal.direction == "YES" else 1.0 - fair_yes) \
                    if fair_yes is not None else blended_conf
        net_edge = fair_side - price - expected_fees
        eff_edge = max(0.0, net_edge)
        if net_edge < config.FEE_EDGE_MARGIN:
            log.info(f"  → PASS: model edge after fees {net_edge:+.4f} "
                     f"< margin {config.FEE_EDGE_MARGIN} "
                     f"(fair={fair_side:.3f} fill={price:.3f} "
                     f"fees={expected_fees:.4f} early_exit={config.ENABLE_EARLY_EXIT})")
            return False

        # ── "Is it worth trading right now?" — selectivity gate ───────────────
        # The agent doesn't blow all its slots on the first qualifying signals.
        # It scores the setup (after costs) and only spends a slot on a genuinely
        # good one, escalating the bar per entry and respecting a cooldown.
        worth, why = self._is_worth_trading(blended_conf, eff_edge)
        if not worth:
            log.info(f"  → HOLD: {why} (mid={mid_price:.3f} ask={price:.3f}, "
                     f"keeping capacity in reserve)")
            return False

        size  = self.risk.compute_position_size(blended_conf, eff_edge, price, balance)

        can_trade, reason = self.risk.can_trade(size, balance)
        if not can_trade:
            log.info(f"  Blocked ({reason}): {market.get('question','')[:60]}")
            return False

        log.info(
            f"  TRADE → {signal.direction} | {market.get('question','')[:60]}\n"
            f"    conf={blended_conf:.2%}  edge={eff_edge:.3f}  "
            f"fill={price:.3f} (mid {mid_price:.3f})  size=${size:.2f}"
        )
        self._execute_trade(market, signal, blended_conf, size, price, candles, token_id)
        self._last_entry_ts = time.time()
        return True

    # ── Trade execution ──────────────────────────────────────────────────────

    def _execute_trade(self, market: Dict, signal, confidence: float,
                       size_usd: float, price: float,
                       candles=None, token_id: str = None):
        if token_id is None:
            token_id = _token_for_side(market, signal.direction)
        order_id = self.poly.place_market_order(token_id or "", signal.direction, size_usd, price)

        # window metadata for local paper-mode resolution
        w_start = self._parse_dt(market.get("startDate"))
        w_end   = self._parse_dt(market.get("endDate"))
        # Use the actual window-start BTC price (not current price) for correct resolution
        btc_open = (window_open_price(candles, w_start) if candles is not None else None
                    ) or get_current_btc_price()

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
                window_start=w_start.replace(tzinfo=None) if w_start else None,
                window_end=w_end.replace(tzinfo=None) if w_end else None,
                btc_open=btc_open or 0.0,
            )
            session.add(trade)
            session.commit()
            log.info(f"  Trade saved: id={trade.id}")
        except Exception as e:
            session.rollback()
            log.error(f"save trade error: {e}")
        finally:
            session.close()

    # ── Early exit / position management ─────────────────────────────────────

    def manage_open_positions(self, market: Dict, candles):
        """
        Look at still-open positions on the CURRENT slug and decide whether to
        sell early — locking a profit, cutting a loss, or bailing when the model
        flips against us. Runs every roll-check, so it reacts within seconds.
        """
        if not config.ENABLE_EARLY_EXIT or not market:
            return

        slug = market.get("slug") or market.get("conditionId")
        # Don't bother in the last seconds — the window resolves on its own.
        secs = self.tracker.seconds_to_end(market)
        if secs is not None and secs < config.ROLL_CUTOFF_SECONDS:
            return

        session = get_session()
        try:
            open_trades = session.query(Trade).filter(
                Trade.resolved == False,
                Trade.market_slug == (market.get("slug") or ""),
            ).all()
            if not open_trades:
                return

            market = self.poly.enrich_market(market)
            yes_mid, no_mid = self.poly.get_market_prices(market)

            # Re-evaluate the model once for this slug (to detect a reversal).
            signal = self.ensemble.generate_signal(candles, yes_mid, no_mid, market)
            # the model's probability that the held side wins (for verification)
            model_p_side = self._model_prob_for_side(candles, yes_mid, no_mid, market)

            live_ids = {t.id for t in open_trades}
            # drop stale pending entries for trades that already closed/rolled off
            self._exit_pending = {k: v for k, v in self._exit_pending.items() if k in live_ids}

            for trade in open_trades:
                side    = trade.side
                mid     = yes_mid if side == "YES" else no_mid
                token   = _token_for_side(market, side)
                sell_px = self.poly.get_sell_price(token, mid)
                p_side  = model_p_side if side == "YES" else (
                          (1.0 - model_p_side) if model_p_side is not None else None)

                # ── candidate exit reason (not yet confirmed) ──────────────
                reason = None
                if sell_px >= config.TAKE_PROFIT_PRICE:
                    # Make sure the take-profit is still net-positive AFTER the
                    # taker fee on both the original buy and this sell — never
                    # sell into a "profit" the fees would eat.
                    shares    = trade.shares or 0
                    net = ((sell_px - trade.price) * shares
                           - config.taker_fee(shares, trade.price or 0)
                           - config.taker_fee(shares, sell_px))
                    if net <= 0:
                        self._exit_pending.pop(trade.id, None)
                        log.info(f"  [hold {trade.id}] take-profit bid {sell_px:.3f} "
                                 f"but net after fees ${net:+.4f} ≤ 0 — holding")
                        continue
                    reason = f"take-profit (bid {sell_px:.3f} ≥ {config.TAKE_PROFIT_PRICE})"
                elif sell_px <= config.STOP_LOSS_PRICE:
                    # only cut the loss if the MODEL also no longer backs our side
                    if p_side is not None and p_side > config.STOP_LOSS_MODEL_MAXPROB:
                        self._exit_pending.pop(trade.id, None)
                        log.info(f"  [hold {trade.id}] bid {sell_px:.3f} low but model "
                                 f"still backs {side} (p={p_side:.2f}) — not selling")
                        continue
                    reason = f"stop-loss (bid {sell_px:.3f} ≤ {config.STOP_LOSS_PRICE})"
                elif (config.EXIT_ON_REVERSAL and signal.direction not in ("PASS", side)
                      and signal.confidence >= config.MODEL_REVERSAL_PROB):
                    reason = (f"model reversal → {signal.direction} "
                              f"conf={signal.confidence:.2f}")

                # ── confirmation: condition must persist across checks ─────
                if not reason:
                    self._exit_pending.pop(trade.id, None)
                    continue

                pend = self._exit_pending.get(trade.id)
                if pend and pend["reason"].split(" ")[0] == reason.split(" ")[0]:
                    pend["count"] += 1
                else:
                    pend = {"reason": reason, "count": 1}
                self._exit_pending[trade.id] = pend

                if pend["count"] < config.EXIT_CONFIRM_COUNT:
                    log.info(f"  [exit-watch {trade.id}] {reason} "
                             f"({pend['count']}/{config.EXIT_CONFIRM_COUNT} confirmations)")
                    continue

                self._exit_pending.pop(trade.id, None)
                # LIVE: actually sell the token on the book before recording the
                # exit. Paper mode skips this (self.poly._client is None).
                if self.poly._client:
                    self.poly.sell_position(token, trade.shares or 0, sell_px)
                self._close_position_early(session, trade, sell_px, reason)

            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"manage_open_positions error: {e}")
        finally:
            session.close()

    def _stat_signal(self, candles, yes_mid, no_mid, market):
        """The raw digital-option model Signal (direction YES/NO/PASS), or None."""
        try:
            strat = self.ensemble.strategies.get("stat_outcome")
            if not strat:
                return None
            return strat.generate_signal(candles, yes_mid, no_mid, market)
        except Exception:
            return None

    def _model_prob_for_side(self, candles, yes_mid, no_mid, market) -> Optional[float]:
        """The statistical model's probability that YES wins (fair_yes), or None."""
        sig = self._stat_signal(candles, yes_mid, no_mid, market)
        return sig.details.get("fair_yes") if sig is not None else None

    def _close_position_early(self, session, trade, sell_price: float, reason: str):
        """Sell a position before the window resolves; record realised P&L.

        An early sell is a TAKER on both legs, so we pay the crypto taker fee
        twice — once on the original buy, once on this sell. Both are subtracted
        so a marginal take-profit can't look like a win the fees actually ate.
        """
        shares    = trade.shares or 0
        entry_fee = config.taker_fee(shares, trade.price or 0)
        exit_fee  = config.taker_fee(shares, sell_price)
        fee = entry_fee + exit_fee
        pnl = round((sell_price - trade.price) * shares - fee, 4)
        roi = round((pnl / trade.size_usd) * 100, 2) if trade.size_usd else 0
        won = pnl > 0

        trade.resolved   = True
        # mark resolution so WIN/LOSS displays correctly for an early close
        trade.resolution = trade.side if won else ("NO" if trade.side == "YES" else "YES")
        trade.exit_price = sell_price
        trade.pnl_usd    = pnl
        trade.roi_pct    = roi
        trade.closed_at  = datetime.utcnow()

        emoji = "✅ WIN" if won else "❌ LOSS"
        log.info(f"  🚪 EARLY EXIT trade {trade.id}: SOLD {trade.side} @ {sell_price:.3f} "
                 f"({reason}) → {emoji} PnL=${pnl:+.4f} ROI={roi:+.1f}%")

        # Free the slug slot. For LOSING exits apply a re-entry cooldown to stop
        # the whipsaw pattern (exit NO at loss → buy YES immediately → model flips
        # back → exit YES at loss). For winning exits (take-profit) allow
        # immediate re-evaluation so a continuing trend isn't missed.
        self._slug_trades = max(0, self._slug_trades - 1)
        if not won:
            cooldown = getattr(config, "POST_EXIT_COOLDOWN_SECONDS", 90)
            self._post_exit_until = time.time() + cooldown
            self._signal_history.clear()   # force N fresh stability readings too
            log.info(f"  [slot freed] slug_trades={self._slug_trades} — "
                     f"loss exit: {cooldown}s cooldown + stability reset (anti-whipsaw)")
        else:
            self._last_entry_ts = 0.0
            log.info(f"  [slot freed] slug_trades={self._slug_trades} — "
                     f"take-profit exit: ready to re-evaluate immediately")

        try:
            self.learner.record_trade_result(
                trade.strategy_used or "ensemble", won, roi,
                signal_data=trade.signal_data, resolution=trade.resolution,
                is_early_exit=True,
            )
        except Exception as e:
            log.debug(f"learner.record (early exit): {e}")

    # ── Resolution checker ───────────────────────────────────────────────────

    @staticmethod
    def _parse_dt(value) -> Optional[datetime]:
        if not value:
            return None
        try:
            s = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def check_resolutions(self):
      """
      Two-pass resolver:
      1. Ask Polymarket API for official YES/NO (works for live trades).
      2. Paper-mode local resolution: once the window endDate has passed, resolve
         against the price at window end. If price > btc_open → YES won, else NO.
         This gives immediate feedback without waiting for Polymarket to settle.
      """
      with self._lock:
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.resolved == False).all()
            now_utc = datetime.now(timezone.utc)
            to_redeem: set = set()   # LIVE: condition ids of winners to claim

            for trade in open_trades:
                resolution = None

                # ── Pass 1: official Polymarket resolution ────────────────
                try:
                    market = self._fetch_market_by_id(trade.market_id)
                    resolved_val = market.get("resolved")
                    api_res = market.get("resolution", "")
                    if resolved_val or api_res in ("YES", "NO"):
                        resolution = api_res or ("YES" if resolved_val else "NO")
                except Exception:
                    pass

                # ── Pass 2: local BTC-price paper resolution ──────────────
                if not resolution and trade.window_end and trade.btc_open:
                    w_end_aware = trade.window_end.replace(tzinfo=timezone.utc)
                    grace = 30  # seconds after end to allow price to settle
                    elapsed = (now_utc - w_end_aware).total_seconds()
                    if elapsed >= grace:
                        # Resolve against the price AT window end (not 'now'). If we
                        # resolve promptly the current price ≈ end price; if late
                        # (restart/busy) the historical kline keeps it correct.
                        btc_close = 0.0
                        if elapsed > 120:
                            btc_close = get_price_at(w_end_aware)
                        if not btc_close:
                            btc_close = get_current_btc_price()
                        if btc_close and trade.btc_open:
                            # YES = price ended strictly ABOVE window open.
                            # A flat tie counts as NO ("did not go up").
                            resolution = "YES" if btc_close > trade.btc_open else "NO"
                            log.info(
                                f"  [paper resolve] trade {trade.id}: "
                                f"BTC {trade.btc_open:.0f} → {btc_close:.0f}  → {resolution}"
                            )

                if not resolution:
                    continue

                won = (resolution == trade.side)
                # In a binary market: winner gets $1/share, loser gets $0.
                # Settlement isn't a taker order, so only the entry buy paid a
                # taker fee — subtract that one fee here.
                exit_p = 1.0 if won else 0.0
                fee    = config.taker_fee(trade.shares or 0, trade.price or 0)
                pnl    = round((exit_p - trade.price) * trade.shares - fee, 4)
                roi    = round((pnl / trade.size_usd) * 100, 2) if trade.size_usd else 0

                trade.resolved   = True
                trade.resolution = resolution
                trade.exit_price = exit_p
                trade.pnl_usd    = pnl
                trade.roi_pct    = roi
                trade.closed_at  = datetime.utcnow()

                # LIVE: queue the winning market for on-chain redemption to USDC.
                if won and self.poly._client and getattr(config, "ENABLE_REDEEM", True):
                    to_redeem.add(trade.market_id)

                emoji = "✅ WIN" if won else "❌ LOSS"
                log.info(
                    f"  {emoji} trade {trade.id}: side={trade.side} res={resolution} "
                    f"PnL=${pnl:+.4f}  ROI={roi:+.1f}%"
                )

                # feed result back to self-learner immediately — pass the full
                # signal so the agent can credit/blame each contributing strategy
                try:
                    self.learner.record_trade_result(
                        trade.strategy_used or "ensemble", won, roi,
                        signal_data=trade.signal_data, resolution=resolution,
                    )
                except Exception as e:
                    log.debug(f"learner.record: {e}")

            session.commit()

            # LIVE: redeem each resolved winning market once (off the DB session).
            for cond in to_redeem:
                try:
                    self.poly.redeem_position(cond)
                except Exception as e:
                    log.debug(f"redeem {cond}: {e}")
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
