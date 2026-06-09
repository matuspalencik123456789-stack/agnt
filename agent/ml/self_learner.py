"""
Self-learning module.
Uses completed trade history to update strategy weights via a bandit-style
Bayesian update + gradient-boosted meta-model for feature-based edge prediction.
"""
from __future__ import annotations   # Python 3.9 compatibility for `X | None` hints
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from agent.database.models import get_session, StrategyWeight, Trade
import config

log = logging.getLogger(__name__)

STRATEGY_NAMES = ["rsi", "macd", "bollinger", "momentum", "vwap", "adx",
                  "stat_outcome", "price_action"]

# Bump whenever the MEANING/scale of _build_features changes (not just the count).
# v2: probability-first ensemble — yes_score/no_score are now probabilities (0–1)
# instead of unbounded vote-scores, so a model trained under v1 is invalid even
# though the feature count is unchanged. A version mismatch forces a clean retrain.
FEATURE_VERSION = 2


class SelfLearner:
    def __init__(self):
        self.model: GradientBoostingClassifier | None = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._load_model()

    # ── Weight management ────────────────────────────────────────────────────

    def get_weights(self) -> Dict[str, float]:
        session = get_session()
        try:
            seed = {"stat_outcome": 1.5, "price_action": 1.5}   # principled signals start heavier
            weights = {}
            for name in STRATEGY_NAMES:
                row = session.query(StrategyWeight).filter_by(name=name).first()
                if row:
                    weights[name] = row.weight
                else:
                    default_w = seed.get(name, 1.0)
                    weights[name] = default_w
                    session.add(StrategyWeight(name=name, weight=default_w))
            session.commit()
            return weights
        finally:
            session.close()

    def update_weights_from_history(self):
        """Recompute strategy weights based on resolved trade outcomes."""
        session = get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.resolved == True,
                Trade.pnl_usd != None
            ).order_by(Trade.closed_at.asc()).all()

            if len(trades) < config.MIN_TRADES_TO_LEARN:
                log.info(f"Not enough resolved trades ({len(trades)}) to learn yet.")
                return

            strategy_stats: Dict[str, Dict] = {
                name: {"wins": 0, "total": 0, "total_roi": 0.0}
                for name in STRATEGY_NAMES
            }

            decay = config.STRATEGY_DECAY
            n = len(trades)

            for i, trade in enumerate(trades):
                w = decay ** (n - 1 - i)   # recent trades matter more
                resolution = trade.resolution
                sub = (trade.signal_data or {}).get("sub", {})
                # Credit each strategy by whether its directional call was right.
                for name, info in sub.items():
                    if name not in strategy_stats:
                        continue
                    direction = info.get("dir")
                    if direction not in ("YES", "NO"):
                        continue
                    correct = (direction == resolution)
                    roi = (trade.roi_pct or 0) if correct else -abs(trade.roi_pct or 0)
                    strategy_stats[name]["wins"]      += w * int(correct)
                    strategy_stats[name]["total"]     += w
                    strategy_stats[name]["total_roi"] += w * roi

            # Minimum effective trades a strategy must have before its weight is
            # updated from history. Strategies that appear too rarely are held at
            # their current (default) weight rather than getting a spuriously high
            # weight from a handful of lucky signal appearances.
            min_trades_for_weight = int(getattr(config, "MIN_TRADES_FOR_WEIGHT", 10))

            for name, stats in strategy_stats.items():
                if stats["total"] == 0:
                    continue
                win_rate = stats["wins"] / stats["total"]
                avg_roi  = stats["total_roi"] / stats["total"]
                max_w = float(getattr(config, "STRATEGY_MAX_WEIGHT", 4.0))
                new_weight = max(0.1, min(max_w, (win_rate / 0.5) * (1 + avg_roi / 200)))

                # Don't promote a strategy on too few data points — keep default.
                if stats["total"] < min_trades_for_weight:
                    log.info(f"  {name}: only {stats['total']:.0f} weighted trades "
                             f"(< {min_trades_for_weight}) — holding default weight")
                    continue

                row = session.query(StrategyWeight).filter_by(name=name).first()
                if not row:
                    row = StrategyWeight(name=name)
                    session.add(row)
                row.weight     = round(new_weight, 4)
                row.win_rate   = round(win_rate, 4)
                row.avg_roi    = round(avg_roi, 4)
                row.trade_cnt  = int(stats["total"])
                row.updated_at = datetime.utcnow()
                log.info(f"  {name}: weight={new_weight:.3f} win_rate={win_rate:.2%} avg_roi={avg_roi:.2f}%")

            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"update_weights error: {e}")
        finally:
            session.close()

    def _update_one(self, session, name: str, correct: bool, roi_pct: float):
        """EMA bandit update of a single strategy's weight."""
        row = session.query(StrategyWeight).filter_by(name=name).first()
        if not row:
            row = StrategyWeight(name=name, weight=1.0,
                                 win_rate=0.5, avg_roi=0.0, trade_cnt=0)
            session.add(row)
        n  = (row.trade_cnt or 0) + 1
        lr = config.LEARNING_RATE
        row.win_rate  = round((row.win_rate or 0.5) * (1 - lr) + int(correct) * lr, 4)
        row.avg_roi   = round((row.avg_roi  or 0.0) * (1 - lr) + roi_pct      * lr, 4)
        # weight emphasises *directional accuracy*: above 0.5 win-rate → >1, below → <1
        max_w = float(getattr(config, "STRATEGY_MAX_WEIGHT", 4.0))
        row.weight    = round(max(0.1, min(max_w,
                            (row.win_rate / 0.5) * (1 + row.avg_roi / 200))), 4)
        row.trade_cnt = n
        row.updated_at = datetime.utcnow()
        return row.win_rate, row.weight

    def record_trade_result(self, strategy: str, won: bool, roi_pct: float,
                            signal_data: dict = None, resolution: str = None,
                            is_early_exit: bool = False):
        """
        Credit-assignment learning. The agent now judges EACH contributing
        strategy by whether its own directional call matched the real outcome —
        this is the agent's "mind": it figures out which signals were actually
        right, independent of what the ensemble decided to trade.

        For an EARLY EXIT the `resolution` is only a pseudo-outcome ("did the
        short-term price move our way"), not the true window result, so we update
        the ensemble's own track record but skip per-strategy directional credit —
        otherwise we'd teach the strategies short-term noise instead of real
        outcomes.
        """
        session = get_session()
        try:
            # Credit/blame each sub-strategy by directional correctness.
            # We do NOT track "ensemble" as a strategy — it is the decision layer,
            # not a signal. Storing its win-rate as a weight caused a feedback loop
            # where the ensemble's own aggregate result was fed back into the weight
            # table, polluting the dashboard and confusing the self-learner.
            sub = (signal_data or {}).get("sub", {})
            if (not is_early_exit) and sub and resolution in ("YES", "NO"):
                for name, info in sub.items():
                    if name not in STRATEGY_NAMES:
                        continue
                    direction = info.get("dir")
                    if direction not in ("YES", "NO"):
                        continue
                    # the strategy was "correct" if it called the winning side
                    correct = (direction == resolution)
                    # reward correct calls with +roi, wrong calls with -roi magnitude
                    signed_roi = roi_pct if correct else -abs(roi_pct)
                    wr, w = self._update_one(session, name, correct, signed_roi)
                    log.info(f"  [learn] {name}: called {direction} | "
                             f"actual {resolution} | {'✓' if correct else '✗'} "
                             f"→ win_rate={wr:.2%} weight={w:.3f}")

            session.commit()
        except Exception as e:
            session.rollback()
            log.debug(f"record_trade_result: {e}")
        finally:
            session.close()

    # ── Calibration & adaptive market anchoring ─────────────────────────────

    ANCHOR_NAME = "_market_anchor"

    def calibration_stats(self, limit: int = 200) -> Tuple[int, float, float]:
        """
        Brier score of OUR probability vs simply trusting the MARKET price, over
        recent resolved trades. Lower Brier = better calibrated. Returns
        (n, brier_model, brier_market). p_yes is stored in signal_data at entry;
        the market's implied P(YES) is recovered from the entry price + side.
        """
        session = get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.resolved == True, Trade.resolution.in_(("YES", "NO")),
                Trade.signal_data != None,
            ).order_by(Trade.closed_at.desc()).limit(limit).all()
        finally:
            session.close()

        n = 0
        sse_model = sse_market = 0.0
        for t in trades:
            sig = t.signal_data or {}
            p_yes = sig.get("p_yes")
            if p_yes is None or t.price is None or t.side not in ("YES", "NO"):
                continue
            outcome = 1.0 if t.resolution == "YES" else 0.0
            p_market_yes = t.price if t.side == "YES" else (1.0 - t.price)
            sse_model  += (float(p_yes) - outcome) ** 2
            sse_market += (float(p_market_yes) - outcome) ** 2
            n += 1
        if n == 0:
            return 0, 0.0, 0.0
        return n, sse_model / n, sse_market / n

    def get_market_anchor(self) -> float:
        """Current adaptive STAT_MARKET_WEIGHT (persisted), or the config base."""
        base = float(getattr(config, "STAT_MARKET_WEIGHT", 0.35))
        session = get_session()
        try:
            row = session.query(StrategyWeight).filter_by(name=self.ANCHOR_NAME).first()
            return float(row.weight) if row else base
        finally:
            session.close()

    def update_market_anchor(self) -> float:
        """
        Recompute the market-anchor weight from calibration: if our probability is
        consistently WORSE than the market (higher Brier) we anchor harder to the
        market; if we're beating it we lean on the model more. Bounded + stepped.
        """
        base = float(getattr(config, "STAT_MARKET_WEIGHT", 0.35))
        n, bm, bk = self.calibration_stats()
        anchor = self.get_market_anchor()
        if n >= int(getattr(config, "CALIBRATION_MIN_SAMPLES", 15)):
            step = float(getattr(config, "MARKET_ANCHOR_STEP", 0.05))
            lo   = float(getattr(config, "MARKET_ANCHOR_MIN", 0.20))
            hi   = float(getattr(config, "MARKET_ANCHOR_MAX", 0.70))
            if bm > bk:        # model worse than market → trust market more
                anchor = min(hi, anchor + step)
            else:              # model at least as good → lean on model
                anchor = max(lo, anchor - step)
        else:
            anchor = base
        session = get_session()
        try:
            row = session.query(StrategyWeight).filter_by(name=self.ANCHOR_NAME).first()
            if not row:
                row = StrategyWeight(name=self.ANCHOR_NAME)
                session.add(row)
            row.weight = round(anchor, 4)
            row.updated_at = datetime.utcnow()
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
        return round(anchor, 4)

    # ── Meta-model (GBM edge predictor) ─────────────────────────────────────

    def _build_features(self, trade: Trade) -> np.ndarray | None:
        sig = trade.signal_data or {}
        sub = sig.get("sub", {})
        row = []
        for name in STRATEGY_NAMES:
            info = sub.get(name, {})
            row.append(float(info.get("conf", 0.0)))
            row.append(float(info.get("edge", 0.0)))
            row.append(1.0 if info.get("dir") == trade.side else -1.0 if info.get("dir") else 0.0)
        row.append(float(sig.get("yes_score", 0.0)))
        row.append(float(sig.get("no_score",  0.0)))
        row.append(float(trade.price or 0.5))
        return np.array(row, dtype=np.float32)

    def train_meta_model(self):
        session = get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.resolved == True,
                Trade.signal_data != None
            ).all()

            if len(trades) < config.MIN_TRADES_TO_LEARN:
                return

            X, y = [], []
            for t in trades:
                feats = self._build_features(t)
                if feats is None:
                    continue
                X.append(feats)
                y.append(1 if (t.pnl_usd or 0) > 0 else 0)

            X = np.array(X)
            y = np.array(y)

            self.scaler.fit(X)
            Xs = self.scaler.transform(X)

            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            )
            self.model.fit(Xs, y)
            self.is_fitted = True
            self._save_model()
            log.info(f"Meta-model trained on {len(X)} trades.")
        except Exception as e:
            log.error(f"train_meta_model error: {e}")
        finally:
            session.close()

    def predict_win_probability(self, signal_data: dict, side: str, price: float) -> float:
        """Return estimated win probability using the meta-model (fallback: 0.5)."""
        if not self.is_fitted or self.model is None:
            return 0.5
        try:
            dummy_trade = type("T", (), {
                "signal_data": signal_data, "side": side, "price": price
            })()
            feats = self._build_features(dummy_trade)
            # Guard against a stale model trained on a different feature count
            # (e.g. after adding a strategy). Invalidate so it retrains cleanly.
            expected = getattr(self.scaler, "n_features_in_", feats.shape[0])
            if expected != feats.shape[0]:
                log.info(f"Meta-model feature mismatch ({feats.shape[0]} vs "
                         f"{expected}) — discarding stale model, will retrain.")
                self.is_fitted = False
                self.model = None
                return 0.5
            Xs = self.scaler.transform(feats.reshape(1, -1))
            prob = self.model.predict_proba(Xs)[0][1]
            return float(prob)
        except Exception as e:
            log.warning(f"predict error: {e}")
            return 0.5

    def get_strategy_report(self) -> pd.DataFrame:
        session = get_session()
        try:
            rows = session.query(StrategyWeight).all()
            data = [{
                "Strategy": r.name,
                "Weight":   round(r.weight, 3),
                "Win Rate": f"{r.win_rate:.1%}" if r.win_rate else "N/A",
                "Avg ROI":  f"{r.avg_roi:.1f}%" if r.avg_roi else "N/A",
                "Trades":   r.trade_cnt,
                "Updated":  r.updated_at.strftime("%Y-%m-%d %H:%M") if r.updated_at else "",
            } for r in rows]
            return pd.DataFrame(data)
        finally:
            session.close()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_model(self):
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        with open(config.MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "is_fitted": self.is_fitted,
                         "feature_version": FEATURE_VERSION}, f)

    def _load_model(self):
        if os.path.exists(config.MODEL_PATH):
            try:
                with open(config.MODEL_PATH, "rb") as f:
                    state = pickle.load(f)
                # Discard a model trained under an older feature schema — its
                # inputs no longer mean the same thing, so its predictions would
                # be miscalibrated. It will retrain automatically once enough
                # trades have accumulated under the new format.
                if state.get("feature_version") != FEATURE_VERSION:
                    log.info("Meta-model on disk is stale (feature schema "
                             f"v{state.get('feature_version')} != v{FEATURE_VERSION}) "
                             "— discarding, will retrain on new-format trades.")
                    return
                self.model    = state["model"]
                self.scaler   = state["scaler"]
                self.is_fitted = state["is_fitted"]
                log.info("Meta-model loaded from disk.")
            except Exception as e:
                log.warning(f"Could not load model: {e}")
