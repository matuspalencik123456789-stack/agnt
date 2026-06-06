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

            for name, stats in strategy_stats.items():
                if stats["total"] == 0:
                    continue
                win_rate = stats["wins"] / stats["total"]
                avg_roi  = stats["total_roi"] / stats["total"]
                # weight emphasises directional accuracy vs a coin-flip baseline
                new_weight = max(0.1, min(3.0, (win_rate / 0.5) * (1 + avg_roi / 200)))

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
        row.weight    = round(max(0.1, min(3.0,
                            (row.win_rate / 0.5) * (1 + row.avg_roi / 200))), 4)
        row.trade_cnt = n
        row.updated_at = datetime.utcnow()
        return row.win_rate, row.weight

    def record_trade_result(self, strategy: str, won: bool, roi_pct: float,
                            signal_data: dict = None, resolution: str = None):
        """
        Credit-assignment learning. The agent now judges EACH contributing
        strategy by whether its own directional call matched the real outcome —
        this is the agent's "mind": it figures out which signals were actually
        right, independent of what the ensemble decided to trade.
        """
        session = get_session()
        try:
            # 1. Always update the ensemble's own track record.
            self._update_one(session, "ensemble", won, roi_pct)

            # 2. Credit/blame each sub-strategy by directional correctness.
            sub = (signal_data or {}).get("sub", {})
            if sub and resolution in ("YES", "NO"):
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
                         "is_fitted": self.is_fitted}, f)

    def _load_model(self):
        if os.path.exists(config.MODEL_PATH):
            try:
                with open(config.MODEL_PATH, "rb") as f:
                    state = pickle.load(f)
                self.model    = state["model"]
                self.scaler   = state["scaler"]
                self.is_fitted = state["is_fitted"]
                log.info("Meta-model loaded from disk.")
            except Exception as e:
                log.warning(f"Could not load model: {e}")
