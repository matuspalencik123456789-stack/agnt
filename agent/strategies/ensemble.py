"""Probability-first ensemble.

The decision core is the digital-option model (`stat_outcome`): it produces the
*fair probability* that the market resolves YES, already Bayesian-anchored to the
market price. The six technical indicators are NOT independent voters — on a flat
15-min binary they are mostly noise — so they only REFINE the model's probability
by a bounded amount in log-odds space, and can never flip the direction the model
chose (if they disagree they can null the trade, not reverse it).

Edge is measured honestly as (fair probability − market price), never as
vote-share, so the agent trades only a genuine mispricing and otherwise passes.
This replaces the old `edge = vote_confidence − price` scheme, which rewarded
buying cheap losers and let 6 noise strategies outvote the one principled model.
"""
import logging
import math
from typing import Dict, List

import pandas as pd

from agent.strategies.base import BaseStrategy, Signal
from agent.strategies.technical import (
    RSIStrategy, MACDStrategy, BollingerStrategy,
    MomentumStrategy, VWAPStrategy, ADXStrategy, StatOutcomeStrategy,
    PriceActionStrategy,
)
import config

log = logging.getLogger(__name__)

ALL_STRATEGIES: List[BaseStrategy] = [
    RSIStrategy(), MACDStrategy(), BollingerStrategy(),
    MomentumStrategy(), VWAPStrategy(), ADXStrategy(),
    StatOutcomeStrategy(), PriceActionStrategy(),
]

# The statistical outcome model and the pro price-action read are the most
# principled signals — start them heavier.
DEFAULT_WEIGHTS = {
    "rsi": 1.0, "macd": 1.0, "bollinger": 1.0,
    "momentum": 1.0, "vwap": 1.0, "adx": 1.0,
    "stat_outcome": 1.5, "price_action": 1.5,
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EnsembleStrategy:
    def __init__(self, weights: Dict[str, float] = None):
        self.strategies = {s.name: s for s in ALL_STRATEGIES}
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def update_weights(self, weights: Dict[str, float]):
        self.weights = weights

    def generate_signal(self, candles: pd.DataFrame,
                        yes_price: float, no_price: float,
                        market_meta: dict) -> Signal:
        # ── Run every sub-strategy once ───────────────────────────────────────
        signals: Dict[str, Signal] = {}
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(
                    candles, yes_price, no_price, market_meta)
            except Exception as e:
                log.warning(f"Strategy {name} failed: {e}")

        stat = signals.get("stat_outcome")
        if stat is not None:
            d = stat.details
            log.info(f"  Model: P(up)={d.get('p_up','?')} → {d.get('side','-')} "
                     f"(model_p={d.get('model_p','?')} mkt={d.get('mkt_price','?')} "
                     f"conv={d.get('conviction','?')} {stat.direction})")

        # Sub-signal record consumed by the self-learner (contract unchanged).
        sub_details = {name: {"dir": s.direction, "conf": round(s.confidence, 3),
                              "edge": round(s.edge, 3)}
                       for name, s in signals.items() if s.direction != "PASS"}

        # ── 1. Base probability = the digital-option model (the brain) ────────
        # `fair_yes` is always present in the model details (even when the model
        # itself PASSes on its own conviction/mispricing gate) and is already a
        # 65/35 blend of the model estimate and the market price.
        fair_yes = stat.details.get("fair_yes") if stat is not None else None
        if fair_yes is None:
            return Signal("ensemble", "PASS", 0.0, 0.0,
                          {"reason": "outcome model unavailable", "sub": sub_details})

        # ── 2. Near coin-flip → the model has no view; don't trade noise ──────
        min_conv = float(getattr(config, "PROB_MIN_CONVICTION", 0.05))
        if abs(fair_yes - 0.5) < min_conv:
            return Signal("ensemble", "PASS", fair_yes, 0.0,
                          {"reason": f"model near coin-flip (p_yes={fair_yes:.3f})",
                           "sub": sub_details, "p_yes": round(fair_yes, 4)})

        # ── 3. Bounded technical refinement (cannot flip the model) ───────────
        # Each non-PASS technical read contributes a signed, weight-scaled vote in
        # [-1, +1]; their weighted mean nudges the model's log-odds by at most
        # TECH_TILT_MAX. Strong agreement strengthens conviction; disagreement
        # weakens it — but the model alone owns the direction.
        min_w = float(getattr(config, "STRATEGY_MIN_WEIGHT", 0.40))
        num = den = 0.0
        excluded = []
        for name, s in signals.items():
            if name == "stat_outcome" or s.direction == "PASS":
                continue
            w = self.weights.get(name, 1.0)
            if w < min_w:
                excluded.append(name)
                continue
            vote = s.confidence if s.direction == "YES" else -s.confidence
            num += w * vote
            den += w
        tilt = (num / den) if den else 0.0          # weighted mean signed vote
        tilt_max = float(getattr(config, "TECH_TILT_MAX", 0.6))
        p_yes = _sigmoid(_logit(fair_yes) + tilt_max * tilt)

        # Technicals may strengthen or null the trade, never reverse the model.
        if (fair_yes - 0.5) * (p_yes - 0.5) <= 0:
            return Signal("ensemble", "PASS", 0.5, 0.0,
                          {"reason": f"model/technical conflict "
                                     f"(model {fair_yes:.3f}, tilt {tilt:+.2f})",
                           "sub": sub_details, "p_yes": round(p_yes, 4)})

        p_yes = min(0.98, max(0.02, p_yes))
        p_no  = 1.0 - p_yes

        if p_yes >= 0.5:
            direction, p_side, price, fair_side = "YES", p_yes, yes_price, fair_yes
        else:
            direction, p_side, price, fair_side = "NO", p_no, no_price, 1.0 - fair_yes

        # The edge that JUSTIFIES a trade comes from the MODEL diverging from the
        # market (fair_side − price), not from the technical tilt. The tilt only
        # feeds `confidence` (p_side) for sizing; it can't manufacture a reason to
        # enter over a fairly-priced market.
        edge = fair_side - price

        details = {"p_yes": round(p_yes, 4), "fair_yes": round(fair_yes, 4),
                   "tilt": round(tilt, 3),
                   # keep yes_score/no_score keys for the meta-model feature builder
                   "yes_score": round(p_yes, 4), "no_score": round(p_no, 4),
                   "sub": sub_details}
        if excluded:
            details["excl"] = excluded

        # ── 4. Final conviction + honest-edge gates ───────────────────────────
        if abs(p_yes - 0.5) < min_conv:
            details["reason"] = f"low conviction after tilt (p_yes={p_yes:.3f})"
            return Signal("ensemble", "PASS", p_side, max(0.0, edge), details)

        if edge < config.MIN_EDGE_THRESHOLD:
            details["reason"] = (f"edge {edge:.3f} < {config.MIN_EDGE_THRESHOLD:.3f} "
                                 f"(model fair {fair_side:.3f} vs price {price:.3f})")
            return Signal("ensemble", "PASS", p_side, max(0.0, edge), details)

        excl_str = f" excl={excluded}" if excluded else ""
        log.info(f"  Ensemble: p_yes={p_yes:.3f} (model {fair_yes:.3f}, tilt {tilt:+.2f}) "
                 f"→ {direction} conf={p_side:.3f} edge={edge:.3f} price={price:.3f}{excl_str}")
        return Signal("ensemble", direction, p_side, edge, details)
