"""Weighted ensemble that combines all strategy signals."""
import logging
from typing import Dict, List
import pandas as pd

from agent.strategies.base import BaseStrategy, Signal
from agent.strategies.technical import (
    RSIStrategy, MACDStrategy, BollingerStrategy,
    MomentumStrategy, VWAPStrategy, ADXStrategy, StatOutcomeStrategy
)
import config

log = logging.getLogger(__name__)

ALL_STRATEGIES: List[BaseStrategy] = [
    RSIStrategy(), MACDStrategy(), BollingerStrategy(),
    MomentumStrategy(), VWAPStrategy(), ADXStrategy(),
    StatOutcomeStrategy(),
]

# The statistical outcome model is the most principled signal — start it heavier.
DEFAULT_WEIGHTS = {
    "rsi": 1.0, "macd": 1.0, "bollinger": 1.0,
    "momentum": 1.0, "vwap": 1.0, "adx": 1.0,
    "stat_outcome": 2.5,
}


class EnsembleStrategy:
    def __init__(self, weights: Dict[str, float] = None):
        self.strategies = {s.name: s for s in ALL_STRATEGIES}
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def update_weights(self, weights: Dict[str, float]):
        self.weights = weights

    def generate_signal(self, candles: pd.DataFrame,
                        yes_price: float, no_price: float,
                        market_meta: dict) -> Signal:
        signals: List[Signal] = []
        for name, strategy in self.strategies.items():
            try:
                sig = strategy.generate_signal(candles, yes_price, no_price, market_meta)
                signals.append(sig)
                if sig.direction != "PASS":
                    log.debug(f"    [{name}] {sig.direction} conf={sig.confidence:.3f} edge={sig.edge:.3f}")
                else:
                    log.debug(f"    [{name}] PASS — {sig.details.get('reason','')}")
            except Exception as e:
                log.warning(f"Strategy {name} failed: {e}")

        yes_score = 0.0
        no_score  = 0.0
        total_w   = 0.0

        for sig in signals:
            if sig.direction == "PASS":
                continue
            w = self.weights.get(sig.strategy, 1.0)
            score = w * sig.confidence * max(sig.edge, 0.01)
            if sig.direction == "YES":
                yes_score += score
            else:
                no_score  += score
            total_w += w

        if total_w == 0 or (yes_score == 0 and no_score == 0):
            return Signal("ensemble", "PASS", 0.0, 0.0,
                          {"reason": "no non-pass signals"})

        if yes_score > no_score:
            direction = "YES"
            raw_conf  = yes_score / (yes_score + no_score)
            price     = yes_price
        else:
            direction = "NO"
            raw_conf  = no_score / (yes_score + no_score)
            price     = no_price

        log.info(f"  Ensemble: YES={yes_score:.4f} NO={no_score:.4f} "
                 f"→ {direction} conf={raw_conf:.3f} price={price:.3f}")

        # require simple majority (≥ 0.51)
        if raw_conf < 0.51:
            return Signal("ensemble", "PASS", raw_conf, 0.0,
                          {"reason": f"weak consensus {raw_conf:.2f}", "yes": yes_score, "no": no_score})

        edge = max(0, raw_conf - price)
        if edge < config.MIN_EDGE_THRESHOLD:
            return Signal("ensemble", "PASS", raw_conf, edge,
                          {"reason": f"edge {edge:.3f} < {config.MIN_EDGE_THRESHOLD:.3f}"})

        sub_details = {s.strategy: {"dir": s.direction, "conf": round(s.confidence, 3),
                                    "edge": round(s.edge, 3)}
                       for s in signals if s.direction != "PASS"}

        return Signal("ensemble", direction, raw_conf, edge,
                      {"yes_score": round(yes_score, 4), "no_score": round(no_score, 4),
                       "sub": sub_details})
