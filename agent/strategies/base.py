"""Base class for all trading strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    strategy: str
    direction: str        # "YES" | "NO" | "PASS"
    confidence: float     # 0.0 – 1.0
    edge: float           # estimated edge over market price
    details: dict


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, candles: pd.DataFrame,
                        yes_price: float, no_price: float,
                        market_meta: dict) -> Signal:
        """Return a Signal given current market state."""
        ...

    def _pass(self, details: dict = None) -> Signal:
        return Signal(self.name, "PASS", 0.0, 0.0, details or {})
