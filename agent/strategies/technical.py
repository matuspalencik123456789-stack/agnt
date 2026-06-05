"""Technical-analysis-based strategies using pure numpy/pandas indicators."""
import numpy as np
import pandas as pd

from agent.strategies.base import BaseStrategy, Signal
from agent.strategies.indicators import rsi, macd, bollinger, adx, vwap


def _implied_btc_direction(market_meta: dict) -> tuple[str, float]:
    """Parse the Polymarket question to get predicted BTC direction & strike."""
    q = market_meta.get("question", "").lower()
    strike = market_meta.get("strike_price", 0.0)
    above = "above" in q or "exceed" in q or "higher" in q
    below = "below" in q or "under" in q or "lower" in q
    return ("above" if above else "below" if below else "unknown"), strike


class RSIStrategy(BaseStrategy):
    name = "rsi"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 30:
            return self._pass({"reason": "not enough candles"})

        rsi_val = rsi(candles["close"], 14).iloc[-1]
        if np.isnan(rsi_val):
            return self._pass({"reason": "rsi nan"})

        direction, strike = _implied_btc_direction(market_meta)
        current = candles["close"].iloc[-1]

        bullish = rsi_val < 35
        bearish = rsi_val > 65

        bet_side = "PASS"
        confidence = 0.0

        if direction == "above":
            if bullish:
                bet_side = "YES"; confidence = (35 - rsi_val) / 35 * 0.8
            elif bearish:
                bet_side = "NO";  confidence = (rsi_val - 65) / 35 * 0.7
        elif direction == "below":
            if bearish:
                bet_side = "YES"; confidence = (rsi_val - 65) / 35 * 0.8
            elif bullish:
                bet_side = "NO";  confidence = (35 - rsi_val) / 35 * 0.7

        if bet_side == "PASS":
            return self._pass({"rsi": round(rsi_val, 2)})

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, min(confidence, 1.0), edge,
                      {"rsi": round(rsi_val, 2), "strike": strike, "current": round(current, 0)})


class MACDStrategy(BaseStrategy):
    name = "macd"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 40:
            return self._pass()

        _, _, hist = macd(candles["close"])
        h     = hist.iloc[-1]
        h_prev = hist.iloc[-2]

        if np.isnan(h) or np.isnan(h_prev):
            return self._pass({"reason": "macd nan"})

        crossover_bull = h_prev < 0 < h
        crossover_bear = h_prev > 0 > h
        direction, _ = _implied_btc_direction(market_meta)

        bet_side = "PASS"
        confidence = 0.0

        if direction == "above":
            if crossover_bull:
                bet_side = "YES"; confidence = 0.65
            elif crossover_bear:
                bet_side = "NO";  confidence = 0.60
        elif direction == "below":
            if crossover_bear:
                bet_side = "YES"; confidence = 0.65
            elif crossover_bull:
                bet_side = "NO";  confidence = 0.60

        if bet_side == "PASS":
            return self._pass({"hist": round(h, 4)})

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, confidence, edge,
                      {"hist": round(h, 4), "cross": "bull" if crossover_bull else "bear"})


class BollingerStrategy(BaseStrategy):
    name = "bollinger"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 30:
            return self._pass()

        upper, mid, lower = bollinger(candles["close"])
        u = upper.iloc[-1]; l = lower.iloc[-1]
        close = candles["close"].iloc[-1]

        if np.isnan(u) or np.isnan(l) or (u - l) == 0:
            return self._pass({"reason": "bb nan"})

        pct_b = (close - l) / (u - l)
        direction, _ = _implied_btc_direction(market_meta)

        near_lower = pct_b < 0.15
        near_upper = pct_b > 0.85

        bet_side = "PASS"
        confidence = 0.0

        if direction == "above":
            if near_lower:
                bet_side = "YES"; confidence = 0.55 + (0.15 - pct_b) * 1.5
            elif near_upper:
                bet_side = "NO";  confidence = 0.55 + (pct_b - 0.85) * 1.5
        elif direction == "below":
            if near_upper:
                bet_side = "YES"; confidence = 0.55 + (pct_b - 0.85) * 1.5
            elif near_lower:
                bet_side = "NO";  confidence = 0.55 + (0.15 - pct_b) * 1.5

        if bet_side == "PASS":
            return self._pass({"pct_b": round(pct_b, 3)})

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, min(confidence, 1.0), edge,
                      {"pct_b": round(pct_b, 3), "upper": round(u, 0), "lower": round(l, 0)})


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 10:
            return self._pass()

        returns = candles["close"].pct_change()
        mom_4 = candles["close"].iloc[-1] / candles["close"].iloc[-4]  - 1
        mom_8 = candles["close"].iloc[-1] / candles["close"].iloc[-8]  - 1
        vol   = returns.iloc[-20:].std()
        trend = (mom_4 + mom_8) / 2

        if np.isnan(trend) or vol == 0:
            return self._pass()

        direction, _ = _implied_btc_direction(market_meta)
        strong_bull = trend > vol
        strong_bear = trend < -vol

        bet_side = "PASS"
        confidence = 0.0

        if direction == "above":
            if strong_bull:
                bet_side = "YES"; confidence = 0.55 + min(trend / vol * 0.1, 0.2)
            elif strong_bear:
                bet_side = "NO";  confidence = 0.55 + min(-trend / vol * 0.1, 0.2)
        elif direction == "below":
            if strong_bear:
                bet_side = "YES"; confidence = 0.55 + min(-trend / vol * 0.1, 0.2)
            elif strong_bull:
                bet_side = "NO";  confidence = 0.55 + min(trend / vol * 0.1, 0.2)

        if bet_side == "PASS":
            return self._pass({"trend": round(trend, 5)})

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, min(confidence, 1.0), edge,
                      {"mom_4": round(mom_4, 4), "mom_8": round(mom_8, 4), "vol": round(vol, 4)})


class VWAPStrategy(BaseStrategy):
    name = "vwap"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 30:
            return self._pass()

        vwap_val = vwap(candles["high"], candles["low"], candles["close"],
                        candles["volume"]).iloc[-1]
        close = candles["close"].iloc[-1]

        if np.isnan(vwap_val) or vwap_val == 0:
            return self._pass({"reason": "vwap nan"})

        dev = (close - vwap_val) / vwap_val
        direction, _ = _implied_btc_direction(market_meta)

        bet_side = "PASS"
        confidence = 0.0

        if direction == "above":
            if dev > 0.002:
                bet_side = "YES"; confidence = 0.55 + min(dev * 10, 0.2)
            elif dev < -0.002:
                bet_side = "NO";  confidence = 0.55 + min(-dev * 10, 0.2)
        elif direction == "below":
            if dev < -0.002:
                bet_side = "YES"; confidence = 0.55 + min(-dev * 10, 0.2)
            elif dev > 0.002:
                bet_side = "NO";  confidence = 0.55 + min(dev * 10, 0.2)

        if bet_side == "PASS":
            return self._pass({"vwap": round(vwap_val, 0), "dev": round(dev, 5)})

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, min(confidence, 1.0), edge,
                      {"vwap": round(vwap_val, 0), "close": round(close, 0), "dev": round(dev, 5)})


class ADXStrategy(BaseStrategy):
    name = "adx"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 30:
            return self._pass()

        adx_val, dip, din = adx(candles["high"], candles["low"], candles["close"])
        a = adx_val.iloc[-1]
        if np.isnan(a) or a < 20:
            return self._pass({"adx": round(a, 2) if not np.isnan(a) else 0, "reason": "weak trend"})

        direction, _ = _implied_btc_direction(market_meta)
        bullish = dip.iloc[-1] > din.iloc[-1]
        bearish = not bullish
        confidence = 0.55 + min((a - 20) / 60, 0.25)

        if direction == "above":
            bet_side = "YES" if bullish else "NO"
        elif direction == "below":
            bet_side = "YES" if bearish else "NO"
        else:
            return self._pass()

        price = yes_price if bet_side == "YES" else no_price
        edge = max(0.0, confidence - price)
        return Signal(self.name, bet_side, confidence, edge,
                      {"adx": round(a, 2), "+di": round(dip.iloc[-1], 2), "-di": round(din.iloc[-1], 2)})
