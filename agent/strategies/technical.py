"""Technical-analysis-based strategies using pure numpy/pandas indicators."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agent.strategies.base import BaseStrategy, Signal
from agent.strategies.indicators import rsi, macd, bollinger, adx, vwap
from agent.strategies.outcome_model import (
    realized_vol_per_step, drift_per_step, window_open_price, prob_up,
)
import config


def _parse_dt(value):
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


def _implied_btc_direction(market_meta: dict) -> tuple[str, float]:
    """
    Parse the Polymarket question to get predicted BTC direction & strike.

    Handles both market types:
      • "Will BTC be above/below $X?"  -> direction above/below + strike
      • "Bitcoin Up or Down?" (15-min) -> direction above (YES=Up) / strike 0
    """
    q = market_meta.get("question", "").lower()
    strike = market_meta.get("strike_price", 0.0)

    above = "above" in q or "exceed" in q or "higher" in q
    below = "below" in q or "under" in q or "lower" in q
    if above:
        return "above", strike
    if below:
        return "below", strike

    # "Up or Down" style markets — YES outcome means price goes UP.
    # Treat as "above" with no strike so a bullish signal -> YES, bearish -> NO.
    if "up or down" in q or "up/down" in q or " up " in q or q.endswith(" up") or "higher or lower" in q:
        return "above", 0.0

    return "unknown", strike


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


class StatOutcomeStrategy(BaseStrategy):
    """
    Professional outcome estimator. Looks back at realized volatility + drift and
    computes the *fair* probability the market resolves UP using a digital-option
    random-walk model, then bets the side where the market price misprices it.
    """
    name = "stat_outcome"

    def generate_signal(self, candles, yes_price, no_price, market_meta):
        if len(candles) < 20:
            return self._pass({"reason": "not enough candles"})

        current = float(candles["close"].iloc[-1])
        step_seconds = float(getattr(config, "CANDLE_RESAMPLE_SEC", 0) or 60)

        # strike: explicit "$X" strike, else the window-open price for up/down
        direction, strike = _implied_btc_direction(market_meta)
        start = _parse_dt(market_meta.get("startDate") or market_meta.get("start_date"))
        if not strike or strike <= 0:
            strike = window_open_price(candles, start)
        if not strike or strike <= 0:
            return self._pass({"reason": "no strike/open"})

        end = _parse_dt(market_meta.get("endDate") or market_meta.get("end_date"))
        secs_left = (end - datetime.now(timezone.utc)).total_seconds() if end else 300.0

        vol   = realized_vol_per_step(candles["close"])
        drift = drift_per_step(candles["close"])
        p_up  = prob_up(current, strike, secs_left, step_seconds, vol, drift)
        if p_up is None:
            return self._pass({"reason": "model n/a"})

        # "below" markets: YES means price ends BELOW strike
        p_yes = p_up if direction != "below" else (1.0 - p_up)
        p_no  = 1.0 - p_yes

        # pick the side the model favours
        if p_yes >= p_no:
            bet_side, model_p, mkt_price = "YES", p_yes, yes_price
        else:
            bet_side, model_p, mkt_price = "NO",  p_no,  no_price

        mispricing  = model_p - mkt_price          # >0 = market underprices our side
        conviction  = model_p - 0.5                # how far from a coin-flip

        details = {"p_up": round(p_up, 3), "fair_yes": round(p_yes, 3),
                   "side": bet_side, "model_p": round(model_p, 3),
                   "mkt_price": round(mkt_price, 3), "misprice": round(mispricing, 3),
                   "conviction": round(conviction, 3),
                   "strike": round(strike, 1), "current": round(current, 1),
                   "secs_left": round(secs_left), "vol_step": round(vol, 6)}

        # Require minimum conviction AND actual market mispricing.
        # Don't bet against an efficient market just because our model feels sure.
        min_conv = float(getattr(config, "STAT_MIN_CONVICTION", 0.06))
        if conviction < min_conv:
            details["reason"] = f"low conviction {conviction:.3f} < {min_conv}"
            return self._pass(details)

        min_misprice = float(getattr(config, "STAT_MIN_MISPRICING", 0.03))
        if mispricing < min_misprice:
            details["reason"] = f"misprice {mispricing:.3f} < {min_misprice}"
            return self._pass(details)

        return Signal(self.name, bet_side, min(model_p, 1.0), mispricing, details)
