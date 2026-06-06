"""
Professional chart-reading toolkit — the techniques top discretionary and quant
traders actually use to read price, distilled into composable functions:

  • Market structure (Dow Theory / Al Brooks): higher-highs+higher-lows = uptrend,
    lower-highs+lower-lows = downtrend. The single most important read.
  • Swing pivots: local highs/lows that define structure and S/R.
  • Support / resistance: recent pivot clusters the price respects.
  • Candlestick patterns: bullish/bearish engulfing, hammer, shooting star, pin bar.
  • RSI divergence: price makes a new extreme the momentum doesn't confirm.
  • Volume confirmation: a move backed by rising volume is more trustworthy.

Each returns a small, normalised read so a strategy can score *confluence* — the
professional principle that no single signal trades alone; edge comes from
several independent reads agreeing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent.strategies.indicators import rsi, atr


def swing_pivots(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    """Return (swing_high_idx, swing_low_idx) — fractal pivots à la Bill Williams."""
    highs, lows = [], []
    h = high.to_numpy(); l = low.to_numpy()
    n = len(h)
    for i in range(left, n - right):
        if h[i] == max(h[i - left:i + right + 1]) and h[i] > h[i - 1]:
            highs.append(i)
        if l[i] == min(l[i - left:i + right + 1]) and l[i] < l[i - 1]:
            lows.append(i)
    return highs, lows


def market_structure(high: pd.Series, low: pd.Series, lookback: int = 60) -> dict:
    """
    Classify trend from swing structure (Dow Theory):
      HH + HL → UPTREND, LH + LL → DOWNTREND, else RANGE.
    """
    high = high.iloc[-lookback:]; low = low.iloc[-lookback:]
    hi_idx, lo_idx = swing_pivots(high, low)
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return {"trend": "RANGE", "hh": False, "ll": False, "n_pivots": len(hi_idx) + len(lo_idx)}

    last_highs = high.to_numpy()[hi_idx][-2:]
    last_lows  = low.to_numpy()[lo_idx][-2:]
    hh = last_highs[-1] > last_highs[-2]
    hl = last_lows[-1]  > last_lows[-2]
    lh = last_highs[-1] < last_highs[-2]
    ll = last_lows[-1]  < last_lows[-2]

    if hh and hl:
        trend = "UPTREND"
    elif lh and ll:
        trend = "DOWNTREND"
    else:
        trend = "RANGE"
    return {"trend": trend, "hh": bool(hh), "hl": bool(hl),
            "lh": bool(lh), "ll": bool(ll), "n_pivots": len(hi_idx) + len(lo_idx)}


def support_resistance(high: pd.Series, low: pd.Series, close: float,
                       lookback: int = 80) -> dict:
    """Nearest support (below) and resistance (above) from recent pivots."""
    high = high.iloc[-lookback:]; low = low.iloc[-lookback:]
    hi_idx, lo_idx = swing_pivots(high, low)
    res_levels = high.to_numpy()[hi_idx] if hi_idx else np.array([])
    sup_levels = low.to_numpy()[lo_idx]  if lo_idx else np.array([])

    res = res_levels[res_levels > close]
    sup = sup_levels[sup_levels < close]
    nearest_res = float(res.min()) if res.size else None
    nearest_sup = float(sup.max()) if sup.size else None
    return {"resistance": nearest_res, "support": nearest_sup}


def candlestick_pattern(candles: pd.DataFrame) -> dict:
    """Detect the most recent classic reversal candle. Returns {pattern, bias}."""
    if len(candles) < 2:
        return {"pattern": None, "bias": 0}
    o = candles["open"].to_numpy(); c = candles["close"].to_numpy()
    h = candles["high"].to_numpy(); l = candles["low"].to_numpy()

    o1, c1, h1, l1 = o[-1], c[-1], h[-1], l[-1]
    o2, c2 = o[-2], c[-2]
    body = abs(c1 - o1)
    rng  = (h1 - l1) or 1e-9
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    # bullish / bearish engulfing
    if c2 < o2 and c1 > o1 and c1 >= o2 and o1 <= c2:
        return {"pattern": "bullish_engulfing", "bias": +1}
    if c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
        return {"pattern": "bearish_engulfing", "bias": -1}
    # hammer / pin bar (long lower wick, small body near top)
    if lower_wick > 2 * body and upper_wick < body and body / rng < 0.4:
        return {"pattern": "hammer", "bias": +1}
    # shooting star (long upper wick, small body near bottom)
    if upper_wick > 2 * body and lower_wick < body and body / rng < 0.4:
        return {"pattern": "shooting_star", "bias": -1}
    return {"pattern": None, "bias": 0}


def rsi_divergence(close: pd.Series, lookback: int = 40) -> dict:
    """
    Classic momentum divergence: price prints a new low but RSI doesn't (bullish),
    or a new high but RSI doesn't (bearish).
    """
    if len(close) < lookback + 5:
        return {"divergence": None, "bias": 0}
    r = rsi(close, 14).dropna()
    if len(r) < lookback:
        return {"divergence": None, "bias": 0}
    px = close.iloc[-lookback:]; rr = r.iloc[-lookback:]
    half = lookback // 2
    p_lo_recent, p_lo_prev = px.iloc[-half:].min(), px.iloc[:half].min()
    r_lo_recent, r_lo_prev = rr.iloc[-half:].min(), rr.iloc[:half].min()
    p_hi_recent, p_hi_prev = px.iloc[-half:].max(), px.iloc[:half].max()
    r_hi_recent, r_hi_prev = rr.iloc[-half:].max(), rr.iloc[:half].max()

    if p_lo_recent < p_lo_prev and r_lo_recent > r_lo_prev:
        return {"divergence": "bullish", "bias": +1}
    if p_hi_recent > p_hi_prev and r_hi_recent < r_hi_prev:
        return {"divergence": "bearish", "bias": -1}
    return {"divergence": None, "bias": 0}


def volume_confirmation(volume: pd.Series, lookback: int = 20) -> float:
    """Ratio of recent volume to its average — >1 means the move is backed."""
    if len(volume) < lookback or volume.iloc[-lookback:].mean() == 0:
        return 1.0
    recent = volume.iloc[-3:].mean()
    base   = volume.iloc[-lookback:].mean()
    return float(recent / base) if base else 1.0


def read_chart(candles: pd.DataFrame) -> dict:
    """
    Combine every professional read into one confluence score in [-1, +1]
    (positive = bullish). This is the 'expert eye' summary.
    """
    close = candles["close"]; high = candles["high"]
    low = candles["low"]; vol = candles["volume"]
    px = float(close.iloc[-1])

    struct = market_structure(high, low)
    sr     = support_resistance(high, low, px)
    pat    = candlestick_pattern(candles)
    div    = rsi_divergence(close)
    vmult  = volume_confirmation(vol)

    score = 0.0
    # 1. market structure — the heaviest weight (trend is king)
    if struct["trend"] == "UPTREND":   score += 0.40
    elif struct["trend"] == "DOWNTREND": score -= 0.40
    # 2. candlestick reversal pattern
    score += 0.20 * pat["bias"]
    # 3. RSI divergence
    score += 0.20 * div["bias"]
    # 4. proximity to S/R (bounce off support = bullish, reject at resistance = bearish)
    if sr["support"] and (px - sr["support"]) / px < 0.0015:
        score += 0.15
    if sr["resistance"] and (sr["resistance"] - px) / px < 0.0015:
        score -= 0.15
    # 5. volume confirmation amplifies an existing read
    if vmult > 1.3:
        score *= 1.15

    score = max(-1.0, min(1.0, score))
    return {
        "score": round(score, 3),
        "structure": struct["trend"],
        "pattern": pat["pattern"],
        "divergence": div["divergence"],
        "support": round(sr["support"], 1) if sr["support"] else None,
        "resistance": round(sr["resistance"], 1) if sr["resistance"] else None,
        "vol_mult": round(vmult, 2),
    }
