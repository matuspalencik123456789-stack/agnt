"""
Statistical outcome model for short-horizon BTC "Up or Down" markets.

A 15-min "Bitcoin Up or Down" market resolves UP if the price at the end of the
window is higher than the price at the START of the window (the window open).
That makes the fair value a *digital option*:

    P(up) = P( S_T > K )

where K is the window-open price (or an explicit strike for "above $X" markets),
S_t is the current price, and the remaining move over τ seconds is modelled as a
zero-/low-drift random walk calibrated on *recent realized volatility* — i.e. we
look back at how the market actually moved and project it forward.

    ln(S_T / K) = ln(S_t / K) + N(μ_rem, σ_rem²)
    z      = (ln(S_t / K) + μ_rem) / σ_rem
    P(up)  = Φ(z)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def realized_vol_per_step(closes: pd.Series, lookback: int = 60) -> float:
    """Std-dev of log returns over the last `lookback` candles (per-candle σ)."""
    if len(closes) < 5:
        return 0.0
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) > lookback:
        rets = rets.iloc[-lookback:]
    return float(rets.std())


def drift_per_step(closes: pd.Series, lookback: int = 60) -> float:
    """Mean log return over the lookback window (per-candle drift)."""
    if len(closes) < 5:
        return 0.0
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) > lookback:
        rets = rets.iloc[-lookback:]
    return float(rets.mean())


def window_open_price(candles: pd.DataFrame, start: Optional[datetime]) -> Optional[float]:
    """Price at the start of the current window (the 'open' the market is judged against)."""
    if candles is None or candles.empty:
        return None
    if start is None:
        # no start known → use the earliest candle we have as a proxy
        return float(candles["open"].iloc[0])
    try:
        idx = candles.index
        if getattr(idx, "tz", None) is None:
            start_cmp = start.replace(tzinfo=None)
        else:
            start_cmp = start.astimezone(timezone.utc)
        at_or_after = candles[idx >= start_cmp]
        if not at_or_after.empty:
            return float(at_or_after["open"].iloc[0])
        # window started before our candle history → use earliest open
        return float(candles["open"].iloc[0])
    except Exception:
        return float(candles["open"].iloc[0])


def analyze_trend(closes: pd.Series, lookback: int = 30,
                  min_strength: float = 0.0003) -> dict:
    """
    Determine the short-term price development ("vývoj") from recent candles.

    Fits a linear regression to the last `lookback` closes and normalises the
    slope by price (slope per candle, as a fraction). Also compares a fast vs
    slow moving average for confirmation. Returns direction UP/DOWN/FLAT plus a
    strength score so callers can require a real, non-flat trend.
    """
    if closes is None or len(closes) < 5:
        return {"direction": "FLAT", "strength": 0.0, "slope": 0.0, "n": 0}

    series = closes.dropna()
    if len(series) > lookback:
        series = series.iloc[-lookback:]
    n = len(series)
    y = series.to_numpy(dtype=float)
    x = np.arange(n, dtype=float)

    # least-squares slope, normalised to a per-candle fractional change
    slope = float(np.polyfit(x, y, 1)[0])
    norm_slope = slope / (y.mean() if y.mean() else 1.0)

    # moving-average confirmation
    fast = y[-max(3, n // 4):].mean()
    slow = y.mean()
    ma_up = fast > slow

    strength = abs(norm_slope)
    if strength < min_strength:
        direction = "FLAT"
    elif norm_slope > 0 and ma_up:
        direction = "UP"
    elif norm_slope < 0 and not ma_up:
        direction = "DOWN"
    else:
        # slope and MA disagree → unconfirmed → treat as flat/choppy
        direction = "FLAT"

    return {"direction": direction, "strength": round(strength, 6),
            "slope": round(norm_slope, 6), "n": n}


def prob_up(current: float, strike: float, seconds_remaining: float,
            step_seconds: float, vol_per_step: float,
            drift_step: float = 0.0,
            mean_rev: float = 0.30) -> Optional[float]:
    """
    Probability that the final price exceeds `strike`.

    Uses a lightly mean-reverting random walk: short-horizon crypto prices show
    partial mean reversion, so an already-moved log-price is pulled back toward
    zero by factor `mean_rev` before computing the remaining diffusion.
    Result is capped to [0.10, 0.90] — extreme certainty is never warranted.
    """
    if current <= 0 or strike <= 0 or vol_per_step <= 0:
        return None

    moved = math.log(current / strike)
    if seconds_remaining <= 0:
        return 1.0 if moved > 0 else 0.0

    steps_left = max(seconds_remaining / max(step_seconds, 1.0), 1e-6)
    sigma_rem  = vol_per_step * math.sqrt(steps_left)
    if sigma_rem <= 0:
        return 1.0 if moved > 0 else 0.0

    # Mean-reversion pull: shrink the "already moved" contribution
    adjusted_moved = moved * (1.0 - mean_rev)

    mu_rem = drift_step * steps_left
    mu_rem = max(-0.3 * sigma_rem, min(0.3 * sigma_rem, mu_rem))

    z = (adjusted_moved + mu_rem) / sigma_rem
    raw = _norm_cdf(z)

    # Cap to [0.10, 0.90] — extreme confidence on a 15-min window is unjustified
    return max(0.10, min(0.90, raw))
