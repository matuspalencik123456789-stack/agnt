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


def prob_up(current: float, strike: float, seconds_remaining: float,
            step_seconds: float, vol_per_step: float,
            drift_step: float = 0.0) -> Optional[float]:
    """
    Probability that the final price exceeds `strike`, given the move so far and
    the projected random walk over the remaining time.
    """
    if current <= 0 or strike <= 0 or vol_per_step <= 0:
        return None

    moved = math.log(current / strike)               # move already achieved
    if seconds_remaining <= 0:
        # window over → outcome is essentially decided
        return 1.0 if moved > 0 else 0.0

    steps_left = max(seconds_remaining / max(step_seconds, 1.0), 1e-6)
    sigma_rem  = vol_per_step * math.sqrt(steps_left)
    if sigma_rem <= 0:
        return 1.0 if moved > 0 else 0.0

    # cap drift contribution so we never get overconfident on a short window
    mu_rem = drift_step * steps_left
    mu_rem = max(-0.5 * sigma_rem, min(0.5 * sigma_rem, mu_rem))

    z = (moved + mu_rem) / sigma_rem
    return _norm_cdf(z)
