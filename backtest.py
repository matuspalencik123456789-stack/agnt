"""
Offline backtest for the BTC 15-min "Up or Down" strategy.

Fetches historical 1-minute BTC candles from Binance, walks them window by
window, and on each 15-min window runs the live ensemble exactly as the agent
would — then resolves the bet by comparing the window's close vs its open.

Two things make this an HONEST test of whether a real edge exists:

  1. FULL FEES.  Polymarket charges a crypto taker fee on entry (and again on
     an early sell). We apply config.taker_fee() so the P&L reflects what you'd
     actually keep — not a fee-free fantasy.

  2. WALK-FORWARD.  Self-learning that's "tuned" on the same data it's tested on
     will always look good (overfitting). We split the timeline: learn strategy
     weights on the FIRST chunk, then trade the held-out LATER chunk the model
     has never seen. If the edge survives that, it's more likely real.

Usage:
    python backtest.py                      # ~16h, walk-forward 50/50
    python backtest.py --hours 72           # more history (paged requests)
    python backtest.py --decide-min 3       # decide N minutes into each window
    python backtest.py --train-frac 0.6     # train on first 60%, test on last 40%
    python backtest.py --no-walk-forward    # single pass, default weights
"""
import argparse
import logging
from datetime import timedelta

import pandas as pd
import requests

import config
from agent.strategies.ensemble import EnsembleStrategy, DEFAULT_WEIGHTS

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("backtest")

KLINES = f"{config.BINANCE_API}/api/v3/klines"


def fetch_1m(hours: int) -> pd.DataFrame:
    """Fetch `hours` worth of 1-minute candles (paged, Binance caps 1000/req)."""
    need = hours * 60
    out = []
    end = None
    while need > 0:
        limit = min(1000, need)
        params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit}
        if end:
            params["endTime"] = end
        try:
            resp = requests.get(KLINES, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"Could not reach Binance ({e}).\n"
                  f"Run this on a machine with internet access to Binance.")
            break
        if not raw:
            break
        out = raw + out
        end = raw[0][0] - 1     # page backwards
        need -= len(raw)
        if len(raw) < limit:
            break
    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tb", "tq", "ig"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")


def evaluate_window(df, ens, w, decide_min, spread):
    """Run the ensemble on one 15-min window. Returns a trade dict or None."""
    w_open  = w
    w_close = w + timedelta(minutes=15)
    decide  = w_open + timedelta(minutes=decide_min)

    hist = df[df.index <= decide]
    if len(hist) < 60:                        # need enough lookback
        return None
    try:
        open_px  = float(df[df.index >= w_open]["open"].iloc[0])
        close_px = float(df[df.index < w_close]["close"].iloc[-1])
    except IndexError:
        return None

    meta = {
        "question": "Bitcoin Up or Down?",
        "startDate": w_open.isoformat(),
        "endDate":   w_close.isoformat(),
        "strike_price": 0.0,
    }
    yes_mid = no_mid = 0.5                     # at the open the market is ~coin-flip
    sig = ens.generate_signal(hist, yes_mid, no_mid, meta)
    if sig.direction == "PASS":
        return None

    fill   = 0.5 + spread / 2.0               # pay the ask
    shares = config.MAX_POSITION_SIZE_USD / fill
    resolution = "YES" if close_px > open_px else "NO"
    won = (sig.direction == resolution)

    # P&L with the REAL entry taker fee (settlement isn't a taker → no exit fee)
    payout    = (1.0 if won else 0.0) * shares
    entry_fee = config.taker_fee(shares, fill)
    pnl = payout - config.MAX_POSITION_SIZE_USD - entry_fee
    roi = 100.0 * pnl / config.MAX_POSITION_SIZE_USD

    return {
        "time": w_open, "dir": sig.direction, "res": resolution, "won": won,
        "conf": sig.confidence, "edge": sig.edge, "fee": entry_fee,
        "pnl": pnl, "roi": roi, "sub": (sig.details or {}).get("sub", {}),
    }


def learn_weights(train_trades):
    """
    Reproduce the agent's credit-assignment learning offline: judge each
    sub-strategy by whether its directional call matched the real outcome,
    then turn win-rate + avg-roi into a weight (same formula as SelfLearner).
    """
    from agent.ml.self_learner import STRATEGY_NAMES
    stats = {n: {"wins": 0.0, "total": 0.0, "roi": 0.0} for n in STRATEGY_NAMES}
    for t in train_trades:
        for name, info in (t["sub"] or {}).items():
            if name not in stats:
                continue
            d = info.get("dir")
            if d not in ("YES", "NO"):
                continue
            correct = (d == t["res"])
            stats[name]["wins"]  += int(correct)
            stats[name]["total"] += 1
            stats[name]["roi"]   += t["roi"] if correct else -abs(t["roi"])

    max_w = float(getattr(config, "STRATEGY_MAX_WEIGHT", 4.0))
    weights = dict(DEFAULT_WEIGHTS)
    for name, s in stats.items():
        if s["total"] == 0:
            continue
        wr  = s["wins"] / s["total"]
        roi = s["roi"] / s["total"]
        weights[name] = round(max(0.1, min(max_w, (wr / 0.5) * (1 + roi / 200))), 4)
    return weights, stats


def summarise(title, trades, spread):
    if not trades:
        print(f"\n=== {title} ===\nNo qualifying signals.")
        return
    t = pd.DataFrame(trades)
    n = len(t)
    wins = int(t["won"].sum())
    total = t["pnl"].sum()
    fees  = t["fee"].sum()
    # equity curve & max drawdown
    eq = t["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    # fee-aware breakeven win rate: at fill≈0.51, fee≈1.8% → need a bit over 51%
    fill = 0.5 + spread / 2.0
    avg_fee_frac = fees / (config.MAX_POSITION_SIZE_USD * n)
    breakeven = fill + avg_fee_frac

    print(f"\n=== {title} ===")
    print(f"Windows traded : {n}")
    print(f"Win rate       : {wins}/{n} = {wins/n:.1%}")
    print(f"Breakeven WR   : {breakeven:.1%}  (spread {spread:.3f} + fees)")
    print(f"Edge vs BE     : {wins/n - breakeven:+.1%}   "
          f"{'✅ positive' if wins/n > breakeven else '❌ NEGATIVE — bleeds'}")
    print(f"Total P&L      : ${total:+.2f}   (fees paid: ${fees:.2f})")
    print(f"Avg P&L/trade  : ${t['pnl'].mean():+.3f}")
    print(f"Max drawdown   : ${dd:.2f}")
    print(f"Avg confidence : {t['conf'].mean():.3f}")


def run(hours, decide_min, train_frac, walk_forward):
    df = fetch_1m(hours)
    if df.empty:
        print("No candle data fetched.")
        return
    print(f"Fetched {len(df)} 1m candles "
          f"({df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC)")

    spread = config.PAPER_SPREAD
    start = df.index[0].ceil("15min")
    end   = df.index[-1].floor("15min")

    # enumerate all tradeable windows
    windows = []
    w = start
    while w + timedelta(minutes=15) <= end:
        windows.append(w)
        w += timedelta(minutes=15)
    if not windows:
        print("Not enough data for a single 15-min window.")
        return

    if not walk_forward:
        ens = EnsembleStrategy()              # default weights
        trades = [r for r in (evaluate_window(df, ens, w, decide_min, spread)
                              for w in windows) if r]
        summarise("SINGLE PASS (default weights, no learning)", trades, spread)
        _note(spread)
        return

    # ── Walk-forward: learn on first chunk, test on held-out last chunk ──────
    split = int(len(windows) * train_frac)
    train_w, test_w = windows[:split], windows[split:]
    print(f"Walk-forward split: {len(train_w)} train windows → "
          f"{len(test_w)} test windows (train_frac={train_frac})")

    ens_default = EnsembleStrategy()
    train_trades = [r for r in (evaluate_window(df, ens_default, w, decide_min, spread)
                                for w in train_w) if r]
    if not train_trades:
        print("No signals in training period — nothing to learn from.")
        return

    learned, stats = learn_weights(train_trades)
    print("\nLearned weights (from TRAIN period only):")
    for name in sorted(learned, key=lambda k: -learned[k]):
        s = stats.get(name, {})
        tot = s.get("total", 0)
        wr  = (s["wins"] / tot) if tot else None
        excl = " [EXCLUDED]" if learned[name] < config.STRATEGY_MIN_WEIGHT else ""
        wr_s = f"{wr:.0%}" if wr is not None else "n/a"
        print(f"  {name:13s} w={learned[name]:.3f}  train_wr={wr_s} n={int(tot)}{excl}")

    # report in-sample (train) for contrast, then the honest out-of-sample (test)
    summarise("IN-SAMPLE (train period — optimistic, overfit-prone)",
              train_trades, spread)

    ens_test = EnsembleStrategy(weights=learned)
    test_trades = [r for r in (evaluate_window(df, ens_test, w, decide_min, spread)
                               for w in test_w) if r]
    summarise("OUT-OF-SAMPLE (test period — the number that matters)",
              test_trades, spread)
    _note(spread)


def _note(spread):
    print("\nNote: a ~coin-flip market must clear the fee-aware breakeven win "
          "rate just to break even. The OUT-OF-SAMPLE result is the honest one — "
          "if it's negative or near zero, there is no reliable edge yet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=16)
    ap.add_argument("--decide-min", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="fraction of windows used to LEARN weights (rest is test)")
    ap.add_argument("--no-walk-forward", action="store_true",
                    help="single pass with default weights (no train/test split)")
    args = ap.parse_args()
    run(args.hours, args.decide_min, args.train_frac, not args.no_walk_forward)
