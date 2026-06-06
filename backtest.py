"""
Offline backtest for the BTC 15-min "Up or Down" strategy.

Fetches historical 1-minute BTC candles from Binance, walks them window by
window, and on each 15-min window runs the live ensemble exactly as the agent
would — then resolves the bet by comparing the window's close vs its open.
Transaction cost (spread) is applied so the result reflects reality.

Usage:
    python backtest.py                 # default: ~16h of 1m candles
    python backtest.py --hours 48      # more history (multiple requests)
    python backtest.py --decide-min 3  # decide N minutes into each window
"""
import argparse
import logging
from datetime import timedelta

import pandas as pd
import requests

import config
from agent.strategies.ensemble import EnsembleStrategy

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


def run(hours: int, decide_min: int):
    df = fetch_1m(hours)
    if df.empty:
        print("No candle data fetched.")
        return
    print(f"Fetched {len(df)} 1m candles "
          f"({df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC)\n")

    ens = EnsembleStrategy()                      # default weights
    spread = config.PAPER_SPREAD

    # align onto 15-min boundaries
    start = df.index[0].ceil("15min")
    end   = df.index[-1].floor("15min")

    trades = []
    w = start
    while w + timedelta(minutes=15) <= end:
        w_open  = w
        w_close = w + timedelta(minutes=15)
        decide  = w_open + timedelta(minutes=decide_min)

        hist = df[df.index <= decide]
        if len(hist) < 60:                        # need enough lookback
            w += timedelta(minutes=15)
            continue

        try:
            open_px  = float(df[df.index >= w_open]["open"].iloc[0])
            close_px = float(df[df.index < w_close]["close"].iloc[-1])
        except IndexError:
            w += timedelta(minutes=15)
            continue

        meta = {
            "question": "Bitcoin Up or Down?",
            "startDate": w_open.isoformat(),
            "endDate":   w_close.isoformat(),
            "strike_price": 0.0,
        }
        # at the open the market is ~coin-flip; we buy crossing the spread
        yes_mid = no_mid = 0.5
        sig = ens.generate_signal(hist, yes_mid, no_mid, meta)
        if sig.direction == "PASS":
            w += timedelta(minutes=15)
            continue

        fill = 0.5 + spread / 2.0                 # pay the ask
        if sig.edge < max(0.0, sig.confidence - fill):
            pass
        resolution = "YES" if close_px > open_px else "NO"
        won = (sig.direction == resolution)
        shares = config.MAX_POSITION_SIZE_USD / fill
        pnl = (1.0 if won else 0.0) * shares - config.MAX_POSITION_SIZE_USD
        trades.append({"dir": sig.direction, "res": resolution, "won": won,
                       "conf": sig.confidence, "edge": sig.edge, "pnl": pnl})
        w += timedelta(minutes=15)

    if not trades:
        print("No qualifying signals over this period.")
        return

    t = pd.DataFrame(trades)
    n = len(t)
    wins = int(t["won"].sum())
    print(f"Windows traded : {n}")
    print(f"Win rate       : {wins}/{n} = {wins/n:.1%}")
    print(f"Total P&L      : ${t['pnl'].sum():+.2f}  (max ${config.MAX_POSITION_SIZE_USD}/trade)")
    print(f"Avg P&L/trade  : ${t['pnl'].mean():+.3f}")
    print(f"Spread assumed : {spread:.3f}  (breakeven win rate ≈ {(0.5+spread/2):.1%})")
    print(f"Avg confidence : {t['conf'].mean():.3f}")
    print("\nNote: a coin-flip market needs win rate > breakeven just to cover the spread.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=16)
    ap.add_argument("--decide-min", type=int, default=3)
    args = ap.parse_args()
    run(args.hours, args.decide_min)
