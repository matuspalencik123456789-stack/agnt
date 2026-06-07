"""
Parameter sweep / "poor-man's hyperopt" for the BTC 15-min strategy.

Freqtrade has Hyperopt; this is the same idea adapted to our backtest. It walks
a grid of the SIGNAL parameters the backtest actually exercises, runs the exact
walk-forward backtest for each combination (learn weights on the first chunk,
score on the held-out chunk), and ranks combos by their OUT-OF-SAMPLE result.

Why only these parameters?
  The backtest decides once per window and holds to resolution — it does NOT
  simulate intra-window execution (stop-loss, early-exit confirmations, the
  dead-zone, the signal-stability filter, the post-exit cooldown). So tuning
  those here would be meaningless: every combo would score identically. They
  are execution/risk knobs and must be validated by a forward (live/paper) run,
  not by this backtest. What CAN be honestly optimised is the signal itself:

      decide_min          – how many minutes into the window we commit
      STAT_MARKET_WEIGHT  – how hard the model anchors to the market price
      MIN_EDGE_THRESHOLD  – minimum model-vs-market edge to fire a trade
      TECH_TILT_MAX       – how much the technicals may nudge the model
      STAT_MIN_MISPRICING – minimum mispricing the outcome model needs

The headline deliverable is NOT "a tune that prints money". It's the honest
answer to: *is there ANY setting whose out-of-sample result beats the
fee-aware breakeven?* If the best combo across the whole grid is still negative
or near zero, that is the verdict — a ~coin-flip market with a ~1.8% taker fee
has no reliable edge, and no amount of tuning fixes the market itself.

Usage:
    python optimize.py                       # default grid, 16h history
    python optimize.py --hours 72            # more history (paged requests)
    python optimize.py --quick               # smaller, faster grid
    python optimize.py --min-trades 15       # ignore combos with too few OOS bets
    python optimize.py --objective sharpe     # rank by per-trade Sharpe instead of P&L
    python optimize.py --top 20              # show this many leaders
"""
import argparse
import itertools
import logging
import math

import pandas as pd

import config
from agent.strategies.ensemble import EnsembleStrategy
from backtest import fetch_1m, evaluate_window, learn_weights

logging.basicConfig(level=logging.ERROR)   # silence per-strategy chatter during the sweep


# ── Parameter grids ───────────────────────────────────────────────────────────
# Each key maps to the place it's applied:
#   decide_min            -> passed straight to evaluate_window
#   stat_market_weight    -> the (shared) StatOutcomeStrategy instance attribute
#   min_edge_threshold    -> config.MIN_EDGE_THRESHOLD  (read by the ensemble)
#   tech_tilt_max         -> config.TECH_TILT_MAX       (read by the ensemble)
#   stat_min_mispricing   -> config.STAT_MIN_MISPRICING (read by stat_outcome)
FULL_GRID = {
    "decide_min":          [2, 3, 5],
    "stat_market_weight":  [0.20, 0.35, 0.50],
    "min_edge_threshold":  [0.02, 0.03, 0.05],
    "tech_tilt_max":       [0.30, 0.60],
    "stat_min_mispricing": [0.02, 0.03],
}

QUICK_GRID = {
    "decide_min":          [3],
    "stat_market_weight":  [0.20, 0.35, 0.50],
    "min_edge_threshold":  [0.02, 0.03, 0.05],
    "tech_tilt_max":       [0.30, 0.60],
    "stat_min_mispricing": [0.03],
}


def _apply(combo: dict, stat_strategy):
    """Push one parameter combination into config + the shared model instance."""
    config.MIN_EDGE_THRESHOLD = combo["min_edge_threshold"]
    config.TECH_TILT_MAX      = combo["tech_tilt_max"]
    config.STAT_MIN_MISPRICING = combo["stat_min_mispricing"]
    config.STAT_MARKET_WEIGHT = combo["stat_market_weight"]
    # The stat_outcome strategy caches its market_weight as an instance attr.
    stat_strategy.market_weight = combo["stat_market_weight"]


def _metrics(trades, spread) -> dict:
    """Reduce a list of trade dicts to the numbers we rank on (None if empty)."""
    if not trades:
        return {"n": 0}
    t = pd.DataFrame(trades)
    n = len(t)
    wins = int(t["won"].sum())
    pnl = t["pnl"]
    total = float(pnl.sum())
    avg = float(pnl.mean())
    std = float(pnl.std(ddof=0)) or 1e-9
    sharpe = avg / std                         # per-trade stability ratio (not annualised)
    eq = pnl.cumsum()
    dd = float((eq - eq.cummax()).min())
    avg_fill = float(t["fill"].mean())
    avg_fee_frac = float(t["fee"].sum()) / (config.MAX_POSITION_SIZE_USD * n)
    breakeven = avg_fill + avg_fee_frac
    wr = wins / n
    return {
        "n": n, "win_rate": wr, "breakeven": breakeven, "edge_vs_be": wr - breakeven,
        "total_pnl": total, "avg_pnl": avg, "sharpe": sharpe, "max_dd": dd,
    }


def _walk_forward(df, windows, combo, train_frac, spread):
    """Run the agent's walk-forward exactly as backtest.py does, for one combo."""
    split = int(len(windows) * train_frac)
    train_w, test_w = windows[:split], windows[split:]

    # Train: default-weight ensemble over the train chunk → learned weights.
    ens_train = EnsembleStrategy()
    _apply(combo, ens_train.strategies["stat_outcome"])
    train_trades = [r for r in (evaluate_window(df, ens_train, w, combo["decide_min"], spread)
                                for w in train_w) if r]
    if not train_trades:
        return {"n": 0}, {"n": 0}
    learned, _ = learn_weights(train_trades)

    # Test: held-out chunk with the learned weights — the honest number.
    ens_test = EnsembleStrategy(weights=learned)
    _apply(combo, ens_test.strategies["stat_outcome"])
    test_trades = [r for r in (evaluate_window(df, ens_test, w, combo["decide_min"], spread)
                               for w in test_w) if r]
    return _metrics(train_trades, spread), _metrics(test_trades, spread)


def run(hours, train_frac, grid, min_trades, objective, top):
    df = fetch_1m(hours)
    if df.empty:
        print("No candle data fetched — run on a machine with Binance access.")
        return
    print(f"Fetched {len(df)} 1m candles "
          f"({df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC)")

    spread = config.PAPER_SPREAD
    start = df.index[0].ceil("15min")
    end   = df.index[-1].floor("15min")
    windows = []
    w = start
    while w + pd.Timedelta(minutes=15) <= end:
        windows.append(w)
        w += pd.Timedelta(minutes=15)
    if len(windows) < 8:
        print("Not enough windows for a walk-forward split — fetch more --hours.")
        return

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    print(f"Grid: {len(combos)} combinations × walk-forward "
          f"({int(len(windows)*train_frac)} train / "
          f"{len(windows)-int(len(windows)*train_frac)} test windows)\n")

    results = []
    for i, combo in enumerate(combos, 1):
        ins, oos = _walk_forward(df, windows, combo, train_frac, spread)
        results.append({"combo": combo, "in": ins, "oos": oos})
        o = oos
        line = (f"[{i:>3}/{len(combos)}] "
                f"dm={combo['decide_min']} mw={combo['stat_market_weight']:.2f} "
                f"edge={combo['min_edge_threshold']:.02f} tilt={combo['tech_tilt_max']:.2f} "
                f"mis={combo['stat_min_mispricing']:.02f} → ")
        if o["n"]:
            line += (f"OOS n={o['n']:>3} pnl=${o['total_pnl']:+7.2f} "
                     f"wr={o['win_rate']:.0%} be={o['breakeven']:.0%} "
                     f"edge={o['edge_vs_be']:+.0%} sharpe={o['sharpe']:+.2f}")
        else:
            line += "OOS n=0 (no qualifying trades)"
        print(line)

    # ── Rank the eligible combos ──────────────────────────────────────────────
    key = {"pnl": "total_pnl", "sharpe": "sharpe", "edge": "edge_vs_be"}[objective]
    eligible = [r for r in results if r["oos"]["n"] >= min_trades]
    eligible.sort(key=lambda r: r["oos"][key], reverse=True)

    print("\n" + "=" * 78)
    print(f"TOP {min(top, len(eligible))} BY OUT-OF-SAMPLE {objective.upper()} "
          f"(min {min_trades} OOS trades)")
    print("=" * 78)
    if not eligible:
        print(f"No combo produced ≥ {min_trades} out-of-sample trades. "
              f"Lower --min-trades or fetch more --hours.")
    for rank, r in enumerate(eligible[:top], 1):
        c, o = r["combo"], r["oos"]
        print(f"{rank:>2}. dm={c['decide_min']} mw={c['stat_market_weight']:.2f} "
              f"edge={c['min_edge_threshold']:.02f} tilt={c['tech_tilt_max']:.2f} "
              f"mis={c['stat_min_mispricing']:.02f}  | "
              f"pnl=${o['total_pnl']:+7.2f} wr={o['win_rate']:.0%} "
              f"edge_vs_be={o['edge_vs_be']:+.1%} sharpe={o['sharpe']:+.2f} "
              f"maxDD=${o['max_dd']:.2f} n={o['n']}")

    # ── Honest verdict ────────────────────────────────────────────────────────
    # A positive P&L is NOT enough — on a ~coin-flip a lucky day prints green.
    # We demand the edge be STATISTICALLY SIGNIFICANT: the win-rate's excess over
    # the fee-aware breakeven must exceed ~2 standard errors (≈95% one-sided), AND
    # the equity curve must not be fragile (max drawdown ≤ total profit). Anything
    # short of that is noise dressed up as a result.
    print("\n" + "=" * 78)
    if eligible:
        best = eligible[0]
        b, o = best["combo"], best["oos"]
        wr, n_ = o["win_rate"], o["n"]
        se = math.sqrt(max(wr * (1.0 - wr), 1e-9) / n_)     # std error of the win-rate
        z  = o["edge_vs_be"] / se if se else 0.0            # how many σ above breakeven
        dd_ratio = (abs(o["max_dd"]) / o["total_pnl"]) if o["total_pnl"] > 0 else float("inf")
        Z_MIN, DD_MAX = 2.0, 1.0
        real = (o["total_pnl"] > 0 and z >= Z_MIN and dd_ratio <= DD_MAX)

        print(f"Best combo significance: edge_vs_be={o['edge_vs_be']:+.1%}  "
              f"(±{se:.1%} SE → z={z:+.2f}σ, need ≥{Z_MIN}σ)  "
              f"maxDD/PnL={dd_ratio:.1f}× (need ≤{DD_MAX:.0f}×)")

        if real:
            print("VERDICT: ✅ a STATISTICALLY SIGNIFICANT out-of-sample edge was found.\n"
                  "         Apply these in config.py, then CONFIRM with a live/paper forward\n"
                  "         run before trusting real money:")
            print(f"    DECIDE at minute        : {b['decide_min']}  (backtest --decide-min)")
            print(f"    STAT_MARKET_WEIGHT      = {b['stat_market_weight']}")
            print(f"    MIN_EDGE_THRESHOLD      = {b['min_edge_threshold']}")
            print(f"    TECH_TILT_MAX           = {b['tech_tilt_max']}")
            print(f"    STAT_MIN_MISPRICING     = {b['stat_min_mispricing']}")
            print(f"  → OOS P&L ${o['total_pnl']:+.2f} over {o['n']} trades, "
                  f"edge vs breakeven {o['edge_vs_be']:+.1%} at {z:.1f}σ.")
        else:
            print("VERDICT: ❌ NO robust edge. The best out-of-sample combo is GREEN but it is\n"
                  "         NOT statistically distinguishable from luck:")
            why = []
            if o["total_pnl"] <= 0:      why.append("P&L ≤ 0")
            if z < Z_MIN:                why.append(f"only {z:+.2f}σ above breakeven (need ≥{Z_MIN}σ)")
            if dd_ratio > DD_MAX:        why.append(f"drawdown {dd_ratio:.1f}× the profit (fragile)")
            print(f"  reasons: {', '.join(why)}.")
            print(f"  best: pnl=${o['total_pnl']:+.2f}  edge_vs_be={o['edge_vs_be']:+.1%}  "
                  f"sharpe={o['sharpe']:+.2f}  n={o['n']}")
            print("  A ~coin-flip 15-min binary with a ~1.8% taker fee has no reliable edge at\n"
                  "  these settings. Re-run with much more history (--hours 72+) and the full\n"
                  "  grid; if the result stays sub-2σ, the market itself isn't tradeable here.")
    print("Note: out-of-sample is the only number that matters. In-sample (train) results\n"
          "      are shown per-combo only for contrast and are overfit-prone by construction.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true", help="smaller, faster grid")
    ap.add_argument("--min-trades", type=int, default=10,
                    help="ignore combos with fewer than this many OOS trades")
    ap.add_argument("--objective", choices=["pnl", "sharpe", "edge"], default="pnl")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    run(args.hours, args.train_frac,
        QUICK_GRID if args.quick else FULL_GRID,
        args.min_trades, args.objective, args.top)
