"""
Reset finances and anchor P&L to your REAL Polymarket balance.

Wipes all recorded trades / candles / snapshots / logs, then reads your live
USDC balance from Polymarket and stores it as the new BASELINE. From that point
on, "Session P&L" = (current real balance − baseline), so the numbers reflect
what actually happened on-chain rather than the sum of local trade records.

    python scripts/reset_finances.py

Safe to run anytime you want a clean slate (e.g. before a fresh live session).
In paper mode (no private key) the baseline falls back to PAPER_START_BALANCE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.database.models import (
    init_db, get_session, set_baseline,
    Trade, StrategyWeight, BTCCandle, MarketSnapshot, AgentLog,
)
from agent.polymarket_client import PolymarketClient


def main():
    init_db()

    # 1) read the REAL Polymarket balance (live), or paper bankroll as fallback
    poly = PolymarketClient()
    if poly._client:
        balance = poly.get_balance() or 0.0
        source = "live Polymarket USDC balance"
    else:
        balance = float(config.PAPER_START_BALANCE)
        source = "paper start balance (no private key set)"

    # 2) wipe all recorded data for a clean slate
    session = get_session()
    try:
        counts = {
            "trades":           session.query(Trade).delete(),
            "strategy_weights": session.query(StrategyWeight).delete(),
            "candles":          session.query(BTCCandle).delete(),
            "snapshots":        session.query(MarketSnapshot).delete(),
            "logs":             session.query(AgentLog).delete(),
        }
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"❌ Error wiping data: {e}")
        return
    finally:
        session.close()

    # 3) anchor the baseline to the real balance
    set_baseline(balance, note=source)

    print("🧹 Finances reset:")
    for k, v in counts.items():
        print(f"   - {k}: {v} rows deleted")
    print(f"\n💰 Baseline set to ${balance:.2f}  ({source})")
    print("   Session P&L will now be measured from this point.")
    print("\n✅ Start trading with:")
    print("   caffeinate -i python3 main.py --both")


if __name__ == "__main__":
    main()
