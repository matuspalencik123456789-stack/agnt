"""
Wipe ALL data for a clean live paper run.
Removes demo trades, candles, market snapshots, strategy weights and logs so the
agent starts from zero and only records REAL paper trades against live slugs.

    python scripts/reset_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.database.models import (
    init_db, get_session, Trade, StrategyWeight, BTCCandle, MarketSnapshot, AgentLog
)


def main():
    init_db()
    session = get_session()
    try:
        counts = {
            "trades":          session.query(Trade).delete(),
            "strategy_weights": session.query(StrategyWeight).delete(),
            "candles":         session.query(BTCCandle).delete(),
            "snapshots":       session.query(MarketSnapshot).delete(),
            "logs":            session.query(AgentLog).delete(),
        }
        session.commit()
        print("🧹 Database wiped clean:")
        for k, v in counts.items():
            print(f"   - {k}: {v} rows deleted")
        print("\n✅ Ready for a fresh LIVE paper run. Start with:")
        print("   python main.py --both")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
