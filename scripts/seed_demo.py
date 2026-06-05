"""
Seed the database with realistic PAPER-MODE demo data so the dashboard shows
populated charts immediately. Safe to run anytime — it only writes demo rows.

    python scripts/seed_demo.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from agent.database.models import (
    init_db, get_session, Trade, StrategyWeight, BTCCandle, AgentLog
)

STRATS = ["rsi", "macd", "bollinger", "momentum", "vwap", "adx"]
random.seed(7)


def seed_candles(session):
    """Synthetic 15m BTC candles (random walk around $68k)."""
    now = datetime.utcnow().replace(second=0, microsecond=0)
    price = 68000.0
    for i in range(200, 0, -1):
        ts = now - timedelta(minutes=15 * i)
        if session.query(BTCCandle).filter_by(timestamp=ts).first():
            continue
        drift = random.gauss(0, 120)
        o = price
        c = max(1000, price + drift)
        h = max(o, c) + abs(random.gauss(0, 40))
        l = min(o, c) - abs(random.gauss(0, 40))
        v = abs(random.gauss(500, 150))
        session.add(BTCCandle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
        price = c
    session.commit()


def seed_trades(session):
    now = datetime.utcnow()
    questions = [
        "Will BTC be above $68,000 in the next 15 min?",
        "Will BTC be below $67,500 in the next 15 min?",
        "Will BTC be above $68,500 in the next 15 min?",
        "Will BTC be above $69,000 in the next 15 min?",
        "Will BTC be below $67,000 in the next 15 min?",
    ]
    for i in range(60):
        opened = now - timedelta(minutes=15 * (60 - i))
        side = random.choice(["YES", "NO"])
        price = round(random.uniform(0.35, 0.65), 3)
        size = round(random.uniform(5, 40), 2)
        shares = round(size / price, 4)
        # 56% win rate, winners weighted to better strategies
        strat = random.choice(STRATS)
        win = random.random() < (0.62 if strat in ("macd", "adx") else 0.5)
        resolved = i < 54  # leave a few open
        trade = Trade(
            timestamp=opened,
            market_id=f"0xdemo{i:03d}",
            market_slug=f"bitcoin-up-or-down-demo-{i}",
            question=random.choice(questions),
            side=side, price=price, size_usd=size, shares=shares,
            order_id=f"PAPER_{i}", strategy_used="ensemble",
            signal_data={"sub": {strat: {"dir": side, "conf": round(random.uniform(.55,.8),3),
                                         "edge": round(random.uniform(.04,.15),3)}},
                         "yes_score": round(random.uniform(0,1),3),
                         "no_score": round(random.uniform(0,1),3)},
            resolved=resolved,
        )
        if resolved:
            resolution = side if win else ("NO" if side == "YES" else "YES")
            exit_p = 1.0 if resolution == side else 0.0
            trade.resolution = resolution
            trade.exit_price = exit_p
            trade.pnl_usd = round((exit_p - price) * shares, 4)
            trade.roi_pct = round((exit_p / price - 1) * 100, 2)
            trade.closed_at = opened + timedelta(minutes=15)
        session.add(trade)
    session.commit()


def seed_weights(session):
    base = {"rsi": 0.92, "macd": 1.45, "bollinger": 0.78,
            "momentum": 1.10, "vwap": 0.85, "adx": 1.55}
    for name in STRATS:
        row = session.query(StrategyWeight).filter_by(name=name).first() or StrategyWeight(name=name)
        row.weight = base[name]
        row.win_rate = round(random.uniform(0.45, 0.66), 4)
        row.avg_roi = round(random.uniform(-5, 22), 4)
        row.trade_cnt = random.randint(20, 60)
        row.updated_at = datetime.utcnow()
        session.add(row)
    session.commit()


def seed_logs(session):
    msgs = [
        ("cycle_start", "Cycle #42 started"),
        ("info", "Rolling onto 15-min BTC slug: bitcoin-up-or-down-demo-60"),
        ("info", "[ws] connected — streaming BTC price + 15m klines"),
        ("info", "TRADE -> YES | Will BTC be above $68,000... conf=64% edge=0.071"),
        ("info", "[roll] New active slug detected (893s to end)"),
    ]
    for lvl, m in msgs:
        session.add(AgentLog(level=lvl, message=m, data={}))
    session.commit()


def main():
    init_db()
    session = get_session()
    try:
        seed_candles(session)
        seed_trades(session)
        seed_weights(session)
        seed_logs(session)
        print("✅ Demo paper-mode data seeded. Refresh the dashboard.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
