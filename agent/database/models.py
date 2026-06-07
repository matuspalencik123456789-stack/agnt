"""SQLite schema via SQLAlchemy ORM."""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import event
import config

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime, default=datetime.utcnow, index=True)
    market_id      = Column(String(128), nullable=False, index=True)
    market_slug    = Column(String(256))
    question       = Column(Text)
    side           = Column(String(4))        # YES / NO
    price          = Column(Float)            # entry price (0–1)
    size_usd       = Column(Float)
    shares         = Column(Float)
    order_id       = Column(String(128))
    strategy_used  = Column(String(64))
    signal_data    = Column(JSON)             # raw signals at entry

    # outcome (filled after resolution)
    resolved         = Column(Boolean, default=False)
    resolution       = Column(String(4))        # YES / NO
    exit_price       = Column(Float)
    pnl_usd          = Column(Float)
    roi_pct          = Column(Float)
    closed_at        = Column(DateTime)

    # for paper-mode local resolution: window start/end + btc open price
    window_start     = Column(DateTime)
    window_end       = Column(DateTime)
    btc_open         = Column(Float)            # BTC price at window start


class StrategyWeight(Base):
    __tablename__ = "strategy_weights"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(64), unique=True, nullable=False)
    weight     = Column(Float, default=1.0)
    win_rate   = Column(Float, default=0.5)
    avg_roi    = Column(Float, default=0.0)
    trade_cnt  = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
    market_id    = Column(String(128), index=True)
    yes_price    = Column(Float)
    no_price     = Column(Float)
    volume_24h   = Column(Float)
    liquidity    = Column(Float)
    end_date     = Column(DateTime)
    question     = Column(Text)


class BTCCandle(Base):
    __tablename__ = "btc_candles"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, index=True)
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float)
    volume    = Column(Float)


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level     = Column(String(10))
    message   = Column(Text)
    data      = Column(JSON)


# A SINGLE engine + session factory for the whole process. Previously every
# get_session() built a fresh engine and re-ran create_all + migrations, which
# (with a 5s roll loop + WS thread) hammered SQLite dozens of times a second and
# risked "database is locked". Now we build once, enable WAL for concurrent
# readers/writers, and hand out sessions from one pooled engine.
_ENGINE = None
_SessionFactory = None


def _configure_sqlite(engine):
    """WAL + a busy timeout so concurrent threads don't trip 'database is locked'."""
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        import os
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        _ENGINE = create_engine(
            f"sqlite:///{config.DB_PATH}", echo=False,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )
        _configure_sqlite(_ENGINE)
        Base.metadata.create_all(_ENGINE)
        _auto_migrate(_ENGINE)
        _ensure_indexes(_ENGINE)
    return _ENGINE


# Columns that may be missing from older databases → auto-added on startup.
_MIGRATIONS = {
    "trades": {
        "window_start": "DATETIME",
        "window_end":   "DATETIME",
        "btc_open":     "FLOAT",
    },
}

# Indexes that speed up the hot queries (resolution sweep, per-slug position
# management, loss-streak lookups). CREATE INDEX IF NOT EXISTS is cheap & safe.
_INDEXES = [
    ("ix_trades_resolved",    "trades", "resolved"),
    ("ix_trades_closed_at",   "trades", "closed_at"),
    ("ix_trades_market_slug", "trades", "market_slug"),
]


def _ensure_indexes(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        for name, table, col in _INDEXES:
            try:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({col})"))
            except Exception:
                pass


def _auto_migrate(engine):
    """Add any columns present in the ORM but missing from the live SQLite file."""
    from sqlalchemy import text
    with engine.begin() as conn:
        for table, cols in _MIGRATIONS.items():
            try:
                existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            except Exception:
                continue
            if not existing:
                continue  # table doesn't exist yet → create_all handles it
            for col, sqltype in cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}"))


def get_session():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory()


def init_db():
    get_engine()   # builds engine, runs create_all + migrations + indexes once
