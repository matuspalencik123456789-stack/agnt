import os
from dotenv import load_dotenv

load_dotenv()

# ── Polymarket ──────────────────────────────────────────────────────────────
POLYMARKET_API_KEY    = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET     = os.getenv("POLYMARKET_SECRET", "")
POLYMARKET_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE", "")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")

CLOB_HOST = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
GAMMA_API  = "https://gamma-api.polymarket.com"

# ── Trading ──────────────────────────────────────────────────────────────────
TRADING_INTERVAL_MINUTES = 15
MAX_POSITION_SIZE_USD    = float(os.getenv("MAX_POSITION_SIZE_USD", "5"))   # max $5 per trade
# Global cap on simultaneously-open positions. Set high: per-window exposure is
# already bounded by MAX_TRADES_PER_SLUG, and positions auto-resolve each window.
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_TRADES_PER_SLUG      = int(os.getenv("MAX_TRADES_PER_SLUG", "3"))   # cap entries per 15-min slug
MIN_EDGE_THRESHOLD       = float(os.getenv("MIN_EDGE_THRESHOLD", "0.01"))   # 1% min edge (was 4%)
# Minimum directional conviction (|P(up)-0.5|) for the outcome model to act.
STAT_MIN_CONVICTION      = float(os.getenv("STAT_MIN_CONVICTION", "0.06"))
STAT_MIN_MISPRICING      = float(os.getenv("STAT_MIN_MISPRICING", "0.03"))
# Minimum professional confluence score (|−1..+1|) for price-action to act.
PRICE_ACTION_MIN_SCORE   = float(os.getenv("PRICE_ACTION_MIN_SCORE", "0.35"))
# How much the outcome model anchors to the market price as a prior (0..1).
# Higher = trust the market more, fight it less. Reduces overconfident bets.
STAT_MARKET_WEIGHT       = float(os.getenv("STAT_MARKET_WEIGHT", "0.35"))

# ── Trade selectivity ("should I bother trading right now?") ──────────────────
# Minimum quality score (confidence × edge) for the FIRST entry on a slug.
ENTRY_QUALITY_MIN        = float(os.getenv("ENTRY_QUALITY_MIN", "0.012"))
# Each additional entry on the same slug must clear a progressively higher bar —
# the agent keeps capacity in reserve and only spends it on a clearly better setup.
ENTRY_QUALITY_ESCALATION = float(os.getenv("ENTRY_QUALITY_ESCALATION", "1.6"))
# Minimum seconds between two entries on the same slug (don't dump all at once).
ENTRY_COOLDOWN_SECONDS   = int(os.getenv("ENTRY_COOLDOWN_SECONDS", "120"))

# ── Warm-up & trend analysis ──────────────────────────────────────────────────
# On startup the agent first OBSERVES and analyses the candle history for this
# long before it's allowed to place its first trade — no blind immediate entry.
WARMUP_SECONDS           = int(os.getenv("WARMUP_SECONDS", "90"))
# Number of recent candles used to determine the short-term trend (the "vývoj").
TREND_LOOKBACK           = int(os.getenv("TREND_LOOKBACK", "30"))
# Minimum |normalised slope| for the trend to count as a real direction (not flat).
TREND_MIN_STRENGTH       = float(os.getenv("TREND_MIN_STRENGTH", "0.00002"))
# Require the trade's side to agree with the detected trend (block counter-trend).
REQUIRE_TREND_AGREEMENT  = os.getenv("REQUIRE_TREND_AGREEMENT", "true").lower() == "true"

# ── Transaction costs (realism) ───────────────────────────────────────────────
# In paper mode we buy at the ASK, not the mid. When a live order book is
# available we use its real ask; otherwise we model an assumed spread. We also
# subtract a fee on the notional. This makes paper P&L match live reality —
# without it a coin-flip strategy looks profitable but bleeds to the spread.
PAPER_SPREAD             = float(os.getenv("PAPER_SPREAD", "0.02"))   # 2¢ assumed bid/ask spread
FEE_RATE                 = float(os.getenv("FEE_RATE", "0.0"))        # legacy flat fee (unused if FEE_ENABLED)

# Polymarket charges a TAKER fee on market orders (our paper orders are takers).
# Crypto is the most expensive category. The real on-chain formula is:
#     fee = shares × feeRate × p × (1 − p)
# which peaks at p=0.50 (max uncertainty) and tapers to ~0 near 0/1. For crypto
# feeRate = 0.072 → max $1.80 per 100 shares (1.8%) at a 50¢ price. Takers pay on
# BOTH buy and sell, so an early sell incurs the fee twice (entry + exit) — we
# model that so paper P&L matches live and we never "win" a trade the fees eat.
FEE_ENABLED              = os.getenv("FEE_ENABLED", "true").lower() == "true"
CRYPTO_FEE_RATE          = float(os.getenv("CRYPTO_FEE_RATE", "0.072"))


def taker_fee(shares: float, price: float) -> float:
    """Polymarket crypto taker fee in USDC for `shares` filled at `price`."""
    if not FEE_ENABLED or shares <= 0 or not (0.0 < price < 1.0):
        return 0.0
    return CRYPTO_FEE_RATE * shares * price * (1.0 - price)

# ── Early exit / position management ("sell mid-trade if something changes") ──
# The agent can close a position BEFORE the window resolves to lock in a profit
# or cut a loss when the market/model moves unexpectedly against it.
ENABLE_EARLY_EXIT        = os.getenv("ENABLE_EARLY_EXIT", "true").lower() == "true"
# Sell to take profit once the held token's bid reaches this price.
TAKE_PROFIT_PRICE        = float(os.getenv("TAKE_PROFIT_PRICE", "0.90"))
# Sell to cut the loss once the held token's bid falls to this price.
STOP_LOSS_PRICE          = float(os.getenv("STOP_LOSS_PRICE", "0.25"))
# Also exit if the model now favours the OPPOSITE side with at least this prob.
EXIT_ON_REVERSAL         = os.getenv("EXIT_ON_REVERSAL", "true").lower() == "true"
MODEL_REVERSAL_PROB      = float(os.getenv("MODEL_REVERSAL_PROB", "0.62"))
# Don't sell on a single price touch — the exit condition must hold for this
# many consecutive roll-checks before we actually close (filters out noise).
EXIT_CONFIRM_COUNT       = int(os.getenv("EXIT_CONFIRM_COUNT", "3"))
# For a stop-loss, also require the MODEL to agree the position is now likely
# losing (its prob for our side below this) — don't panic-sell a temporary dip
# if the model still backs our side.
STOP_LOSS_MODEL_MAXPROB  = float(os.getenv("STOP_LOSS_MODEL_MAXPROB", "0.45"))
MAX_DAILY_LOSS_USD       = float(os.getenv("MAX_DAILY_LOSS_USD", "200"))
KELLY_FRACTION           = float(os.getenv("KELLY_FRACTION", "0.25"))       # quarter Kelly
# Minimum weight below which a strategy is excluded from ensemble voting entirely.
# Strategies that fall below this are clearly harmful — silence them rather than
# letting them drag the ensemble in the wrong direction.
STRATEGY_MIN_WEIGHT      = float(os.getenv("STRATEGY_MIN_WEIGHT", "0.40"))
# Maximum weight a single strategy can reach (raises the ceiling for top performers).
STRATEGY_MAX_WEIGHT      = float(os.getenv("STRATEGY_MAX_WEIGHT", "4.0"))

# ── Self-learning ────────────────────────────────────────────────────────────
LEARNING_RATE         = float(os.getenv("LEARNING_RATE", "0.08"))
MIN_TRADES_TO_LEARN   = int(os.getenv("MIN_TRADES_TO_LEARN", "8"))
STRATEGY_DECAY        = float(os.getenv("STRATEGY_DECAY", "0.99"))  # older trades matter less

# ── Data ─────────────────────────────────────────────────────────────────────
DB_PATH      = os.getenv("DB_PATH", "data/trading.db")
LOG_PATH     = os.getenv("LOG_PATH", "logs/agent.log")
MODEL_PATH   = os.getenv("MODEL_PATH", "data/model.pkl")

# ── Market data ──────────────────────────────────────────────────────────────
BTC_SYMBOL  = "BTC/USDT"
BINANCE_API = "https://api.binance.com"
# Candle timeframe for indicators. Smaller = finer price detail + faster signals.
# Binance native intervals: 1s, 1m, 3m, 5m, ...  We fetch CANDLE_INTERVAL and,
# if CANDLE_RESAMPLE_SEC is set, aggregate into custom-second buckets (e.g. 20s)
# since Binance has no native 20-second candle.
CANDLE_INTERVAL     = os.getenv("CANDLE_INTERVAL", "1s")
CANDLE_LIMIT        = int(os.getenv("CANDLE_LIMIT", "1000"))   # max 1000 for 1s
CANDLE_RESAMPLE_SEC = int(os.getenv("CANDLE_RESAMPLE_SEC", "20"))  # 0 = no resample
# Candle cache TTL (s): protects Binance REST from a fast polling loop. Live
# price stays sub-second via WebSocket, so this only throttles candle refetch.
CANDLE_CACHE_SEC    = int(os.getenv("CANDLE_CACHE_SEC", "5"))

# ── WebSocket live feeds ─────────────────────────────────────────────────────
ENABLE_WEBSOCKET   = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"
BINANCE_WS         = os.getenv("BINANCE_WS", "wss://stream.binance.com:9443/ws")
# Kline interval for the live candle-close trigger (fires re-evaluation).
WS_KLINE_INTERVAL  = os.getenv("WS_KLINE_INTERVAL", "1m")
POLYMARKET_WS      = os.getenv("POLYMARKET_WS", "wss://ws-subscriptions-clob.polymarket.com/ws/market")
WS_RECONNECT_MAX   = int(os.getenv("WS_RECONNECT_MAX", "8"))      # max backoff exponent cap (seconds = 2^n)
WS_STALE_SECONDS   = int(os.getenv("WS_STALE_SECONDS", "30"))     # treat feed as stale after this

# BTC Polymarket market search keywords
BTC_MARKET_KEYWORDS = ["bitcoin", "btc", "BTC"]

# ── 15-min slug rolling ──────────────────────────────────────────────────────
# Phrases that identify the recurring short-duration BTC markets.
ROLLING_MARKET_PHRASES = [
    "up or down", "higher or lower", "15 min", "15-min", "15m",
    "up/down", "price up", "price down", "will btc", "will bitcoin",
    "above", "below", "end of", "by end",
]
# A market counts as a "15-min" market if (endDate - startDate) is within this
# window (seconds). Covers 15-min markets with a little slack.
ROLLING_MAX_DURATION_SEC = int(os.getenv("ROLLING_MAX_DURATION_SEC", "3600"))   # 1-hour slack
# Also accept any BTC market that ends within this many seconds from now.
ROLLING_NEAR_END_SEC = int(os.getenv("ROLLING_NEAR_END_SEC", "14400"))  # 4 hours
# How often (seconds) to check whether the active slug has ended and roll to the next.
ROLL_CHECK_SECONDS = int(os.getenv("ROLL_CHECK_SECONDS", "5"))
# Seconds before end to stop opening new positions on the current slug.
ROLL_CUTOFF_SECONDS = int(os.getenv("ROLL_CUTOFF_SECONDS", "60"))
