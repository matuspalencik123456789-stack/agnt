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
MAX_POSITION_SIZE_USD    = float(os.getenv("MAX_POSITION_SIZE_USD", "50"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
MIN_EDGE_THRESHOLD       = float(os.getenv("MIN_EDGE_THRESHOLD", "0.04"))   # 4 % min edge
MAX_DAILY_LOSS_USD       = float(os.getenv("MAX_DAILY_LOSS_USD", "200"))
KELLY_FRACTION           = float(os.getenv("KELLY_FRACTION", "0.25"))       # quarter Kelly

# ── Self-learning ────────────────────────────────────────────────────────────
LEARNING_RATE         = float(os.getenv("LEARNING_RATE", "0.05"))
MIN_TRADES_TO_LEARN   = int(os.getenv("MIN_TRADES_TO_LEARN", "20"))
STRATEGY_DECAY        = float(os.getenv("STRATEGY_DECAY", "0.99"))  # older trades matter less

# ── Data ─────────────────────────────────────────────────────────────────────
DB_PATH      = os.getenv("DB_PATH", "data/trading.db")
LOG_PATH     = os.getenv("LOG_PATH", "logs/agent.log")
MODEL_PATH   = os.getenv("MODEL_PATH", "data/model.pkl")

# ── Market data ──────────────────────────────────────────────────────────────
BTC_SYMBOL  = "BTC/USDT"
BINANCE_API = "https://api.binance.com"

# BTC Polymarket market search keywords
BTC_MARKET_KEYWORDS = ["bitcoin", "btc", "BTC"]
