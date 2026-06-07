"""Risk management: position sizing, daily loss limit, Kelly criterion."""
import logging
from datetime import datetime, timedelta

from agent.database.models import get_session, Trade
import config

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.daily_pnl_cache: float = 0.0
        self.last_cache_update: datetime = datetime.utcnow() - timedelta(hours=1)
        self._breaker_until: datetime | None = None   # paused-until timestamp
        # Only trades closed AFTER this moment count toward the loss streak.
        # Initialised to startup so stale pre-restart losses can't deadlock the
        # breaker, and advanced each time the breaker trips so the same streak
        # isn't punished twice once the cooldown expires.
        self._streak_after: datetime = datetime.utcnow()

    def get_daily_pnl(self) -> float:
        if (datetime.utcnow() - self.last_cache_update).seconds < 60:
            return self.daily_pnl_cache
        session = get_session()
        try:
            today = datetime.utcnow().date()
            trades = session.query(Trade).filter(
                Trade.resolved == True,
                Trade.closed_at >= datetime.combine(today, datetime.min.time())
            ).all()
            self.daily_pnl_cache = sum(t.pnl_usd or 0 for t in trades)
            self.last_cache_update = datetime.utcnow()
            return self.daily_pnl_cache
        finally:
            session.close()

    def is_daily_limit_breached(self) -> bool:
        pnl = self.get_daily_pnl()
        if pnl < -config.MAX_DAILY_LOSS_USD:
            log.warning(f"Daily loss limit breached: {pnl:.2f} USD")
            return True
        return False

    def is_circuit_broken(self) -> bool:
        """
        Trip after MAX_CONSECUTIVE_LOSSES losing trades in a row, then stay
        paused for CIRCUIT_BREAKER_COOLDOWN seconds. Protects capital during a
        bad streak (regime change, broken signal) instead of bleeding through it.
        """
        limit = getattr(config, "MAX_CONSECUTIVE_LOSSES", 0)
        if limit <= 0:
            return False
        # still inside an active pause?
        if self._breaker_until and datetime.utcnow() < self._breaker_until:
            return True
        # Cooldown just expired → give the agent a fresh start: only losses from
        # NOW on count toward a new streak (don't re-trip on the old one).
        if self._breaker_until and datetime.utcnow() >= self._breaker_until:
            self._breaker_until = None
            self._streak_after = datetime.utcnow()
            log.info("Circuit breaker cooldown expired — resuming, streak reset.")
            return False
        session = get_session()
        try:
            recent = session.query(Trade).filter(
                Trade.resolved == True, Trade.pnl_usd != None,
                Trade.closed_at >= self._streak_after,
            ).order_by(Trade.closed_at.desc()).limit(limit).all()
        finally:
            session.close()
        if len(recent) < limit:
            return False
        if all((t.pnl_usd or 0) < 0 for t in recent):
            cooldown = getattr(config, "CIRCUIT_BREAKER_COOLDOWN", 900)
            self._breaker_until = datetime.utcnow() + timedelta(seconds=cooldown)
            log.warning(f"Circuit breaker TRIPPED: {limit} losses in a row — "
                        f"pausing entries for {cooldown}s.")
            return True
        return False

    def count_open_positions(self) -> int:
        session = get_session()
        try:
            return session.query(Trade).filter(Trade.resolved == False).count()
        finally:
            session.close()

    def kelly_size(self, win_prob: float, avg_win: float, avg_loss: float,
                   balance: float) -> float:
        """Kelly criterion position size in USD."""
        if avg_loss == 0 or win_prob <= 0 or win_prob >= 1:
            return config.MAX_POSITION_SIZE_USD * 0.1

        b = avg_win / avg_loss          # win/loss ratio
        q = 1 - win_prob
        kelly = (win_prob * b - q) / b  # full Kelly
        kelly = max(0, kelly)

        fraction = kelly * config.KELLY_FRACTION
        size = fraction * balance
        return min(size, config.MAX_POSITION_SIZE_USD)

    def compute_position_size(self, confidence: float, edge: float,
                               yes_price: float, balance: float) -> float:
        """Blend Kelly and fixed sizing, scale by edge & confidence."""
        implied_win = confidence * (1 + edge)

        avg_win  = (1 - yes_price) * 1.0   # rough estimate
        avg_loss = yes_price

        base = self.kelly_size(implied_win, avg_win, avg_loss, balance)

        # confidence multiplier: 0.5–1.0
        conf_mult = 0.5 + 0.5 * min(confidence, 1.0)
        # edge multiplier: 0.5–1.5
        edge_mult = 0.5 + min(edge / 0.10, 1.0)

        size = base * conf_mult * edge_mult
        size = max(1.0, min(size, config.MAX_POSITION_SIZE_USD))
        return round(size, 2)

    def can_trade(self, size_usd: float, balance: float) -> tuple[bool, str]:
        if self.is_daily_limit_breached():
            return False, "daily_loss_limit"
        if self.is_circuit_broken():
            return False, "circuit_breaker (loss streak)"
        if self.count_open_positions() >= config.MAX_CONCURRENT_POSITIONS:
            return False, "max_positions"
        if size_usd > balance * 0.30:
            return False, "size_exceeds_30pct_balance"
        return True, "ok"
