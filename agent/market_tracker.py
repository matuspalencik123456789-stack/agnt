"""
Rolling 15-minute BTC market tracker.

Polymarket runs recurring short-duration BTC markets ("BTC Up or Down" every
15 minutes). This module always locks onto the *current* active 15-min slug and,
the moment it ends/resolves, automatically rolls onto the *next* one — with no
gap. It does NOT hardcode slugs; it discovers them live from the Gamma API by
matching the recurring phrasing + a short start→end duration.
"""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests

import config

log = logging.getLogger(__name__)


def _parse_dt(value) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (with optional trailing Z) to aware UTC."""
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class MarketTracker:
    """Finds and follows the rolling 15-min BTC slug."""

    def __init__(self):
        self._gamma = config.GAMMA_API
        self.active_slug: Optional[str] = None
        self.active_market: Optional[Dict] = None

    # ── Discovery ────────────────────────────────────────────────────────────

    def _is_rolling_btc(self, market: Dict) -> bool:
        """True if this looks like a recurring 15-min BTC market."""
        q = (market.get("question", "") or "").lower()
        slug = (market.get("slug", "") or "").lower()
        text = f"{q} {slug}"

        is_btc = any(k in text for k in ["btc", "bitcoin"])
        if not is_btc:
            return False

        # phrase match OR short duration
        phrase_hit = any(p in text for p in config.ROLLING_MARKET_PHRASES)

        start = _parse_dt(market.get("startDate") or market.get("start_date"))
        end   = _parse_dt(market.get("endDate")   or market.get("end_date"))
        short_duration = False
        if start and end:
            dur = (end - start).total_seconds()
            short_duration = 0 < dur <= config.ROLLING_MAX_DURATION_SEC

        return phrase_hit or short_duration

    def fetch_rolling_markets(self, limit: int = 100) -> List[Dict]:
        """Return active rolling 15-min BTC markets, soonest-ending first."""
        candidates: List[Dict] = []
        for kw in ("bitcoin", "btc"):
            try:
                resp = requests.get(
                    f"{self._gamma}/markets",
                    params={
                        "q": kw,
                        "active": "true",
                        "closed": "false",
                        "limit": limit,
                        "order": "endDate",
                        "ascending": "true",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("markets", [])
                for m in data:
                    if self._is_rolling_btc(m):
                        candidates.append(m)
            except Exception as e:
                log.error(f"fetch_rolling_markets error ({kw}): {e}")

        # de-dupe by slug/conditionId
        seen, unique = set(), []
        for m in candidates:
            key = m.get("slug") or m.get("conditionId") or m.get("id")
            if key and key not in seen:
                seen.add(key)
                unique.append(m)

        now = datetime.now(timezone.utc)
        # keep only markets that haven't ended yet, sort by soonest end
        live = []
        for m in unique:
            end = _parse_dt(m.get("endDate") or m.get("end_date"))
            if end is None or end > now:
                live.append(m)
        live.sort(key=lambda m: _parse_dt(m.get("endDate") or m.get("end_date")) or now)
        return live

    # ── Current / next selection ─────────────────────────────────────────────

    def get_current_market(self) -> Optional[Dict]:
        """
        Return the slug to trade right now: the active market with the nearest
        end time that is still open for trading. Automatically becomes the next
        slug once the previous one ends (because ended markets are filtered out).
        """
        live = self.fetch_rolling_markets()
        if not live:
            log.info("No rolling 15-min BTC market currently available.")
            return None

        now = datetime.now(timezone.utc)
        # prefer a market already started (start <= now < end); else nearest upcoming
        started = []
        upcoming = []
        for m in live:
            start = _parse_dt(m.get("startDate") or m.get("start_date"))
            if start is None or start <= now:
                started.append(m)
            else:
                upcoming.append(m)

        chosen = started[0] if started else upcoming[0]

        new_slug = chosen.get("slug") or chosen.get("conditionId")
        if new_slug != self.active_slug:
            log.info(f"Rolling onto 15-min BTC slug: {new_slug} "
                     f"(ends {chosen.get('endDate')})")
            self.active_slug = new_slug
            self.active_market = chosen
        return chosen

    def seconds_to_end(self, market: Dict) -> Optional[float]:
        end = _parse_dt(market.get("endDate") or market.get("end_date"))
        if end is None:
            return None
        return (end - datetime.now(timezone.utc)).total_seconds()

    def is_tradeable(self, market: Dict) -> bool:
        """False if the market is within the cutoff window of ending."""
        secs = self.seconds_to_end(market)
        if secs is None:
            return True
        return secs > config.ROLL_CUTOFF_SECONDS

    def has_ended(self, market: Dict) -> bool:
        secs = self.seconds_to_end(market)
        return secs is not None and secs <= 0
