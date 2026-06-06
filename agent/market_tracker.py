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
        self.roll_number: int = self._load_last_roll_number()

    def _load_last_roll_number(self) -> int:
        """Resume the slug counter from the DB so it survives restarts."""
        try:
            from agent.database.models import get_session, AgentLog
            s = get_session()
            row = (s.query(AgentLog)
                     .filter(AgentLog.level == "slug_roll")
                     .order_by(AgentLog.id.desc()).first())
            n = int(row.data.get("number", 0)) if row and row.data else 0
            s.close()
            return n
        except Exception:
            return 0

    def _persist_roll(self, number: int, slug: str, market: Dict):
        """Record the active slug + its real Polymarket ids for the dashboard."""
        try:
            from agent.database.models import get_session, AgentLog
            s = get_session()
            s.add(AgentLog(level="slug_roll", message=f"Slug #{number}: {slug}",
                           data={"number": number,
                                 "slug": slug,
                                 "condition_id": market.get("conditionId", ""),
                                 "market_id": str(market.get("id", "")),
                                 "end": str(market.get("endDate") or ""),
                                 "question": market.get("question", "")}))
            s.commit(); s.close()
        except Exception as e:
            log.debug(f"persist_roll error: {e}")

    # ── Discovery ────────────────────────────────────────────────────────────

    def _is_rolling_btc(self, market: Dict) -> bool:
        """True if this looks like a short-duration BTC market worth trading."""
        q = (market.get("question", "") or "").lower()
        slug = (market.get("slug", "") or "").lower()
        text = f"{q} {slug}"

        is_btc = any(k in text for k in ["btc", "bitcoin"])
        if not is_btc:
            return False

        # phrase match
        phrase_hit = any(p in text for p in config.ROLLING_MARKET_PHRASES)
        if phrase_hit:
            return True

        # short duration (start → end within cap)
        start = _parse_dt(market.get("startDate") or market.get("start_date"))
        end   = _parse_dt(market.get("endDate")   or market.get("end_date"))
        if start and end:
            dur = (end - start).total_seconds()
            if 0 < dur <= config.ROLLING_MAX_DURATION_SEC:
                return True

        # fallback: any BTC market ending within ROLLING_NEAR_END_SEC seconds
        now = datetime.now(timezone.utc)
        if end and 0 < (end - now).total_seconds() <= config.ROLLING_NEAR_END_SEC:
            return True

        return False

    # ── Event-based discovery (the real 15-min BTC markets) ───────────────────
    #
    # Polymarket's recurring 15-min BTC markets use a predictable event slug:
    #     btc-updown-15m-<unix_ts>
    # where <unix_ts> is the period START aligned to 15-minute (900s) boundaries.
    # We compute the current boundary and probe a small window of events, so we
    # always lock onto the live slug and auto-roll onto the next one for free.

    SLUG_PREFIX = "btc-updown-15m-"
    PERIOD_SEC = 900  # 15 minutes

    def _event_to_market(self, event: Dict) -> Optional[Dict]:
        """Flatten a Gamma /events item into the market dict the agent expects."""
        markets = event.get("markets") or []
        if not markets:
            return None
        m = markets[0]
        # clobTokenIds often arrives as a JSON-encoded string
        tok = m.get("clobTokenIds")
        if isinstance(tok, str):
            try:
                import json
                tok = json.loads(tok)
            except Exception:
                tok = []
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            try:
                import json
                prices = json.loads(prices)
            except Exception:
                prices = None
        return {
            "slug":         event.get("slug") or m.get("slug"),
            "conditionId":  m.get("conditionId", ""),
            "id":           m.get("id", event.get("id", "")),
            "question":     m.get("question") or event.get("title", "Bitcoin Up or Down?"),
            "clobTokenIds": tok or [],
            "outcomePrices": prices or [0.5, 0.5],
            "startDate":    m.get("startDate") or event.get("startDate"),
            "endDate":      m.get("endDate")   or event.get("endDate"),
            "volume24hr":   m.get("volume24hr", event.get("volume", 0)),
            "liquidity":    m.get("liquidity", event.get("liquidity", 0)),
        }

    def _fetch_event_by_slug(self, slug: str) -> Optional[Dict]:
        try:
            resp = requests.get(f"{self._gamma}/events",
                                params={"slug": slug}, timeout=8)
            if not resp.ok:
                return None
            data = resp.json()
            if isinstance(data, dict):
                data = data.get("events", data.get("data", []))
            if not data:
                return None
            return self._event_to_market(data[0])
        except Exception as e:
            log.debug(f"event fetch {slug} error: {e}")
            return None

    def fetch_rolling_markets(self, limit: int = 200) -> List[Dict]:
        """
        Discover the live 15-min BTC markets via the predictable event-slug
        pattern. Probes the current boundary plus a few neighbours so we catch
        the active slug and pre-fetch the next one.
        """
        import time as _t
        now_ts = int(_t.time())
        base = now_ts - (now_ts % self.PERIOD_SEC)  # current 15m boundary
        # probe previous, current and a couple upcoming boundaries
        offsets = [-self.PERIOD_SEC, 0, self.PERIOD_SEC, 2 * self.PERIOD_SEC]

        live: List[Dict] = []
        seen = set()
        for off in offsets:
            ts = base + off
            slug = f"{self.SLUG_PREFIX}{ts}"
            mkt = self._fetch_event_by_slug(slug)
            if mkt and mkt.get("slug") and mkt["slug"] not in seen:
                seen.add(mkt["slug"])
                live.append(mkt)

        # fall back to text search if the slug pattern returned nothing
        if not live:
            live = self._fetch_rolling_markets_textsearch(limit)

        now = datetime.now(timezone.utc)
        live = [m for m in live
                if (_parse_dt(m.get("endDate")) is None) or _parse_dt(m.get("endDate")) > now]
        live.sort(key=lambda m: _parse_dt(m.get("endDate")) or now)
        return live

    def _fetch_rolling_markets_textsearch(self, limit: int = 200) -> List[Dict]:
        """Legacy fallback: keyword search on /markets."""
        candidates: List[Dict] = []
        for kw in ("bitcoin", "btc"):
            try:
                resp = requests.get(
                    f"{self._gamma}/markets",
                    params={"q": kw, "active": "true", "closed": "false",
                            "limit": limit, "order": "endDate", "ascending": "true"},
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
                log.error(f"textsearch error ({kw}): {e}")
        seen, unique = set(), []
        for m in candidates:
            key = m.get("slug") or m.get("conditionId") or m.get("id")
            if key and key not in seen:
                seen.add(key); unique.append(m)
        return unique

    # ── Current / next selection ─────────────────────────────────────────────

    def get_current_market(self) -> Optional[Dict]:
        """
        Return the slug to trade right now: the active market with the nearest
        end time that is still open for trading. Automatically becomes the next
        slug once the previous one ends (because ended markets are filtered out).
        """
        live = self.fetch_rolling_markets()
        if not live:
            log.warning("No BTC market found — running raw Gamma probe...")
            self._debug_probe()
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
            self.roll_number += 1
            log.info(f"Rolling onto slug #{self.roll_number}: {new_slug} | "
                     f"conditionId={chosen.get('conditionId','')} "
                     f"id={chosen.get('id','')} (ends {chosen.get('endDate')})")
            self.active_slug = new_slug
            self.active_market = chosen
            self._persist_roll(self.roll_number, new_slug, chosen)
        return chosen

    def _debug_probe(self):
        """Log the first few active Gamma markets so we can see what's available."""
        try:
            r = requests.get(f"{self._gamma}/markets",
                             params={"q": "bitcoin", "active": "true", "closed": "false", "limit": "5"},
                             timeout=10)
            data = r.json()
            if isinstance(data, dict):
                data = data.get("markets", [])
            for m in data[:5]:
                log.warning(f"  probe: slug={m.get('slug')} q={m.get('question','')[:60]} "
                            f"start={m.get('startDate')} end={m.get('endDate')}")
        except Exception as e:
            log.warning(f"probe error: {e}")

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
