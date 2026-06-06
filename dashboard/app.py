"""Streamlit dashboard for the Polymarket BTC trading agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from agent.database.models import get_session, Trade, MarketSnapshot, BTCCandle, AgentLog
from agent.market_data import fetch_candles, get_current_btc_price
from agent.ml.self_learner import SelfLearner

st.set_page_config(
    page_title="Polymarket BTC Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.stApp { background: #0e1117; }
.metric-card {
    background: #1a1d26; border-radius: 10px;
    padding: 1rem 1.5rem; border: 1px solid #2d3143;
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_trades() -> pd.DataFrame:
    session = get_session()
    try:
        trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(500).all()
        if not trades:
            return pd.DataFrame()
        rows = [{
            "id": t.id, "timestamp": t.timestamp, "question": t.question,
            "side": t.side, "price": t.price, "size_usd": t.size_usd,
            "shares": t.shares, "strategy": t.strategy_used,
            "resolved": t.resolved, "resolution": t.resolution,
            "exit_price": t.exit_price, "pnl_usd": t.pnl_usd, "roi_pct": t.roi_pct,
            "closed_at": t.closed_at,
        } for t in trades]
        return pd.DataFrame(rows)
    finally:
        session.close()


@st.cache_data(ttl=30)
def load_candles() -> pd.DataFrame:
    df = fetch_candles()
    return df


@st.cache_data(ttl=60)
def load_strategy_report() -> pd.DataFrame:
    return SelfLearner().get_strategy_report()


@st.cache_data(ttl=15)
def load_recent_logs() -> pd.DataFrame:
    session = get_session()
    try:
        logs = session.query(AgentLog).order_by(AgentLog.timestamp.desc()).limit(50).all()
        if not logs:
            return pd.DataFrame()
        return pd.DataFrame([{
            "time": l.timestamp.strftime("%H:%M:%S"),
            "level": l.level,
            "message": l.message[:120],
        } for l in logs])
    finally:
        session.close()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Agent Control")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (15s)", value=True)
    if auto_refresh:
        st.caption("Dashboard auto-refreshes every 15 seconds.")

    st.divider()
    st.markdown("**Configuration**")
    import config
    st.caption(f"Interval: `{config.TRADING_INTERVAL_MINUTES}m`")
    st.caption(f"Max position: `${config.MAX_POSITION_SIZE_USD}`")
    st.caption(f"Max daily loss: `${config.MAX_DAILY_LOSS_USD}`")
    st.caption(f"Min edge: `{config.MIN_EDGE_THRESHOLD:.0%}`")
    st.caption(f"Kelly fraction: `{config.KELLY_FRACTION:.0%}`")

    st.divider()
    mode = "🟢 Live" if config.POLYMARKET_PRIVATE_KEY else "🟡 Paper (testing)"
    st.markdown(f"**Mode:** {mode}")

    ws_on = "🟢 ON" if config.ENABLE_WEBSOCKET else "⚪ OFF"
    st.markdown(f"**WebSocket feed:** {ws_on}")
    try:
        from agent.websocket_feed import LIVE
        _wsp = LIVE.get_btc_price()
        st.caption(f"Live BTC (WS): ${_wsp:,.0f}" if _wsp else "Live BTC (WS): waiting…")
    except Exception:
        pass

    st.divider()
    st.markdown("**🔄 Active 15-min slug**")
    try:
        from agent.database.models import get_session as _gs, AgentLog as _AL
        _s = _gs()
        _row = (_s.query(_AL).filter(_AL.level == "slug_roll")
                  .order_by(_AL.id.desc()).first())
        _data = dict(_row.data) if (_row and _row.data) else {}
        _total = _s.query(_AL).filter(_AL.level == "slug_roll").count()
        _s.close()
        if _data:
            st.metric("Slug #", _data.get("number", "—"))
            st.caption("**Polymarket slug:**")
            st.code(_data.get("slug", ""), language=None)
            if _data.get("condition_id"):
                st.caption("**conditionId:**")
                st.code(_data.get("condition_id", ""), language=None)
            if _data.get("market_id"):
                st.caption(f"market id: `{_data.get('market_id','')}`")
            if _data.get("end"):
                st.caption(f"ends: {_data['end']}")
            st.caption(f"Total slugs traded: {_total}")
        else:
            st.caption("No slug yet — start `python main.py --both`.")
    except Exception as e:
        st.caption(f"tracker error: {e}")


# ── Main layout ──────────────────────────────────────────────────────────────
st.title("📈 Polymarket BTC 15-min Trading Agent")
st.caption(f"Last loaded: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

trades_df = load_trades()
candles   = load_candles()
btc_price = get_current_btc_price()

# Warn if leftover DEMO data is present (real runs use real slug ids)
if not trades_df.empty and trades_df["question"].astype(str).str.contains("demo", case=False).any():
    st.warning("⚠️ Demo data detected. Run `python scripts/reset_db.py` to wipe it "
               "and start a clean LIVE paper run.")
elif trades_df.empty:
    st.info("🟢 Clean slate — LIVE paper mode. Start the agent with "
            "`python main.py --both`. Real trades will appear here as the agent "
            "trades live Polymarket 15-min BTC slugs.")

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

resolved = trades_df[trades_df["resolved"] == True] if not trades_df.empty else pd.DataFrame()
total_pnl    = resolved["pnl_usd"].sum()   if not resolved.empty else 0
wins         = (resolved["pnl_usd"] > 0).sum() if not resolved.empty else 0
total_trades = len(resolved)
win_rate     = wins / total_trades if total_trades else 0
open_trades  = (trades_df["resolved"] == False).sum() if not trades_df.empty else 0
avg_roi      = resolved["roi_pct"].mean() if not resolved.empty else 0

c1.metric("BTC Price",    f"${btc_price:,.0f}" if btc_price else "N/A")
c2.metric("Total P&L",    f"${total_pnl:+.2f}",
          delta_color="normal" if total_pnl >= 0 else "inverse")
c3.metric("Win Rate",     f"{win_rate:.1%}")
c4.metric("Trades",       f"{total_trades}  ({open_trades} open)")
c5.metric("Avg ROI",      f"{avg_roi:.1f}%")
if not resolved.empty and "closed_at" in resolved.columns:
    today = datetime.utcnow().date()
    mask = pd.to_datetime(resolved["closed_at"]).dt.date == today
    today_val = resolved.loc[mask, "pnl_usd"].sum() if mask.any() else 0.0
else:
    today_val = 0.0
c6.metric("Today P&L", f"${today_val:+.2f}")

# ── Row 1: BTC chart + Equity curve ──────────────────────────────────────────
st.divider()
col_btc, col_eq = st.columns([3, 2])

with col_btc:
    st.subheader("BTC/USDT — 15m Candles")
    if not candles.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=candles.index,
            open=candles["open"], high=candles["high"],
            low=candles["low"],   close=candles["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )])
        fig.update_layout(
            template="plotly_dark", height=380,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1f2235"),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Waiting for candle data...")

with col_eq:
    st.subheader("Equity Curve")
    if not resolved.empty and "closed_at" in resolved.columns:
        eq = resolved.sort_values("closed_at").copy()
        eq["cum_pnl"] = eq["pnl_usd"].cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eq["closed_at"], y=eq["cum_pnl"],
            fill="tozeroy",
            line=dict(color="#26a69a" if total_pnl >= 0 else "#ef5350", width=2),
        ))
        fig2.update_layout(
            template="plotly_dark", height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Cumulative P&L (USD)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1f2235"),
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No resolved trades yet.")

# ── Row 2: Strategy weights + P&L distribution ───────────────────────────────
st.divider()
col_s, col_d = st.columns([2, 2])

with col_s:
    st.subheader("🧠 Strategy Weights (Self-Learning)")
    strat_df = load_strategy_report()
    if not strat_df.empty:
        fig3 = px.bar(
            strat_df, x="Strategy", y="Weight",
            color="Weight", color_continuous_scale="RdYlGn",
            text="Win Rate",
        )
        fig3.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, width='stretch')
        st.dataframe(strat_df, width='stretch', hide_index=True)
    else:
        st.info("Weights update after first learning cycle.")

with col_d:
    st.subheader("P&L Distribution")
    if not resolved.empty:
        fig4 = px.histogram(
            resolved, x="pnl_usd", nbins=30,
            color_discrete_sequence=["#7c4dff"],
            labels={"pnl_usd": "P&L (USD)"},
        )
        fig4.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig4.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, width='stretch')

        col_a, col_b = st.columns(2)
        col_a.metric("Best Trade",  f"${resolved['pnl_usd'].max():.2f}")
        col_b.metric("Worst Trade", f"${resolved['pnl_usd'].min():.2f}")
    else:
        st.info("No resolved trades yet.")

# ── Row 3: Trade log ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Trade History")

if not trades_df.empty:
    display_df = trades_df.copy()
    display_df["time"]    = pd.to_datetime(display_df["timestamp"]).dt.strftime("%H:%M:%S")
    display_df["market"]  = display_df["question"].str[:55]
    display_df["stake"]   = display_df["size_usd"].map("${:.2f}".format)
    display_df["entry"]   = display_df["price"].map("{:.3f}".format)

    def _outcome(row):
        if not row["resolved"]:
            return "⏳ open"
        won = row["resolution"] == row["side"]
        return "✅ WIN" if won else "❌ LOSS"

    def _pnl(row):
        if pd.isna(row["pnl_usd"]):
            return "—"
        return f"${row['pnl_usd']:+.4f}"

    def _roi(row):
        if pd.isna(row["roi_pct"]):
            return "—"
        arrow = "▲" if row["roi_pct"] > 0 else ("▼" if row["roi_pct"] < 0 else "=")
        return f"{arrow} {abs(row['roi_pct']):.1f}%"

    display_df["outcome"] = display_df.apply(_outcome, axis=1)
    display_df["P&L"]     = display_df.apply(_pnl,     axis=1)
    display_df["ROI"]     = display_df.apply(_roi,     axis=1)

    show_cols = ["time", "side", "market", "entry", "stake", "outcome", "P&L", "ROI"]

    def row_color(row):
        if "WIN"  in str(row["outcome"]): return ["background-color: #0d3323"] * len(row)
        if "LOSS" in str(row["outcome"]): return ["background-color: #3b0d0d"] * len(row)
        return ["background-color: #1a1d26"] * len(row)

    st.dataframe(
        display_df[show_cols].style.apply(row_color, axis=1),
        width='stretch', height=380,
        hide_index=True,
    )

    # summary counts below the table
    if not resolved.empty:
        n_win  = int((resolved["pnl_usd"] > 0).sum())
        n_loss = int((resolved["pnl_usd"] <= 0).sum())
        st.caption(
            f"Resolved: **{len(resolved)}** total — "
            f"✅ {n_win} wins  ❌ {n_loss} losses  |  "
            f"Best: **${resolved['pnl_usd'].max():+.4f}**  "
            f"Worst: **${resolved['pnl_usd'].min():+.4f}**"
        )
else:
    st.info("No trades yet. Agent will start trading on first cycle.")

# ── Row 4: Open positions + Agent logs ───────────────────────────────────────
st.divider()
col_op, col_log = st.columns([2, 2])

with col_op:
    st.subheader("🔓 Open Positions")
    if not trades_df.empty:
        open_df = trades_df[trades_df["resolved"] == False].copy()
        if not open_df.empty:
            open_df["btc_now"] = btc_price
            open_df["unrealized"] = open_df.apply(
                lambda r: round((1.0 - r["price"]) * r["shares"], 2), axis=1
            )
            show = open_df[["timestamp", "question", "side", "price", "size_usd", "unrealized"]].copy()
            show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime("%m-%d %H:%M")
            show["question"] = show["question"].str[:60]

            def _pos_color(row):
                c = "#0d3323" if row["unrealized"] >= 0 else "#3b0d0d"
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                show.style.apply(_pos_color, axis=1),
                width='stretch', hide_index=True,
            )
        else:
            st.info("No open positions.")

with col_log:
    st.subheader("🖥️ Agent Logs")
    logs_df = load_recent_logs()
    if not logs_df.empty:
        st.dataframe(logs_df, width='stretch', height=250, hide_index=True)
    else:
        st.info("No logs yet.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(15)
    st.rerun()
