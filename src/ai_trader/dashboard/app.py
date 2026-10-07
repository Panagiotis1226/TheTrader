"""Streamlit dashboard (read-only).

Run: streamlit run src/ai_trader/dashboard/app.py
In Docker it listens on 127.0.0.1:8501 only; reach it through an SSH tunnel.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import altair as alt
import streamlit as st

from ai_trader.config import load_env, load_trading_settings
from ai_trader.dashboard.data import (
    account_order,
    decisions_frame,
    equity_frame,
    fills_frame,
    latest_equity,
    llm_cost_by_day,
    rejections_frame,
)
from ai_trader.evaluation import scorecard
from ai_trader.storage.repo import Repository

# Categorical slots in fixed order (validated for color-vision deficiencies); the color
# follows the account, never its rank. Identity is also carried by the legend and tables.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

st.set_page_config(page_title="AI Trader", layout="wide")


def _dec(value: float) -> Decimal:
    return Decimal(str(value))


@st.cache_resource
def _repo(database_url: str) -> Repository:
    return Repository.from_url(database_url)


env = load_env()
settings = load_trading_settings(env.settings_path)
repo = _repo(env.database_url)
preferred = [f"paper-{m.name}" for m in settings.models] + [
    f"paper-{b}" for b in settings.benchmarks
]
accounts = account_order(repo, preferred)
colors = alt.Scale(domain=accounts, range=SERIES_COLORS[: len(accounts)] or SERIES_COLORS[:1])

TIME = st.column_config.DatetimeColumn("time (UTC)", format="YYYY-MM-DD HH:mm")

st.title("AI Trader — paper trading")
st.caption("Read-only. Returns are after fees and slippage. Refresh the page for new data.")

equity = equity_frame(repo)
latest = latest_equity(equity)

if latest.empty:
    st.info("No accounts yet. Run `ai-trader once` or start the bot.")
    st.stop()

now = datetime.now(UTC)
halts = repo.active_halts(None, now)
if halts:
    for h in halts:
        st.error(
            f"TRADING HALTED ({h.kind.value}) since {h.created_at:%Y-%m-%d %H:%M} UTC: "
            f"{h.reason}. Run `ai-trader resume` to lift."
        )
last_decisions = repo.list_decisions(limit=1)
if last_decisions:
    last_at = last_decisions[0]["created_at"]
    age = now - last_at
    status = "Trading active" if not halts else "Halted"
    minutes = int(age.total_seconds() // 60)
    msg = f"{status}. Last decision {last_at:%Y-%m-%d %H:%M} UTC ({minutes} min ago)."
    if age > timedelta(minutes=2 * settings.decision_interval_minutes + 30):
        st.warning(msg + " That's older than expected: is the bot running?")
    else:
        st.caption(msg)

# Headline tiles: one per account.
cols = st.columns(len(latest))
for col, (_, row) in zip(cols, latest.set_index("account").loc[accounts].iterrows(), strict=False):
    delta = f"{row['return_pct']:+.2f}%" if round(row["return_pct"], 2) != 0 else None
    col.metric(row.name, f"{row['equity']:,.2f} CAD", delta)

st.subheader("Return since start (%)")
hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False)
base = alt.Chart(equity).encode(
    x=alt.X("time:T", title=None),
    y=alt.Y("return_pct:Q", title="Return %", axis=alt.Axis(format=".1f")),
    color=alt.Color("account:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
)
lines = base.mark_line(strokeWidth=2)
points = (
    base.mark_point(size=64, filled=True)
    .encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("account:N"),
            alt.Tooltip("time:T", format="%Y-%m-%d %H:%M"),
            alt.Tooltip("equity:Q", format=",.2f", title="equity CAD"),
            alt.Tooltip("return_pct:Q", format="+.2f", title="return %"),
        ],
    )
    .add_params(hover)
)
zero = alt.Chart().mark_rule(color="#a8a7a0", strokeDash=[4, 4]).encode(y=alt.datum(0))
st.altair_chart((zero + lines + points), use_container_width=True)
with st.expander("Table view"):
    st.dataframe(latest, hide_index=True, use_container_width=True)

st.subheader("Phase 5 evaluation")
card = scorecard(
    settings, repo, dict(zip(latest["account"], map(_dec, latest["equity"]), strict=True)), now
)
if card.start is None:
    st.caption("Not started: no answer from the model with the current configuration yet.")
else:
    st.caption(
        f"Since {card.start:%Y-%m-%d} ({card.weeks} weeks), configuration {card.fingerprint}. "
        "Changing the prompt or risk settings restarts the clock."
    )
st.dataframe(
    [{"criterion": c.name, "verdict": c.verdict.value, "detail": c.detail} for c in card.criteria],
    hide_index=True,
    use_container_width=True,
    column_config={"detail": st.column_config.TextColumn(width="large")},
)

decisions = decisions_frame(repo)

left, right = st.columns(2)
with left:
    st.subheader("Trades")
    st.dataframe(
        fills_frame(repo),
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={"time": TIME},
    )
with right:
    st.subheader("Risk rejections")
    st.dataframe(
        rejections_frame(decisions)[
            ["time", "account", "action", "pair", "size_pct", "risk_reason"]
        ],
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={"time": TIME, "risk_reason": st.column_config.TextColumn(width="large")},
    )

st.subheader("Decision log")
st.dataframe(
    decisions[
        [
            "time",
            "account",
            "action",
            "pair",
            "size_pct",
            "confidence",
            "reason",
            "risk",
            "risk_reason",
            "error",
            "model",
            "prompt_hash",
            "snapshot_hash",
        ]
    ],
    hide_index=True,
    use_container_width=True,
    height=420,
    column_config={
        "time": TIME,
        "reason": st.column_config.TextColumn("reason (model)", width="large"),
        "risk_reason": st.column_config.TextColumn(width="medium"),
    },
)

st.subheader("Recent alerts")
alerts = repo.recent_alerts(50)
if alerts:
    st.dataframe(
        [{"time": t, "level": lv, "alert": text} for t, lv, text in alerts],
        hide_index=True,
        use_container_width=True,
        height=260,
        column_config={"time": TIME, "alert": st.column_config.TextColumn(width="large")},
    )
else:
    st.caption("No alerts yet.")

st.subheader("LLM cost per day (USD, API-equivalent for Claude Code)")
cost = llm_cost_by_day(decisions)
if cost.empty:
    st.caption("No LLM calls recorded yet.")
else:
    bars = (
        alt.Chart(cost)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("day:O", title=None),
            y=alt.Y("sum(cost_usd):Q", title="USD"),
            color=alt.Color("account:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
            tooltip=["day:O", "account:N", alt.Tooltip("cost_usd:Q", format="$.4f")],
        )
    )
    st.altair_chart(bars, use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(cost, hide_index=True, use_container_width=True)
