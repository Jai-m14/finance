# =============================================================================
# TAIL-RISK INTELLIGENCE DASHBOARD
# Streamlit app
#
# Run: streamlit run app.py
#
# WHAT THIS SHOWS
# ───────────────
# 1. Model-implied physical downside risk (from CR-CWB-EVT paths)
# 2. Market-implied risk-neutral downside probability (from options chain)
# 3. Where and how these two views disagree, with liquidity quality scoring
# 4. Scenario narratives for the worst synthetic paths
#
# WHAT THIS DOES NOT CLAIM
# ─────────────────────────
# This is not an arbitrage engine.
# Disagreements may reflect risk premia, hedging demand, or model error.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from risk_engine import generate_crcwb_evt_paths
from options_engine import (
    get_option_expiries,
    fetch_put_chain,
    clean_put_chain,
    add_bs_metrics,
    vol_surface_summary,
)
from scoring import (
    add_model_physical_probabilities,
    score_model_market_disagreement,
    summarise_dashboard,
)
from config import DEFAULT_TICKER, DEFAULT_HORIZON_DAYS, DEFAULT_N_SIMS

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Tail-Risk Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #1D9E75;
    }
    .warning-card {
        background: #fff8e1;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #EF9F27;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x40?text=Tail-Risk+Intelligence", use_column_width=True)
    st.markdown("---")

    st.header("Parameters")

    ticker = st.text_input("Ticker", DEFAULT_TICKER).upper().strip()

    horizon = st.slider(
        "Forecast horizon (trading days)",
        min_value=5, max_value=60,
        value=DEFAULT_HORIZON_DAYS, step=5,
    )

    n_sims = st.selectbox(
        "Simulation paths",
        [2000, 5000, 10000, 20000],
        index=2,
    )

    st.markdown("---")
    st.subheader("Options filter")

    min_oi = st.slider(
        "Minimum open interest",
        min_value=0, max_value=1000,
        value=100, step=50,
    )
    max_spread = st.slider(
        "Max bid-ask spread (%)",
        min_value=1, max_value=50,
        value=20, step=1,
    )

    st.markdown("---")
    run_button = st.button("▶  Run dashboard", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Tail-Risk Intelligence Dashboard  \n"
        "Compares physical scenario probabilities with options-implied "
        "risk-neutral probabilities.  \n\n"
        "**Not a trading recommendation.**"
    )


# =============================================================================
# HEADER
# =============================================================================

st.title("📉 Tail-Risk Intelligence Dashboard")
st.markdown(
    "Compares **model-implied physical downside risk** (crisis-aware scenario generator) "
    "with **market-implied risk-neutral downside probability** (options chain).  \n"
    "Where they disagree materially, we show the size, direction, and liquidity quality "
    "of that disagreement."
)

st.markdown(
    """<div class="warning-card">
    ⚠️  Physical probability ≠ risk-neutral probability.
    Gaps may reflect risk premia, crash-risk premia, liquidity costs, or model uncertainty — 
    not necessarily mispricing.
    </div>""",
    unsafe_allow_html=True,
)
st.markdown("")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if not run_button:
    st.info("Set parameters in the sidebar and click **Run dashboard** to begin.")
    st.stop()


# ── 1. Generate model paths ──────────────────────────────────────────────────
with st.spinner(f"Generating {n_sims:,} regime-conditioned scenarios for {ticker}..."):
    try:
        risk_out = generate_crcwb_evt_paths(
            ticker=ticker,
            horizon=horizon,
            n_sims=n_sims,
        )
    except Exception as e:
        st.error(f"Risk engine failed: {e}")
        st.stop()

paths         = risk_out["paths"]
stats         = risk_out["stats"]
narratives    = risk_out["narratives"]
lambda_t      = risk_out["lambda_t"]
lambda_details= risk_out["lambda_details"]
evt_fit       = risk_out["evt_fit"]
spot          = risk_out["spot"]
features      = risk_out["features"]
block_info    = risk_out["block_info"]


# =============================================================================
# SECTION 1 — REGIME GATE
# =============================================================================

st.markdown("---")
st.subheader("1 · Regime state")

regime_color = {"calm": "🟢", "fragile": "🟡", "crisis": "🔴"}
regime_emoji = regime_color.get(lambda_details["regime"], "⚪")

col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
col_r1.metric("λ(t) crisis score",  f"{lambda_t:.3f}")
col_r2.metric("Regime",             f"{regime_emoji} {lambda_details['regime'].title()}")
col_r3.metric("Vol percentile",     f"{lambda_details['vol_pct']:.0%}")
col_r4.metric("VIX percentile",     f"{lambda_details['vix_pct']:.0%}")
col_r5.metric("EVT shape (GPD)",    f"{evt_fit['shape']:.3f}")

with st.expander("Regime score components"):
    st.dataframe(pd.DataFrame([lambda_details]), use_container_width=True)


# =============================================================================
# SECTION 2 — MODEL RISK METRICS
# =============================================================================

st.markdown("---")
st.subheader("2 · Model-implied stress risk")
st.caption(f"Based on {n_sims:,} CR-CWB-EVT paths  |  Ticker: {ticker}  |  "
           f"Horizon: {horizon} trading days  |  Spot: {spot:.2f}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("P(>5% drawdown)",  f"{stats.get('prob_drawdown_below_5pct',  np.nan):.1%}")
m2.metric("P(>8% drawdown)",  f"{stats.get('prob_drawdown_below_8pct',  np.nan):.1%}")
m3.metric("P(>10% drawdown)", f"{stats.get('prob_drawdown_below_10pct', np.nan):.1%}")
m4.metric("VaR 5% (20d)",     f"{stats.get('var_5pct_loss',  np.nan):.2%}")
m5.metric("ES 5% (20d)",      f"{stats.get('es_5pct_loss',   np.nan):.2%}")
m6.metric("Median return",    f"{stats.get('median_return',  np.nan):.2%}")

# Return distribution
cum_returns     = paths.sum(axis=1)
terminal_prices = spot * np.exp(cum_returns)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=cum_returns,
    nbinsx=80,
    name="Simulated paths",
    marker_color="#1D9E75",
    opacity=0.75,
))
for level, label, color in [
    (-0.05, "-5%", "#EF9F27"),
    (-0.08, "-8%", "#E24B4A"),
    (-0.10, "-10%", "#7F77DD"),
]:
    fig_dist.add_vline(
        x=level, line_dash="dash",
        line_color=color,
        annotation_text=label,
        annotation_position="top right",
    )
fig_dist.update_layout(
    title=f"Model-implied {horizon}-day cumulative return distribution",
    xaxis_title="Cumulative log return",
    yaxis_title="Count",
    showlegend=False,
    height=320,
    margin=dict(t=40, b=40, l=40, r=20),
)
st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# SECTION 3 — OPTIONS COMPARISON
# =============================================================================

st.markdown("---")
st.subheader("3 · Options-implied comparison")

expiries = get_option_expiries(ticker)

if not expiries:
    st.warning(f"No options data available for {ticker} from yfinance.")
    st.stop()

selected_expiry = st.selectbox(
    "Option expiry date",
    expiries,
    index=min(1, len(expiries) - 1),
    help="Choose the expiry closest to your forecast horizon.",
)

with st.spinner("Fetching options chain and computing implied metrics..."):
    try:
        raw_puts   = fetch_put_chain(ticker, expiry=selected_expiry)
        clean_puts = clean_put_chain(raw_puts)

        # Apply UI filters
        clean_puts = clean_puts[
            (clean_puts["openInterest"].fillna(0) >= min_oi) &
            (clean_puts["bid_ask_pct"].fillna(1) <= max_spread / 100)
        ]

        if len(clean_puts) == 0:
            st.warning("No liquid options remain after applying filters. Relax the filters.")
            st.stop()

        bs_puts    = add_bs_metrics(clean_puts)
        model_puts = add_model_physical_probabilities(bs_puts, paths)
        scored     = score_model_market_disagreement(model_puts)
        summary    = summarise_dashboard(scored)
        vol_summ   = vol_surface_summary(bs_puts)

    except Exception as e:
        st.error(f"Options engine failed: {e}")
        st.stop()

# ── Header metrics ────────────────────────────────────────────────────────────

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Liquid puts analysed",      summary["n_options"])
h2.metric("ATM implied vol",
          f"{vol_summ.get('atm_implied_vol', np.nan):.1%}" if vol_summ.get("atm_implied_vol") else "N/A")
h3.metric("Vol skew (ATM vs 95% strike)",
          f"{vol_summ.get('vol_skew_5pct', np.nan):.2%}" if vol_summ.get("vol_skew_5pct") else "N/A")
h4.metric("Model sees more downside",   summary["n_model_sees_more_downside"])
h5.metric("Options price more downside",summary["n_options_price_more_downside"])


# ── Top disagreement ─────────────────────────────────────────────────────────

st.markdown("#### Top model–market disagreements")

DISPLAY_COLS = [
    "contractSymbol", "strike", "moneyness", "mid", "bid_ask_pct",
    "openInterest", "implied_vol", "physical_prob_ITM",
    "risk_neutral_prob_ITM", "probability_disagreement",
    "physical_expected_payoff", "payoff_minus_mid",
    "tail_disagreement_score", "interpretation", "confidence",
]
display_cols = [c for c in DISPLAY_COLS if c in scored.columns]
display_top  = scored[display_cols].head(30).copy()

# Format
for col in ["moneyness", "mid", "bid_ask_pct", "implied_vol",
            "physical_prob_ITM", "risk_neutral_prob_ITM",
            "probability_disagreement"]:
    if col in display_top.columns:
        display_top[col] = display_top[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )

st.dataframe(display_top, use_container_width=True, height=350)


# ── Probability surface chart ─────────────────────────────────────────────────

st.markdown("#### Physical vs risk-neutral probability by strike")

sorted_df = scored.sort_values("strike")
fig_prob  = go.Figure()

fig_prob.add_trace(go.Scatter(
    x=sorted_df["moneyness"],
    y=sorted_df["physical_prob_ITM"],
    mode="lines+markers",
    name="Physical P (model)",
    line=dict(color="#1D9E75", width=2),
    marker=dict(size=6),
))
fig_prob.add_trace(go.Scatter(
    x=sorted_df["moneyness"],
    y=sorted_df["risk_neutral_prob_ITM"],
    mode="lines+markers",
    name="Risk-neutral P_Q (options)",
    line=dict(color="#7F77DD", width=2, dash="dash"),
    marker=dict(size=6, symbol="square"),
))

# Shade gap
fig_prob.add_trace(go.Scatter(
    x=pd.concat([sorted_df["moneyness"], sorted_df["moneyness"].iloc[::-1]]),
    y=pd.concat([sorted_df["physical_prob_ITM"],
                 sorted_df["risk_neutral_prob_ITM"].iloc[::-1]]),
    fill="toself",
    fillcolor="rgba(29, 158, 117, 0.10)",
    line=dict(color="rgba(255,255,255,0)"),
    showlegend=False,
    name="Disagreement zone",
))

fig_prob.add_vline(x=1.0, line_dash="dot", line_color="#888780",
                   annotation_text="ATM")
fig_prob.update_layout(
    title="P(S_T < K): physical model vs options-implied risk-neutral",
    xaxis_title="Moneyness (K / Spot)",
    yaxis_title="Probability of finishing below strike",
    height=380,
    legend=dict(x=0.02, y=0.98),
    margin=dict(t=50, b=40, l=50, r=20),
)
st.plotly_chart(fig_prob, use_container_width=True)


# ── Disagreement scatter ──────────────────────────────────────────────────────

st.markdown("#### Disagreement by strike and liquidity")

color_map = {
    "MODEL_SEES_MORE_DOWNSIDE_THAN_OPTIONS": "#1D9E75",
    "OPTIONS_PRICE_MORE_DOWNSIDE_THAN_MODEL": "#E24B4A",
    "NO_MATERIAL_DISAGREEMENT": "#B4B2A9",
}

fig_scatter = px.scatter(
    scored,
    x="moneyness",
    y="probability_disagreement",
    size="openInterest",
    color="interpretation",
    color_discrete_map=color_map,
    hover_data={
        "strike": ":.2f",
        "mid": ":.4f",
        "physical_prob_ITM": ":.3f",
        "risk_neutral_prob_ITM": ":.3f",
        "probability_disagreement": ":.3f",
        "confidence": True,
    },
    title="Physical − risk-neutral probability disagreement per strike",
    labels={
        "moneyness": "Moneyness (K/S)",
        "probability_disagreement": "Disagreement (physical − risk-neutral)",
        "openInterest": "Open Interest",
    },
    height=400,
)
fig_scatter.add_hline(y=0, line_dash="solid", line_color="#888780", line_width=1)
fig_scatter.add_hline(y=0.03,  line_dash="dot", line_color="#1D9E75",
                      annotation_text="+3% threshold")
fig_scatter.add_hline(y=-0.03, line_dash="dot", line_color="#E24B4A",
                      annotation_text="-3% threshold")
fig_scatter.update_layout(
    legend=dict(x=0.02, y=0.98),
    margin=dict(t=50, b=40, l=50, r=20),
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ── Vol surface comparison ────────────────────────────────────────────────────

with st.expander("Implied vol surface"):
    fig_vol = go.Figure()
    sorted_bs = bs_puts.dropna(subset=["implied_vol"]).sort_values("strike")
    fig_vol.add_trace(go.Scatter(
        x=sorted_bs["moneyness"],
        y=sorted_bs["implied_vol"],
        mode="lines+markers",
        name="Market implied vol",
        line=dict(color="#7F77DD", width=2),
    ))
    fig_vol.update_layout(
        title="Implied volatility surface",
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Implied vol (annualised)",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# =============================================================================
# SECTION 4 — SCENARIO NARRATIVES
# =============================================================================

st.markdown("---")
st.subheader("4 · Worst-path scenario narratives")
st.caption(
    "The 10 worst simulated paths by cumulative return over the forecast horizon. "
    "Source blocks are anchored to real historical market episodes."
)

if len(narratives) > 0:
    styled_narratives = narratives.copy()

    for col in ["total_return", "max_drawdown", "worst_1d", "worst_5d"]:
        if col in styled_narratives.columns:
            styled_narratives[col] = styled_narratives[col].map(lambda x: f"{x:.2%}")

    st.dataframe(styled_narratives, use_container_width=True)
else:
    st.info("No scenario narratives generated.")


# =============================================================================
# SECTION 5 — REGIME HISTORY CHART
# =============================================================================

with st.expander("Historical regime score λ(t) — last 252 days"):
    st.markdown(
        "Shows how the crisis intensity score has evolved. "
        "Spikes correspond to periods of elevated stress."
    )

    if features is not None and len(features) > 0:
        from scipy.stats import percentileofscore

        lambda_history = []
        eval_dates = features.index[-252::5]

        for dt in eval_dates:
            hist = features.loc[:dt]
            if len(hist) < 50:
                continue
            latest = hist.iloc[-1]

            vol_pct   = percentileofscore(hist["vol_20d"].dropna(),     latest["vol_20d"])    / 100
            vix_pct   = percentileofscore(hist["vix_level"].dropna(),   latest["vix_level"])  / 100
            dd_stress = 1 - percentileofscore(hist["drawdown_20d"].dropna(), latest["drawdown_20d"]) / 100
            ret_stress= 1 - percentileofscore(hist["ret_20d"].dropna(),      latest["ret_20d"])       / 100

            raw   = 0.35*vol_pct + 0.30*vix_pct + 0.20*dd_stress + 0.15*ret_stress
            lam   = 1 / (1 + np.exp(-8 * (raw - 0.60)))
            lambda_history.append({"date": dt, "lambda_t": lam})

        if lambda_history:
            lh_df = pd.DataFrame(lambda_history).set_index("date")
            fig_lh = go.Figure()
            fig_lh.add_trace(go.Scatter(
                x=lh_df.index, y=lh_df["lambda_t"],
                mode="lines", name="λ(t)",
                line=dict(color="#1D9E75", width=2),
                fill="tozeroy",
                fillcolor="rgba(29,158,117,0.10)",
            ))
            for thresh, label, color in [
                (0.25, "Fragile", "#EF9F27"),
                (0.50, "Stress",  "#E24B4A"),
                (0.75, "Crisis",  "#7F77DD"),
            ]:
                fig_lh.add_hline(y=thresh, line_dash="dash",
                                 line_color=color,
                                 annotation_text=label,
                                 annotation_position="left")
            fig_lh.update_layout(
                title="Historical regime score (last 252 trading days)",
                yaxis_title="λ(t)",
                xaxis_title="Date",
                height=280,
                margin=dict(t=40, b=40, l=50, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_lh, use_container_width=True)


# =============================================================================
# SECTION 6 — DOWNLOAD
# =============================================================================

st.markdown("---")
st.subheader("5 · Export")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv_scored = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download disagreement table (CSV)",
        data=csv_scored,
        file_name=f"{ticker}_{selected_expiry}_tailrisk_disagreement.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    csv_narratives = narratives.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download scenario narratives (CSV)",
        data=csv_narratives,
        file_name=f"{ticker}_scenario_narratives.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    **Methodology summary**

    The model uses a **Current-Regime Crisis-Weighted Bootstrap** (CR-CWB) sampler
    to generate synthetic forward paths anchored to the current market regime (measured via λ(t)),
    with **Extreme Value Theory** (GPD) tail correction applied to the deepest synthetic losses.

    Physical probabilities P(S_T < K) are estimated from these paths.
    Risk-neutral probabilities P_Q(S_T < K) are derived from live options prices via
    Black-Scholes implied volatility.

    **Disagreements reflect differences between the physical and risk-neutral measures —
    not model error or tradeable mispricings.**
    Common explanations include the volatility risk premium, crash-risk premium, 
    liquidity premium, and dealer hedging flows.
    """,
    unsafe_allow_html=False,
)
