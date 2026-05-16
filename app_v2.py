# =============================================================================
# TAILGUARD DASHBOARD v2
# Three-tab production interface:
#
#   Tab 1: Market Stress
#          Learned gate stress probability + scenario distribution
#
#   Tab 2: Model Health
#          Four-layer validation status — tells you when NOT to trust
#
#   Tab 3: Options Disagreement
#          Physical vs risk-neutral probability gap by strike
#
# Run: streamlit run app_v2.py
# =============================================================================

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

from risk_engine import generate_crcwb_evt_paths, compute_lambda
from options_engine import (
    get_option_expiries, fetch_put_chain, clean_put_chain,
    add_bs_metrics, vol_surface_summary,
)
from scoring import (
    add_model_physical_probabilities,
    score_model_market_disagreement,
    summarise_dashboard,
)
from metrics.learned_gate import predict_current_stress
from metrics.model_health  import compute_model_health, health_report_to_dataframe
from metrics.var_backtest  import var_es_backtest_panel, pathwise_var_stability
from metrics.body_realism  import compute_body_realism
from metrics.crisis_coverage import crisis_coverage_report
from config import DEFAULT_TICKER, DEFAULT_HORIZON_DAYS


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TailGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .health-pass    { color: #1D9E75; font-weight: 500; }
  .health-warning { color: #EF9F27; font-weight: 500; }
  .health-fail    { color: #E24B4A; font-weight: 500; }
  .health-unknown { color: #888780; font-weight: 500; }
  .big-metric     { font-size: 2rem; font-weight: 500; }
  .warn-box {
    background: #fff8e1; border-left: 3px solid #EF9F27;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.85rem; margin: 8px 0;
  }
  .good-box {
    background: #f0faf6; border-left: 3px solid #1D9E75;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.85rem; margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🛡️ TailGuard")
    st.markdown("*Tail-Risk Intelligence*")
    st.divider()

    ticker = st.text_input("Ticker", DEFAULT_TICKER).upper().strip()

    horizon = st.slider(
        "Forecast horizon (trading days)",
        5, 60, DEFAULT_HORIZON_DAYS, step=5
    )

    n_sims = st.selectbox(
        "Simulation paths",
        [2000, 5000, 10000, 20000], index=2
    )

    run_validation = st.checkbox(
        "Run model health validation",
        value=False,
        help="Adds ~60s. Runs VaR backtest, crisis coverage, pathwise stability."
    )

    run_gate = st.checkbox(
        "Calibrate learned gate",
        value=False,
        help="Adds ~30s. Trains supervised stress probability model on full history."
    )

    st.divider()
    go_button = st.button("▶  Run TailGuard", use_container_width=True)

    st.divider()
    st.caption(
        "Physical probability ≠ risk-neutral probability.  \n"
        "Gaps may reflect risk premia — not mispricing.  \n\n"
        "**Not a trading recommendation.**"
    )


# =============================================================================
# LANDING STATE
# =============================================================================

if not go_button:
    st.title("🛡️ TailGuard Tail-Risk Intelligence")
    st.markdown(
        "A **regime-aware stress monitoring platform** that compares "
        "model-implied physical downside risk with options-implied risk-neutral probabilities, "
        "and tells you **how much to trust the model**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Tab 1: Market Stress")
        st.markdown(
            "Learned gate stress probabilities, drawdown risk metrics, "
            "scenario distribution from CR-CWB-EVT paths."
        )
    with col2:
        st.markdown("#### Tab 2: Model Health")
        st.markdown(
            "Four-layer validation: body realism, VaR/ES backtesting, "
            "crisis coverage, pathwise stability. Identifies false-safety patterns."
        )
    with col3:
        st.markdown("#### Tab 3: Options Disagreement")
        st.markdown(
            "Physical P(S_T < K) vs risk-neutral P_Q(S_T < K) per strike. "
            "Gap scored by liquidity quality."
        )

    st.info("Set parameters in the sidebar and click **Run TailGuard**.")
    st.stop()


# =============================================================================
# DATA GENERATION
# =============================================================================

with st.spinner(f"Generating {n_sims:,} regime-conditioned scenarios for {ticker}..."):
    try:
        risk_out = generate_crcwb_evt_paths(ticker=ticker, horizon=horizon, n_sims=n_sims)
    except Exception as e:
        st.error(f"Risk engine failed: {e}")
        st.stop()

paths         = risk_out["paths"]
stats         = risk_out["stats"]
lambda_t      = risk_out["lambda_t"]
lambda_details= risk_out["lambda_details"]
features      = risk_out["features"]
returns       = risk_out["returns"]
spot          = risk_out["spot"]
evt_fit       = risk_out["evt_fit"]
narratives    = risk_out["narratives"]
block_info    = risk_out["block_info"]


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Market Stress", "🏥 Model Health", "📉 Options Disagreement"])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — MARKET STRESS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab1:
    st.subheader(f"Market Stress Outlook — {ticker}")

    # ── Regime gate ───────────────────────────────────────────────────────────
    regime_emoji = {"calm": "🟢", "fragile": "🟡", "crisis": "🔴"}.get(
        lambda_details["regime"], "⚪"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("λ(t) crisis score",   f"{lambda_t:.3f}")
    c2.metric("Regime",              f"{regime_emoji} {lambda_details['regime'].title()}")
    c3.metric("P(>5% drawdown)",     f"{stats.get('prob_drawdown_below_5pct', np.nan):.1%}")
    c4.metric("P(>8% drawdown)",     f"{stats.get('prob_drawdown_below_8pct', np.nan):.1%}")
    c5.metric("VaR 5% (20d)",        f"{stats.get('var_5pct_loss', np.nan):.2%}")

    # ── Learned gate ──────────────────────────────────────────────────────────
    if run_gate:
        with st.spinner("Calibrating learned gate (training supervised model on full history)..."):
            try:
                spy_returns = returns[ticker].dropna() if ticker in returns.columns else returns.iloc[:, 0].dropna()
                gate_result = predict_current_stress(
                    spy_returns, features, target_col="stress_20d"
                )
            except Exception as e:
                gate_result = {"error": str(e), "p_stress": np.nan, "confidence": "UNKNOWN"}

        if "error" not in gate_result:
            st.markdown("---")
            st.markdown("#### Learned gate — calibrated stress probability")

            g1, g2, g3 = st.columns(3)
            g1.metric("Ensemble P(stress 20d)", f"{gate_result.get('p_stress', np.nan):.1%}")
            g2.metric("Confidence",              gate_result.get("confidence", "UNKNOWN"))
            g3.metric("Model range",             f"±{gate_result.get('model_range', 0):.1%}")

            # Individual model probabilities
            ind = gate_result.get("individual_models", {})
            if ind:
                ind_df = pd.DataFrame([
                    {"model": k, "p_stress": v}
                    for k, v in ind.items() if not np.isnan(v)
                ])
                fig_ind = px.bar(
                    ind_df, x="model", y="p_stress",
                    title="Stress probability by gate model",
                    color_discrete_sequence=["#1D9E75"],
                    labels={"p_stress": "P(stress 20d)", "model": "Model"},
                )
                fig_ind.add_hline(y=0.15, line_dash="dash", line_color="#EF9F27",
                                  annotation_text="Alert threshold 15%")
                fig_ind.update_layout(height=280, margin=dict(t=40, b=30))
                st.plotly_chart(fig_ind, use_container_width=True)

            # Top drivers
            drivers = gate_result.get("top_drivers", [])
            if drivers:
                st.markdown("**Top stress drivers (feature importance)**")
                st.dataframe(pd.DataFrame(drivers), use_container_width=True, height=180)

            st.markdown(
                """<div class="warn-box">
                Learned gate outputs are calibrated historical probabilities, not guaranteed forecasts.
                Confidence reflects agreement across model variants — not certainty about the future.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Learned gate failed: {gate_result.get('error')}")

    # ── Return distribution ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Simulated return distribution")

    cum_returns = paths.sum(axis=1)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=cum_returns, nbinsx=80, name="Simulated paths",
        marker_color="#1D9E75", opacity=0.75
    ))
    for level, label, color in [(-0.05,"-5%","#EF9F27"),(-0.08,"-8%","#E24B4A"),(-0.10,"-10%","#7F77DD")]:
        fig_dist.add_vline(x=level, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_position="top right")
    fig_dist.update_layout(
        title=f"CR-CWB-EVT {horizon}-day return distribution ({n_sims:,} paths)",
        xaxis_title="Cumulative log return", yaxis_title="Count",
        height=320, margin=dict(t=50, b=40, l=40, r=20), showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Scenario narratives ───────────────────────────────────────────────────
    with st.expander("Worst-path scenario narratives"):
        if len(narratives) > 0:
            disp = narratives.copy()
            for col in ["total_return", "max_drawdown", "worst_1d", "worst_5d"]:
                if col in disp.columns:
                    disp[col] = disp[col].map(lambda x: f"{x:.2%}")
            st.dataframe(disp, use_container_width=True)

    # ── λ history ────────────────────────────────────────────────────────────
    with st.expander("Historical λ(t) — last 252 days"):
        from scipy.stats import percentileofscore as _pcts
        lh = []
        for dt in features.index[-252::5]:
            hf = features.loc[:dt]
            if len(hf) < 50:
                continue
            lat = hf.iloc[-1]
            vp  = _pcts(hf["vol_20d"].dropna(),    lat["vol_20d"])   / 100
            xp  = _pcts(hf["vix_level"].dropna(),  lat["vix_level"]) / 100
            ds  = 1 - _pcts(hf["drawdown_20d"].dropna(), lat["drawdown_20d"]) / 100
            rs  = 1 - _pcts(hf["ret_20d"].dropna(),      lat["ret_20d"])       / 100
            raw = 0.35*vp + 0.30*xp + 0.20*ds + 0.15*rs
            lam = 1 / (1 + np.exp(-8 * (raw - 0.60)))
            lh.append({"date": dt, "lambda_t": lam})

        if lh:
            lh_df = pd.DataFrame(lh).set_index("date")
            fig_lh = go.Figure()
            fig_lh.add_trace(go.Scatter(
                x=lh_df.index, y=lh_df["lambda_t"],
                mode="lines", line=dict(color="#1D9E75", width=2),
                fill="tozeroy", fillcolor="rgba(29,158,117,0.10)"
            ))
            for th, lab, col in [(0.25,"Fragile","#EF9F27"),(0.50,"Stress","#E24B4A"),(0.75,"Crisis","#7F77DD")]:
                fig_lh.add_hline(y=th, line_dash="dash", line_color=col,
                                 annotation_text=lab, annotation_position="left")
            fig_lh.update_layout(
                title="Historical regime score λ(t)",
                yaxis_title="λ(t)", xaxis_title="Date",
                height=250, margin=dict(t=40,b=30,l=40,r=20), showlegend=False
            )
            st.plotly_chart(fig_lh, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL HEALTH                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab2:
    st.subheader("Model Health Report")
    st.caption(
        "The model is valuable only if it tells you when not to trust it. "
        "This tab runs all four validation layers."
    )

    if not run_validation:
        st.info(
            "Enable **Run model health validation** in the sidebar to see the full report. "
            "Adds ~60 seconds. Results include VaR backtest, crisis coverage, and pathwise stability."
        )
        st.stop()

    spy_returns = (
        returns[ticker].dropna()
        if ticker in returns.columns
        else returns.iloc[:, 0].dropna()
    )
    n_total  = len(spy_returns)
    n_train  = int(n_total * 0.80)
    r_train  = spy_returns.values[:n_train]
    r_test   = spy_returns.values[n_train:]

    with st.spinner("Running body realism check..."):
        body_row = compute_body_realism(r_test, paths, model_name="CR-CWB-EVT")

    with st.spinner("Running VaR/ES backtest..."):
        var_df = var_es_backtest_panel(r_test, paths, model_name="CR-CWB-EVT")
        stab   = pathwise_var_stability(r_test, paths, model_name="CR-CWB-EVT", alpha=0.05)

    with st.spinner("Running crisis coverage analysis..."):
        crisis = crisis_coverage_report(r_train, paths, model_name="CR-CWB-EVT", n_clusters=5)

    with st.spinner("Computing model health score..."):
        health = compute_model_health(
            var_backtest_df  = var_df,
            tail_mass_ratio  = body_row.get("tail_mass_ratio_0_5pct"),
            crisis_summary   = crisis["summary"],
            stability_result = stab,
            features         = features,
            model_name       = "CR-CWB-EVT",
        )
        health_df = health_report_to_dataframe(health)

    # ── Headline ──────────────────────────────────────────────────────────────
    conf_color = {"HIGH":"#1D9E75","MEDIUM":"#EF9F27","LOW":"#E24B4A"}.get(
        health["overall_confidence"], "#888780"
    )
    st.markdown(
        f'<div style="background:{conf_color}1A;border-left:4px solid {conf_color};'
        f'padding:14px 18px;border-radius:6px;margin-bottom:16px;">'
        f'<span style="font-size:1.1rem;font-weight:500;color:{conf_color};">'
        f'Overall confidence: {health["overall_confidence"]}</span><br>'
        f'<span style="font-size:0.9rem">{health["headline"]}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Health checks table ───────────────────────────────────────────────────
    st.markdown("#### Validation checks")

    def status_icon(s):
        return {"PASS":"✅","FAIL":"❌","WARNING":"⚠️","LOW":"✅","MODERATE":"⚠️",
                "HIGH":"❌","NARROW":"✅","WIDE":"❌","UNKNOWN":"❓"}.get(s,"❓")

    for _, row in health_df.iterrows():
        if row["check"] == "OVERALL":
            continue
        icon = status_icon(row["status"])
        with st.expander(f"{icon}  **{row['check'].replace('_',' ').title()}** — {row['status']}"):
            st.write(row["detail"])

    st.markdown("---")

    # ── VaR backtest table ────────────────────────────────────────────────────
    st.markdown("#### VaR / ES backtest panel")
    display_var = var_df[[
        "alpha","n_obs","n_breaches","expected_breaches","obs_exp_ratio",
        "kupiec_verdict","independence_verdict","es_severity_ratio","overall_verdict"
    ]].copy()
    display_var["alpha"] = display_var["alpha"].map(lambda x: f"{x:.1%}")
    display_var["obs_exp_ratio"] = display_var["obs_exp_ratio"].map(lambda x: f"{x:.2f}")
    st.dataframe(display_var, use_container_width=True, height=220)

    # ── Crisis coverage ───────────────────────────────────────────────────────
    st.markdown("#### Crisis archetype coverage")
    arch = crisis.get("archetype_table", pd.DataFrame())
    if len(arch) > 0:
        st.dataframe(arch, use_container_width=True, height=220)

        fig_arch = px.bar(
            arch,
            x="archetype_name",
            y=[c for c in arch.columns if c.startswith("coverage")],
            title="Crisis archetype coverage@2",
            color_discrete_sequence=["#1D9E75"],
            labels={"value": "Coverage", "archetype_name": "Archetype"},
        )
        fig_arch.add_hline(y=0.60, line_dash="dash", line_color="#EF9F27",
                           annotation_text="60% target")
        fig_arch.update_layout(height=280, showlegend=False, margin=dict(t=40,b=60))
        st.plotly_chart(fig_arch, use_container_width=True)

    # ── Body realism ──────────────────────────────────────────────────────────
    with st.expander("Body realism metrics (Layer 1)"):
        body_disp = {
            "KS statistic":           f"{body_row.get('ks_stat', np.nan):.4f}",
            "Wasserstein distance":   f"{body_row.get('wasserstein', np.nan):.6f}",
            "ACF(r) error":           f"{body_row.get('acf_returns_error', np.nan):.4f}",
            "ACF(r²) error":          f"{body_row.get('acf_sq_returns_error', np.nan):.4f}",
            "Tail mass ratio (0.5%)": f"{body_row.get('tail_mass_ratio_0_5pct', np.nan):.3f}",
        }
        st.table(pd.Series(body_disp).rename("Value"))
        st.markdown(
            '<div class="warn-box">Body realism PASS does not imply risk validity. '
            'Check VaR and crisis coverage results above.</div>',
            unsafe_allow_html=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────
    csv_health = health_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download health report (CSV)",
        data=csv_health,
        file_name=f"{ticker}_model_health.csv",
        mime="text/csv",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — OPTIONS DISAGREEMENT                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with tab3:
    st.subheader("Options Disagreement Monitor")
    st.caption(
        "Compares model physical P(S_T < K) with options-implied risk-neutral P_Q(S_T < K). "
        "Differences may reflect risk premia — not necessarily model error."
    )

    expiries = get_option_expiries(ticker)
    if not expiries:
        st.warning(f"No options data for {ticker}.")
        st.stop()

    selected_expiry = st.selectbox(
        "Expiry date",
        expiries,
        index=min(1, len(expiries) - 1),
    )

    with st.spinner("Fetching options chain and computing disagreement..."):
        try:
            raw_puts   = fetch_put_chain(ticker, expiry=selected_expiry)
            clean_puts = clean_put_chain(raw_puts)
            bs_puts    = add_bs_metrics(clean_puts)
            model_puts = add_model_physical_probabilities(bs_puts, paths)
            scored     = score_model_market_disagreement(model_puts)
            summary    = summarise_dashboard(scored)
            vol_summ   = vol_surface_summary(bs_puts)
        except Exception as e:
            st.error(f"Options engine failed: {e}")
            st.stop()

    # ── Header metrics ────────────────────────────────────────────────────────
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Puts analysed",          summary["n_options"])
    h2.metric("ATM implied vol",
              f"{vol_summ.get('atm_implied_vol', np.nan):.1%}" if vol_summ.get("atm_implied_vol") else "N/A")
    h3.metric("Vol skew (OTM 5%)",
              f"{vol_summ.get('vol_skew_5pct', np.nan):.2%}" if vol_summ.get("vol_skew_5pct") else "N/A")
    h4.metric("Mean disagreement",
              f"{summary.get('mean_probability_disagreement', np.nan):.2%}")
    h5.metric("Top interpretation",     summary.get("top_interpretation","N/A"))

    # ── Probability surface ───────────────────────────────────────────────────
    sorted_df = scored.sort_values("strike")
    fig_prob  = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=sorted_df["moneyness"], y=sorted_df["physical_prob_ITM"],
        mode="lines+markers", name="Physical P (model)",
        line=dict(color="#1D9E75", width=2), marker=dict(size=6)
    ))
    fig_prob.add_trace(go.Scatter(
        x=sorted_df["moneyness"], y=sorted_df["risk_neutral_prob_ITM"],
        mode="lines+markers", name="Risk-neutral P_Q (options)",
        line=dict(color="#7F77DD", width=2, dash="dash"), marker=dict(size=6, symbol="square")
    ))
    fig_prob.add_vline(x=1.0, line_dash="dot", line_color="#888780", annotation_text="ATM")
    fig_prob.update_layout(
        title="P(S_T < K): physical vs risk-neutral by strike",
        xaxis_title="Moneyness (K/S)", yaxis_title="Probability",
        height=360, legend=dict(x=0.02, y=0.98),
        margin=dict(t=50, b=40, l=50, r=20)
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Disagreement scatter ──────────────────────────────────────────────────
    color_map = {
        "MODEL_SEES_MORE_DOWNSIDE_THAN_OPTIONS": "#1D9E75",
        "OPTIONS_PRICE_MORE_DOWNSIDE_THAN_MODEL": "#E24B4A",
        "NO_MATERIAL_DISAGREEMENT": "#B4B2A9",
    }
    fig_sc = px.scatter(
        scored, x="moneyness", y="probability_disagreement",
        size="openInterest", color="interpretation",
        color_discrete_map=color_map,
        hover_data={"strike":":.2f","mid":":.4f",
                    "physical_prob_ITM":":.3f","risk_neutral_prob_ITM":":.3f"},
        title="Disagreement (physical − risk-neutral) per strike",
        height=380,
    )
    fig_sc.add_hline(y=0, line_color="#888780", line_width=1)
    fig_sc.add_hline(y=0.03, line_dash="dot", line_color="#1D9E75",
                     annotation_text="+3% flag")
    fig_sc.add_hline(y=-0.03, line_dash="dot", line_color="#E24B4A",
                     annotation_text="-3% flag")
    fig_sc.update_layout(margin=dict(t=50,b=40,l=50,r=20), legend=dict(x=0.02,y=0.98))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown("#### Top disagreements by liquidity-weighted score")
    disp_cols = [c for c in [
        "contractSymbol","strike","moneyness","mid","bid_ask_pct","openInterest",
        "implied_vol","physical_prob_ITM","risk_neutral_prob_ITM",
        "probability_disagreement","tail_disagreement_score","interpretation","confidence"
    ] if c in scored.columns]
    st.dataframe(scored[disp_cols].head(25), use_container_width=True, height=350)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        """<div class="warn-box">
        Physical probability P(S_T &lt; K) and risk-neutral probability P_Q(S_T &lt; K)
        are derived from different measures. A gap between them does not imply mispricing.
        Common explanations include the volatility risk premium, crash-risk premium,
        liquidity premium, and dealer hedging flows.
        <strong>This is not a trading recommendation.</strong>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "⬇  Download disagreement table (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{selected_expiry}_disagreement.csv",
        mime="text/csv",
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "**TailGuard** — Regime-calibrated tail-risk intelligence. "
    "Built on CR-CWB-EVT (Current-Regime Crisis-Weighted Bootstrap + EVT tail correction). "
    "Model health validated via four-layer framework: body realism, VaR/ES backtesting, "
    "crisis archetype coverage, and pathwise stability."
)
