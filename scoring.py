# =============================================================================
# SCORING ENGINE
# Compares model-implied PHYSICAL probabilities with options-implied
# RISK-NEUTRAL probabilities.
#
# IMPORTANT FRAMING
# ─────────────────
# Physical probability P(S_T < K):
#   What our crisis-aware simulator believes about the real world.
#   Based on historical return patterns, regime detection, EVT tail.
#
# Risk-neutral probability P_Q(S_T < K):
#   What the options market implies about the world AFTER adjusting
#   for risk premia, hedging demand, and liquidity costs.
#
# The gap between them is a DISAGREEMENT SIGNAL, not an arbitrage.
# It may reflect:
#   - Volatility risk premium (market pays to be protected)
#   - Crash risk premium (deep OTM puts systematically expensive)
#   - Liquidity premium
#   - Model error in either direction
#
# We report the gap clearly with a confidence score and liquidity flag.
# We do not recommend trades.
# =============================================================================

import numpy as np
import pandas as pd

from config import MIN_PROB_GAP


# =============================================================================
# PHYSICAL PROBABILITY ESTIMATION
# =============================================================================

def add_model_physical_probabilities(option_df, simulated_paths):
    """
    For each option strike, compute the model's physical probability
    P(S_T < K) from the simulated path distribution.

    Also computes the model-implied expected put payoff
    (discounted expected value of max(K - S_T, 0)).

    Parameters
    ──────────
    option_df        : pd.DataFrame from options_engine.add_bs_metrics
    simulated_paths  : np.ndarray, shape (n_sims, horizon)

    Returns
    ───────
    Enriched DataFrame with physical_prob_ITM and physical_expected_payoff
    """
    df    = option_df.copy()
    paths = np.asarray(simulated_paths)
    spot  = float(df["spot"].iloc[0])

    cum_returns     = paths.sum(axis=1)
    terminal_prices = spot * np.exp(cum_returns)

    physical_probs   = []
    expected_payoffs = []

    for _, row in df.iterrows():
        K       = float(row["strike"])
        payoff  = np.maximum(K - terminal_prices, 0.0)

        physical_probs.append(float(np.mean(terminal_prices < K)))
        expected_payoffs.append(float(np.mean(payoff)))

    df["physical_prob_ITM"]          = physical_probs
    df["physical_expected_payoff"]   = expected_payoffs

    return df


# =============================================================================
# DISAGREEMENT SCORING
# =============================================================================

def score_model_market_disagreement(option_df):
    """
    Compute the disagreement between physical and risk-neutral probabilities.

    Output columns:
    ───────────────
    probability_disagreement    physical - risk_neutral (signed)
    abs_probability_disagreement absolute value of the above
    payoff_minus_mid             model expected payoff minus market mid price
    payoff_to_mid_ratio          ratio of expected payoff to market price
    liquidity_score              0-1 quality score (OI, volume, spread)
    tail_disagreement_score      weighted composite for ranking
    interpretation               human-readable classification
    confidence                   LOW / MEDIUM / HIGH based on liquidity

    Parameters
    ──────────
    option_df : pd.DataFrame with both physical and risk-neutral columns

    Returns
    ───────
    Scored and sorted DataFrame
    """
    df = option_df.copy()

    # Core disagreement
    df["probability_disagreement"]     = df["physical_prob_ITM"] - df["risk_neutral_prob_ITM"]
    df["abs_probability_disagreement"] = df["probability_disagreement"].abs()

    # Price disagreement
    df["payoff_minus_mid"]    = df["physical_expected_payoff"] - df["mid"]
    df["payoff_to_mid_ratio"] = np.where(
        df["mid"] > 0,
        df["physical_expected_payoff"] / df["mid"],
        np.nan
    )

    # Liquidity score (0-1)
    oi_rank     = df["openInterest"].fillna(0).rank(pct=True)
    vol_rank    = df.get("volume", pd.Series(0, index=df.index)).fillna(0).rank(pct=True)
    spread_pen  = 1 / (1 + 10 * df["bid_ask_pct"].fillna(1.0))

    df["liquidity_score"] = (0.60 * oi_rank + 0.40 * vol_rank) * spread_pen

    # Confidence tier
    df["confidence"] = pd.cut(
        df["liquidity_score"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)

    # Composite disagreement score for ranking
    # Uses absolute gap × liquidity quality × sqrt(OI) for prominence
    df["tail_disagreement_score"] = (
        df["abs_probability_disagreement"]
        * df["liquidity_score"]
        * np.sqrt(df["openInterest"].fillna(0).clip(lower=1))
    )

    # Interpretation — use careful, neutral language
    interpretations = []
    for _, row in df.iterrows():
        gap = row["probability_disagreement"]
        if gap >= MIN_PROB_GAP:
            interpretations.append("MODEL_SEES_MORE_DOWNSIDE_THAN_OPTIONS")
        elif gap <= -MIN_PROB_GAP:
            interpretations.append("OPTIONS_PRICE_MORE_DOWNSIDE_THAN_MODEL")
        else:
            interpretations.append("NO_MATERIAL_DISAGREEMENT")

    df["interpretation"] = interpretations

    return df.sort_values("tail_disagreement_score", ascending=False).reset_index(drop=True)


# =============================================================================
# DASHBOARD SUMMARY
# =============================================================================

def summarise_dashboard(scored_df):
    """
    Return a flat dict of headline metrics for the dashboard header.
    """
    if len(scored_df) == 0:
        return {
            "n_options":                     0,
            "top_interpretation":            "NO_DATA",
            "mean_probability_disagreement": None,
            "max_abs_disagreement":          None,
        }

    top = scored_df.iloc[0]

    sees_more   = (scored_df["interpretation"] == "MODEL_SEES_MORE_DOWNSIDE_THAN_OPTIONS").sum()
    sees_less   = (scored_df["interpretation"] == "OPTIONS_PRICE_MORE_DOWNSIDE_THAN_MODEL").sum()
    no_material = (scored_df["interpretation"] == "NO_MATERIAL_DISAGREEMENT").sum()

    return {
        "n_options":                    int(len(scored_df)),
        "top_strike":                   float(top["strike"]),
        "top_moneyness":                float(top["moneyness"]),
        "top_physical_prob":            float(top["physical_prob_ITM"]),
        "top_risk_neutral_prob":        float(top["risk_neutral_prob_ITM"]),
        "top_probability_disagreement": float(top["probability_disagreement"]),
        "top_interpretation":           top["interpretation"],
        "top_confidence":               top["confidence"],
        "n_model_sees_more_downside":   int(sees_more),
        "n_options_price_more_downside":int(sees_less),
        "n_no_material_disagreement":   int(no_material),
        "mean_probability_disagreement":float(scored_df["probability_disagreement"].mean()),
        "max_abs_disagreement":         float(scored_df["abs_probability_disagreement"].max()),
        "mean_liquidity_score":         float(scored_df["liquidity_score"].mean()),
    }


# =============================================================================
# HISTORICAL CONTEXT
# =============================================================================

def add_historical_context(scored_df, r_train=None):
    """
    Optional: attach historical context to each disagreement row.
    - nearest historical analogue for the modelled tail severity
    - percentile rank of current disagreement vs past observations

    Parameters
    ──────────
    scored_df : DataFrame from score_model_market_disagreement
    r_train   : pd.Series of historical log returns (optional)

    Returns
    ───────
    Enriched DataFrame
    """
    df = scored_df.copy()

    if r_train is not None:
        hist_losses = -r_train.values
        tail_threshold = np.quantile(hist_losses, 0.95)
        crisis_episodes = int(np.sum(hist_losses > tail_threshold))

        df["hist_tail_threshold"]   = tail_threshold
        df["hist_crisis_episode_n"] = crisis_episodes
        df["model_exceeds_hist_95pct"] = df["physical_prob_ITM"] > (
            np.mean(hist_losses > df["physical_prob_ITM"].median())
        )

    return df
