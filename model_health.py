# =============================================================================
# MODEL HEALTH SCORING
# The product feature that turns this from a dashboard into a
# risk-validity operating system.
#
# PRINCIPLE (from your dissertation):
# The model is valuable only if it tells users WHEN NOT TO TRUST THE MODEL.
#
# This module produces a Model Health Report:
#
#   Calibration:         PASS / WARNING / FAIL
#   Tail mass:           PASS / WARNING / FAIL
#   Crisis coverage:     PASS / WARNING / FAIL
#   Pathwise stability:  PASS / WARNING / FAIL
#   Recent regime drift: LOW / MODERATE / HIGH
#   Options disagreement:NARROW / MODERATE / WIDE
#   Overall confidence:  HIGH / MEDIUM / LOW
#
# Every output includes an explanation of WHY.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# INDIVIDUAL HEALTH CHECKS
# =============================================================================

def check_calibration(var_backtest_df, primary_alpha=0.05):
    """
    Pass if Kupiec and conditional coverage both pass at primary alpha.
    """
    if var_backtest_df is None or len(var_backtest_df) == 0:
        return {
            "status":      "UNKNOWN",
            "detail":      "No VaR backtest data provided.",
            "confidence_weight": 0.0,
        }

    row = var_backtest_df[var_backtest_df["alpha"] == primary_alpha]
    if len(row) == 0:
        return {
            "status":  "UNKNOWN",
            "detail":  f"No results for alpha={primary_alpha}.",
            "confidence_weight": 0.0,
        }

    row = row.iloc[0]
    kup    = row.get("kupiec_verdict", "UNKNOWN")
    cc     = row.get("conditional_cov_verdict", "UNKNOWN")
    es_v   = row.get("es_verdict", "UNKNOWN")
    obs_exp= row.get("obs_exp_ratio", np.nan)

    if kup == "PASS" and cc == "PASS":
        status = "PASS"
        weight = 1.0
        detail = (
            f"Breach frequency ({obs_exp:.2f}× expected) is within acceptable range. "
            f"Breaches are not clustered."
        )
    elif kup == "PASS":
        status = "WARNING"
        weight = 0.5
        detail = (
            f"Breach frequency is acceptable ({obs_exp:.2f}×) but breaches may be clustered. "
            f"Independence test: {cc}."
        )
    else:
        status = "FAIL"
        weight = 0.0
        detail = (
            f"Breach frequency is {obs_exp:.2f}× expected. "
            f"Model VaR is not reliable at {primary_alpha*100:.0f}% level."
        )

    if es_v == "FAIL":
        status  = "FAIL"
        weight  = 0.0
        detail += " ES severity also fails — tail losses exceed model estimates."

    return {"status": status, "detail": detail, "confidence_weight": weight}


def check_tail_mass(var_backtest_df=None, tail_mass_ratio=None):
    """
    Check whether the model generates enough mass in the deep tail.
    Tail mass ratio = synthetic_mass_below_0.5pct / historical_mass_below_0.5pct
    Ideal = 1.0. Below 0.5 = serious under-coverage.
    """
    if tail_mass_ratio is None:
        if var_backtest_df is not None:
            # Derive from body realism metrics if available
            tail_mass_ratio = var_backtest_df.get("tail_mass_ratio_0_5pct", None)

    if tail_mass_ratio is None:
        return {
            "status": "UNKNOWN",
            "detail": "Tail mass ratio not provided.",
            "confidence_weight": 0.5,
        }

    tmr = float(tail_mass_ratio)

    if 0.60 <= tmr <= 2.00:
        status = "PASS"
        weight = 1.0
        detail = f"Tail mass ratio {tmr:.2f} — model generates adequate extreme-loss mass."
    elif 0.30 <= tmr < 0.60:
        status = "WARNING"
        weight = 0.4
        detail = (
            f"Tail mass ratio {tmr:.2f} — model under-represents extreme losses. "
            f"EVT correction may be insufficient."
        )
    elif tmr > 2.00:
        status = "WARNING"
        weight = 0.5
        detail = f"Tail mass ratio {tmr:.2f} — model over-represents extreme losses."
    else:
        status = "FAIL"
        weight = 0.0
        detail = (
            f"Tail mass ratio {tmr:.2f} — serious tail under-coverage. "
            f"Risk estimates will understate true exposure."
        )

    return {"status": status, "detail": detail, "confidence_weight": weight}


def check_crisis_coverage(crisis_summary=None, coverage_at_2=None, precision_trap=None):
    """
    Check whether the model covers historical crisis archetypes.
    """
    if crisis_summary is None and coverage_at_2 is None:
        return {
            "status": "UNKNOWN",
            "detail": "Crisis coverage not computed.",
            "confidence_weight": 0.5,
        }

    # Extract from summary dict
    if crisis_summary is not None and isinstance(crisis_summary, dict):
        coverage_at_2  = crisis_summary.get("coverage_at_2.0", coverage_at_2)
        precision_trap = crisis_summary.get("precision_trap_detected", precision_trap)
        n_arch_fail    = crisis_summary.get("n_archetypes_fail", 0)
    else:
        n_arch_fail = 0

    if coverage_at_2 is None:
        return {
            "status": "UNKNOWN",
            "detail": "Coverage@2 not available.",
            "confidence_weight": 0.5,
        }

    cov = float(coverage_at_2)

    if cov >= 0.70 and not precision_trap and n_arch_fail == 0:
        status = "PASS"
        weight = 1.0
        detail = f"Coverage@2 = {cov:.1%}. All crisis archetypes represented."
    elif cov >= 0.55:
        status = "WARNING"
        weight = 0.5
        detail = (
            f"Coverage@2 = {cov:.1%}. "
            f"{n_arch_fail} archetype(s) under-represented."
        )
        if precision_trap:
            detail += " Precision trap detected: crisis paths cluster in a narrow region."
    else:
        status = "FAIL"
        weight = 0.0
        detail = (
            f"Coverage@2 = {cov:.1%} — model misses large portions of crisis manifold. "
            f"{n_arch_fail} archetype(s) failing."
        )

    return {"status": status, "detail": detail, "confidence_weight": weight}


def check_pathwise_stability(stability_result=None):
    """
    Check whether VaR validity holds across aggregation modes.
    """
    if stability_result is None:
        return {
            "status": "UNKNOWN",
            "detail": "Pathwise stability not computed.",
            "confidence_weight": 0.5,
        }

    sensitive = stability_result.get("aggregation_sensitive", False)
    modes     = stability_result.get("modes", {})

    verdicts = {k: v["verdict"] for k, v in modes.items() if v["verdict"] != "INSUFFICIENT_DATA"}
    n_pass   = sum(v == "PASS" for v in verdicts.values())
    n_total  = len(verdicts)

    if not sensitive and n_pass == n_total:
        status = "PASS"
        weight = 1.0
        detail = "VaR validity is stable across all aggregation modes."
    elif not sensitive:
        status = "WARNING"
        weight = 0.6
        detail = f"Consistent across modes but {n_total - n_pass}/{n_total} modes fail Kupiec."
    else:
        status = "WARNING"
        weight = 0.3
        detail = (
            "Aggregation-sensitive: VaR validity depends on how returns are pooled. "
            "Pooled result may not reflect path-level risk."
        )

    return {"status": status, "detail": detail, "confidence_weight": weight}


def check_regime_drift(features, lookback_days=60):
    """
    Check whether current market regime is far from the training distribution.
    High drift = model predictions less reliable.
    """
    if features is None or len(features) < lookback_days + 100:
        return {
            "status": "UNKNOWN",
            "detail": "Insufficient feature history.",
            "confidence_weight": 0.5,
        }

    recent = features.iloc[-lookback_days:]
    hist   = features.iloc[:-lookback_days]

    drift_scores = []
    for col in ["vol_20d", "vix_level", "drawdown_20d"]:
        if col not in features.columns:
            continue
        hist_q25 = hist[col].quantile(0.25)
        hist_q75 = hist[col].quantile(0.75)
        iqr      = hist_q75 - hist_q25
        if iqr <= 0:
            continue
        recent_mean = recent[col].mean()
        drift = abs(recent_mean - hist[col].median()) / max(iqr, 1e-12)
        drift_scores.append(drift)

    if not drift_scores:
        return {
            "status": "UNKNOWN",
            "detail": "Could not compute drift score.",
            "confidence_weight": 0.5,
        }

    mean_drift = float(np.mean(drift_scores))

    if mean_drift < 1.0:
        status = "LOW"
        weight = 1.0
        detail = f"Current regime (drift={mean_drift:.2f}) is within historical training distribution."
    elif mean_drift < 2.5:
        status = "MODERATE"
        weight = 0.6
        detail = (
            f"Current regime (drift={mean_drift:.2f}) is somewhat unusual vs training history. "
            f"Predictions carry moderate uncertainty."
        )
    else:
        status = "HIGH"
        weight = 0.2
        detail = (
            f"Current regime (drift={mean_drift:.2f}) is significantly outside training history. "
            f"Model predictions are less reliable."
        )

    return {"status": status, "detail": detail, "confidence_weight": weight, "drift_score": mean_drift}


def check_options_disagreement(disagreement_summary=None):
    """
    Classify the current options market disagreement level.
    """
    if disagreement_summary is None:
        return {
            "status": "UNKNOWN",
            "detail": "Options disagreement not computed.",
            "confidence_weight": 0.5,
        }

    mean_gap = disagreement_summary.get("mean_probability_disagreement", np.nan)
    max_gap  = disagreement_summary.get("max_abs_disagreement", np.nan)
    n_more   = disagreement_summary.get("n_model_sees_more_downside", 0)
    n_less   = disagreement_summary.get("n_options_price_more_downside", 0)

    if np.isnan(mean_gap):
        return {
            "status": "UNKNOWN",
            "detail": "Options data not available.",
            "confidence_weight": 0.5,
        }

    gap = abs(float(mean_gap))

    if gap < 0.03:
        status = "NARROW"
        weight = 0.9
        detail = (
            f"Model and options market broadly agree (mean gap = {mean_gap:.1%}). "
            f"Model predictions carry higher confidence."
        )
    elif gap < 0.08:
        status = "MODERATE"
        weight = 0.6
        detail = (
            f"Moderate disagreement (mean gap = {mean_gap:.1%}). "
            f"Model sees {'more' if mean_gap > 0 else 'less'} downside than options. "
            f"Gap may reflect risk premia — not necessarily model error."
        )
    else:
        status = "WIDE"
        weight = 0.3
        detail = (
            f"Wide disagreement (mean gap = {mean_gap:.1%}). "
            f"Either model or options market may be significantly wrong. "
            f"Treat predictions with caution."
        )

    return {"status": status, "detail": detail, "confidence_weight": weight, "mean_gap": mean_gap}


# =============================================================================
# COMPOSITE HEALTH SCORE
# =============================================================================

def compute_model_health(
    var_backtest_df=None,
    tail_mass_ratio=None,
    crisis_summary=None,
    stability_result=None,
    features=None,
    disagreement_summary=None,
    model_name="model",
):
    """
    Compute overall model health from all validation layers.

    Returns a structured health report dict.
    Usage: display in the dashboard Model Health tab.
    """

    checks = {
        "calibration":        check_calibration(var_backtest_df),
        "tail_mass":          check_tail_mass(tail_mass_ratio=tail_mass_ratio),
        "crisis_coverage":    check_crisis_coverage(crisis_summary),
        "pathwise_stability": check_pathwise_stability(stability_result),
        "regime_drift":       check_regime_drift(features),
        "options_disagreement":check_options_disagreement(disagreement_summary),
    }

    # Overall confidence = weighted average of individual weights
    weights = [c["confidence_weight"] for c in checks.values()]
    overall_confidence_score = float(np.mean(weights)) if weights else 0.0

    overall_confidence = (
        "HIGH"   if overall_confidence_score >= 0.75 else
        "MEDIUM" if overall_confidence_score >= 0.45 else
        "LOW"
    )

    # Status summary
    status_counts = {}
    for check_name, check_result in checks.items():
        s = check_result.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Generate headline interpretation
    n_fail    = status_counts.get("FAIL", 0)
    n_warning = status_counts.get("WARNING", 0) + status_counts.get("MODERATE", 0) + status_counts.get("HIGH", 0)
    n_pass    = status_counts.get("PASS", 0) + status_counts.get("LOW", 0) + status_counts.get("NARROW", 0)

    if n_fail > 0:
        headline = (
            f"{n_fail} validation layer(s) failing. "
            f"Risk estimates from this model should not be relied upon without further investigation."
        )
    elif n_warning > 1:
        headline = (
            f"Multiple validation warnings ({n_warning}). "
            f"Predictions carry meaningful uncertainty. Use alongside other indicators."
        )
    elif n_warning == 1:
        headline = (
            f"One validation warning. "
            f"Predictions are broadly reliable but check the flagged layer."
        )
    else:
        headline = "All validation layers pass. Predictions carry high confidence."

    return {
        "model":                  model_name,
        "computed_at":            datetime.utcnow().isoformat(),
        "overall_confidence":     overall_confidence,
        "overall_confidence_score":overall_confidence_score,
        "headline":               headline,
        "n_pass":                 n_pass,
        "n_warning":              n_warning,
        "n_fail":                 n_fail,
        "checks":                 checks,
    }


def health_report_to_dataframe(health_report):
    """
    Flatten the health report into a DataFrame for display or export.
    """
    rows = []
    for check_name, check_result in health_report.get("checks", {}).items():
        rows.append({
            "check":             check_name,
            "status":            check_result.get("status", "UNKNOWN"),
            "confidence_weight": check_result.get("confidence_weight", np.nan),
            "detail":            check_result.get("detail", ""),
        })

    rows.append({
        "check":             "OVERALL",
        "status":            health_report.get("overall_confidence", "UNKNOWN"),
        "confidence_weight": health_report.get("overall_confidence_score", np.nan),
        "detail":            health_report.get("headline", ""),
    })

    return pd.DataFrame(rows)
