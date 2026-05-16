# =============================================================================
# VAR AND ES BACKTESTING
# Layer 2 of validation — the first real risk validity test.
#
# Tests:
#   Kupiec (1995)           — breach frequency
#   Christoffersen (1998)   — breach independence
#   Conditional coverage    — joint frequency + independence
#   ES severity error       — tail adequacy beyond VaR
#   Pathwise stability      — does it pass at path level, not just pooled?
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2


# =============================================================================
# CORE TESTS
# =============================================================================

def kupiec_test(n_breaches, n_obs, alpha, eps=1e-12):
    """
    Kupiec (1995) Proportion of Failures (POF) test.
    H0: breach frequency = alpha (model is correctly calibrated)

    Returns
    ───────
    lr_stat : likelihood-ratio statistic
    pval    : p-value (PASS = high p-value, FAIL = low p-value)
    verdict : "PASS" / "FAIL" / "INSUFFICIENT_DATA"
    """
    if n_obs < 30:
        return np.nan, np.nan, "INSUFFICIENT_DATA"

    x    = int(n_breaches)
    phat = min(max(x / n_obs, eps), 1 - eps)
    p0   = min(max(alpha, eps), 1 - eps)

    ll_null = (n_obs - x) * np.log(1 - p0)  + x * np.log(p0)
    ll_alt  = (n_obs - x) * np.log(1 - phat) + x * np.log(phat)

    lr   = -2 * (ll_null - ll_alt)
    pval = float(1 - chi2.cdf(lr, df=1))

    return float(lr), pval, ("PASS" if pval >= 0.05 else "FAIL")


def christoffersen_independence_test(breach_series, eps=1e-12):
    """
    Christoffersen (1998) independence test.
    H0: breaches are serially independent

    breach_series : 1D binary array (1 = breach, 0 = no breach)

    Returns lr_stat, pval, verdict
    """
    b = np.asarray(breach_series).astype(int)
    n = len(b)

    if n < 30:
        return np.nan, np.nan, "INSUFFICIENT_DATA"

    n00 = int(np.sum((b[:-1] == 0) & (b[1:] == 0)))
    n01 = int(np.sum((b[:-1] == 0) & (b[1:] == 1)))
    n10 = int(np.sum((b[:-1] == 1) & (b[1:] == 0)))
    n11 = int(np.sum((b[:-1] == 1) & (b[1:] == 1)))

    p01 = n01 / max(n00 + n01, eps)
    p11 = n11 / max(n10 + n11, eps)
    p   = (n01 + n11) / max(n00 + n01 + n10 + n11, eps)

    p01  = min(max(p01, eps), 1 - eps)
    p11  = min(max(p11, eps), 1 - eps)
    p    = min(max(p,   eps), 1 - eps)

    ll_null = (
        (n00 + n10) * np.log(1 - p)  +
        (n01 + n11) * np.log(p)
    )
    ll_alt  = (
        n00 * np.log(1 - p01) + n01 * np.log(p01) +
        n10 * np.log(1 - p11) + n11 * np.log(p11)
    )

    lr   = -2 * (ll_null - ll_alt)
    pval = float(1 - chi2.cdf(lr, df=1))

    return float(lr), pval, ("PASS" if pval >= 0.05 else "FAIL")


def conditional_coverage_test(breach_series, alpha, eps=1e-12):
    """
    Christoffersen (1998) conditional coverage = Kupiec + independence.
    H0: correct frequency AND serial independence

    Returns lr_stat, pval, verdict
    """
    b = np.asarray(breach_series).astype(int)
    n = len(b)

    if n < 30:
        return np.nan, np.nan, "INSUFFICIENT_DATA"

    _, pof_lr, _  = kupiec_test(b.sum(), n, alpha)
    _, ind_lr, _  = christoffersen_independence_test(b)

    if np.isnan(pof_lr) or np.isnan(ind_lr):
        return np.nan, np.nan, "INSUFFICIENT_DATA"

    lr   = float(pof_lr + ind_lr)        # Approx — joint statistic
    pval = float(1 - chi2.cdf(lr, df=2))

    return lr, pval, ("PASS" if pval >= 0.05 else "FAIL")


def es_severity_error(losses_real, var_estimate, es_estimate, alpha):
    """
    Compute the ratio of realised ES to modelled ES for exceedances.
    A ratio near 1 = well-calibrated.
    Ratio < 1 = model over-estimates tail loss (conservative)
    Ratio > 1 = model under-estimates tail loss (dangerous)

    Parameters
    ──────────
    losses_real  : 1D array, realised losses (positive = loss)
    var_estimate : float, modelled VaR at confidence level
    es_estimate  : float, modelled ES at confidence level
    alpha        : float, VaR confidence level (e.g. 0.05)

    Returns
    ───────
    realised_es  : float
    modelled_es  : float
    severity_ratio: float (realised / modelled)
    verdict      : "PASS" / "FAIL" / "INSUFFICIENT_DATA"
    """
    exceedances = losses_real[losses_real >= var_estimate]

    if len(exceedances) < 5:
        return np.nan, es_estimate, np.nan, "INSUFFICIENT_DATA"

    realised_es   = float(exceedances.mean())
    severity_ratio= realised_es / max(es_estimate, 1e-12)

    verdict = (
        "PASS"             if 0.70 <= severity_ratio <= 1.50 else
        "WARNING_MILD"     if 0.50 <= severity_ratio <= 2.00 else
        "FAIL"
    )

    return realised_es, es_estimate, severity_ratio, verdict


# =============================================================================
# FULL VAR/ES PANEL FOR ONE MODEL
# =============================================================================

def var_es_backtest_panel(
    real_returns,
    synthetic_paths,
    model_name="model",
    alphas=None,
):
    """
    Run the full VaR/ES validation panel for one model.

    Parameters
    ──────────
    real_returns    : 1D array of realised log returns (test period)
    synthetic_paths : 2D array (n_sims × horizon) OR 1D flat array
    model_name      : str
    alphas          : list of confidence levels (e.g. [0.10, 0.05, 0.01])

    Returns
    ───────
    pd.DataFrame, one row per alpha level
    """
    if alphas is None:
        alphas = [0.20, 0.10, 0.05, 0.025, 0.01, 0.005]

    real  = np.asarray(real_returns).reshape(-1)
    synth = np.asarray(synthetic_paths).reshape(-1)

    rows = []
    for alpha in alphas:
        # Modelled VaR and ES from synthetic distribution
        losses_synth = -synth
        var_model    = float(np.quantile(losses_synth, 1 - alpha))
        tail_synth   = losses_synth[losses_synth >= var_model]
        es_model     = float(tail_synth.mean()) if len(tail_synth) > 0 else np.nan

        # Realised breaches
        losses_real  = -real
        breaches     = (losses_real >= var_model).astype(int)
        n_breaches   = int(breaches.sum())
        n_obs        = len(real)
        expected_n   = n_obs * alpha
        obs_exp_ratio= n_breaches / max(expected_n, 1e-12)

        # Tests
        kup_lr, kup_pval, kup_verdict = kupiec_test(n_breaches, n_obs, alpha)
        ind_lr, ind_pval, ind_verdict = christoffersen_independence_test(breaches)
        cc_lr,  cc_pval,  cc_verdict  = conditional_coverage_test(breaches, alpha)

        # ES severity
        real_es, model_es, sev_ratio, es_verdict = es_severity_error(
            losses_real, var_model, es_model, alpha
        )

        # Overall verdict
        pass_count = sum([
            kup_verdict == "PASS",
            ind_verdict == "PASS",
            cc_verdict  == "PASS",
            es_verdict  == "PASS",
        ])
        overall = (
            "PASS"    if pass_count >= 3 else
            "WARNING" if pass_count == 2 else
            "FAIL"
        )

        rows.append({
            "model":             model_name,
            "alpha":             alpha,
            "n_obs":             n_obs,
            "n_breaches":        n_breaches,
            "expected_breaches": float(expected_n),
            "obs_exp_ratio":     float(obs_exp_ratio),
            "var_model":         var_model,
            "es_model":          model_es,
            "realised_es":       real_es,
            "es_severity_ratio": sev_ratio,
            "kupiec_lr":         kup_lr,
            "kupiec_pval":       kup_pval,
            "kupiec_verdict":    kup_verdict,
            "independence_lr":   ind_lr,
            "independence_pval": ind_pval,
            "independence_verdict": ind_verdict,
            "conditional_cov_pval": cc_pval,
            "conditional_cov_verdict": cc_verdict,
            "es_verdict":        es_verdict,
            "overall_verdict":   overall,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PATHWISE STABILITY TEST
# =============================================================================

def pathwise_var_stability(
    real_returns,
    synthetic_paths,
    model_name="model",
    alpha=0.05,
    aggregation_modes=None,
):
    """
    Test whether VaR validity is stable across aggregation modes.

    The key risk:
    A model can PASS pooled VaR but FAIL pathwise median VaR.
    That means the validation result depends on how you aggregate.
    This is a false-safety pattern.

    Modes tested:
        pooled          all path steps concatenated
        pathwise_median VaR of median path cumulative return
        pathwise_q25    VaR of Q25 path
        pathwise_q75    VaR of Q75 path

    Returns dict with Kupiec p-values and verdicts per mode.
    Flags aggregation-sensitivity if verdicts differ across modes.
    """
    if aggregation_modes is None:
        aggregation_modes = ["pooled", "pathwise_median", "pathwise_q25", "pathwise_q75"]

    real   = np.asarray(real_returns).reshape(-1)
    paths  = np.asarray(synthetic_paths)

    if paths.ndim == 1:
        # Cannot do pathwise if already flat
        paths = paths.reshape(1, -1)

    cum_returns = paths.sum(axis=1)
    n_real = len(real)
    losses_real = -real

    results = {}

    for mode in aggregation_modes:
        if mode == "pooled":
            synth_dist = paths.reshape(-1)
        elif mode == "pathwise_median":
            median_val = float(np.median(cum_returns))
            synth_dist = np.array([median_val])
        elif mode == "pathwise_q25":
            q25_val = float(np.quantile(cum_returns, 0.25))
            synth_dist = np.array([q25_val])
        elif mode == "pathwise_q75":
            q75_val = float(np.quantile(cum_returns, 0.75))
            synth_dist = np.array([q75_val])
        else:
            continue

        losses_synth = -synth_dist
        var_mode = float(np.quantile(losses_synth, max(1 - alpha, 1e-6)))

        breaches  = (losses_real >= var_mode).astype(int)
        n_breach  = int(breaches.sum())

        _, kup_pval, verdict = kupiec_test(n_breach, n_real, alpha)
        results[mode] = {
            "var":     var_mode,
            "n_breach":n_breach,
            "kupiec_pval": kup_pval,
            "verdict": verdict,
        }

    # Aggregation sensitivity: do verdicts differ?
    verdicts = [v["verdict"] for v in results.values() if v["verdict"] != "INSUFFICIENT_DATA"]
    is_sensitive = len(set(verdicts)) > 1

    return {
        "model":                      model_name,
        "alpha":                      alpha,
        "aggregation_sensitive":      is_sensitive,
        "aggregation_sensitive_flag": "WARNING" if is_sensitive else "OK",
        "modes":                      results,
    }


# =============================================================================
# MULTI-MODEL PANEL
# =============================================================================

def var_backtest_multi_model(real_returns, models_dict, alphas=None):
    """
    Run full VaR/ES panel for multiple models.

    Parameters
    ──────────
    real_returns : 1D array
    models_dict  : {model_name: synthetic_paths_array}
    alphas       : list of alpha levels

    Returns
    ───────
    pd.DataFrame, one row per (model, alpha)
    """
    dfs = []
    for name, paths in models_dict.items():
        df = var_es_backtest_panel(real_returns, paths, model_name=name, alphas=alphas)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
