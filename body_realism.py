# =============================================================================
# BODY REALISM METRICS
# Layer 1 of validation — necessary but not sufficient.
#
# KEY WARNING (embedded in every output):
# Passing body realism tests does NOT prove risk validity.
# TimeGAN can have KS=0.03 and still fail Kupiec.
# These metrics tell you the path looks plausible.
# They do NOT tell you VaR is trustworthy.
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance


def compute_body_realism(real_returns, synthetic_paths, model_name="model"):
    """
    Compute distributional realism metrics between real returns
    and synthetic paths (flattened).

    Parameters
    ──────────
    real_returns    : 1D array-like of historical log returns
    synthetic_paths : 2D array (n_sims × horizon) OR 1D flat array
    model_name      : str label for the output row

    Returns
    ───────
    dict of scalar metrics
    """
    real = np.asarray(real_returns).reshape(-1)
    synth = np.asarray(synthetic_paths).reshape(-1)

    # KS test
    ks_stat, ks_pval = ks_2samp(real, synth)

    # Wasserstein distance
    wass = wasserstein_distance(real, synth)

    # Moment matching
    moments = {}
    for label, arr in [("real", real), ("synth", synth)]:
        moments[f"{label}_mean"]     = float(np.mean(arr))
        moments[f"{label}_std"]      = float(np.std(arr))
        moments[f"{label}_skew"]     = float(stats.skew(arr))
        moments[f"{label}_kurtosis"] = float(stats.kurtosis(arr))

    # Autocorrelation of returns (lags 1-5)
    acf_real  = _rolling_acf(real,  lags=5)
    acf_synth = _rolling_acf(synth, lags=5)
    acf_error = float(np.mean(np.abs(acf_real - acf_synth)))

    # Autocorrelation of squared returns (volatility clustering)
    acf2_real  = _rolling_acf(real**2,  lags=5)
    acf2_synth = _rolling_acf(synth**2, lags=5)
    acf2_error = float(np.mean(np.abs(acf2_real - acf2_synth)))

    # Tail mass ratio (how much mass below 0.5th percentile of real)
    q_005 = np.quantile(real, 0.005)
    tail_real  = float(np.mean(real  <= q_005))
    tail_synth = float(np.mean(synth <= q_005))
    tail_mass_ratio = tail_synth / max(tail_real, 1e-12)

    # Quantile match errors
    quantile_errors = {}
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        qr = float(np.quantile(real,  q))
        qs = float(np.quantile(synth, q))
        quantile_errors[f"q{int(q*100):02d}_error"] = abs(qs - qr)

    row = {
        "model":               model_name,
        "ks_stat":             float(ks_stat),
        "ks_pval":             float(ks_pval),
        "wasserstein":         float(wass),
        "acf_returns_error":   acf_error,
        "acf_sq_returns_error":acf2_error,
        "tail_mass_ratio_0_5pct": float(tail_mass_ratio),
        "mean_quantile_error": float(np.mean(list(quantile_errors.values()))),
        **moments,
        **quantile_errors,
        # Flag: do NOT interpret as risk validity
        "_warning": (
            "Body realism PASS does not imply risk validity. "
            "See var_backtest and crisis_coverage metrics."
        ),
    }

    return row


def _rolling_acf(arr, lags=5):
    """Compute autocorrelation at lags 1..lags. Returns array of length `lags`."""
    arr = arr - arr.mean()
    n   = len(arr)
    acfs = []
    for lag in range(1, lags + 1):
        if n > lag:
            c = float(np.corrcoef(arr[lag:], arr[:-lag])[0, 1])
        else:
            c = 0.0
        acfs.append(c if np.isfinite(c) else 0.0)
    return np.array(acfs)


def body_realism_panel(real_returns, models_dict):
    """
    Run body realism for multiple models.

    Parameters
    ──────────
    real_returns : 1D array
    models_dict  : {model_name: synthetic_paths_array}

    Returns
    ───────
    pd.DataFrame, one row per model
    """
    rows = []
    for name, paths in models_dict.items():
        rows.append(compute_body_realism(real_returns, paths, model_name=name))
    return pd.DataFrame(rows)
