# =============================================================================
# CRISIS COVERAGE METRICS
# Layer 3 of validation — your strongest differentiator.
#
# Tests whether the synthetic distribution covers the full diversity
# of historical crisis regimes, not just their pooled statistics.
#
# KEY INSIGHT (from your dissertation):
# TimeGAN has a precision trap — it generates crisis-like paths
# but only one TYPE of crisis path. It misses the full crisis manifold.
# This module detects that failure.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

CRISIS_FEATURE_COLS = [
    "cum_return",
    "realised_vol_ann",
    "max_drawdown",
    "worst_1d",
    "worst_5d",
    "negative_days",
    "tail_days",
    "vol_of_abs_returns",
]


def extract_window_features(returns, window=20, tail_threshold=None):
    """
    Extract crisis-relevant features from every rolling window.

    Parameters
    ──────────
    returns        : 1D array of log returns
    window         : int, window length in days
    tail_threshold : float or None (uses 5th percentile of returns)

    Returns
    ───────
    pd.DataFrame with one row per window start date
    """
    returns = np.asarray(returns).reshape(-1)
    if tail_threshold is None:
        tail_threshold = float(np.quantile(returns, 0.05))

    rows = []
    for start in range(0, len(returns) - window + 1):
        w   = returns[start : start + window]
        cum = np.cumsum(w)
        pk  = np.maximum.accumulate(cum)
        dd  = cum - pk

        worst_5d = float(min(
            [np.sum(w[j:j+5]) for j in range(max(1, len(w)-4))]
        ))

        rows.append({
            "start":               start,
            "cum_return":          float(w.sum()),
            "realised_vol_ann":    float(w.std() * np.sqrt(252)),
            "max_drawdown":        float(dd.min()),
            "worst_1d":            float(w.min()),
            "worst_5d":            worst_5d,
            "negative_days":       int((w < 0).sum()),
            "tail_days":           int((w <= tail_threshold).sum()),
            "vol_of_abs_returns":  float(np.abs(w).std()),
        })

    return pd.DataFrame(rows)


# =============================================================================
# CRISIS WINDOW IDENTIFICATION
# =============================================================================

def identify_crisis_windows(real_windows, vol_quantile=0.95, dd_quantile=0.05):
    """
    Label windows as crisis based on extreme volatility or deep drawdown.

    Returns the crisis sub-DataFrame.
    """
    vol_thr = real_windows["realised_vol_ann"].quantile(vol_quantile)
    dd_thr  = real_windows["max_drawdown"].quantile(dd_quantile)

    is_crisis = (
        (real_windows["realised_vol_ann"] >= vol_thr) |
        (real_windows["max_drawdown"]     <= dd_thr)
    )

    return real_windows[is_crisis].copy()


# =============================================================================
# CRISIS ARCHETYPE CLUSTERING
# =============================================================================

ARCHETYPE_NAMES = {
    0: "Fast crash",
    1: "Slow bleed",
    2: "Volatility spike",
    3: "Rate-shock selloff",
    4: "Liquidity freeze",
    5: "Bear-market grind",
    6: "Rebound volatility",
}


def cluster_crisis_archetypes(crisis_windows, n_clusters=5, seed=42):
    """
    Cluster historical crisis windows into archetypes.

    Returns crisis_windows with 'archetype' column added,
    plus the fitted scaler and KMeans object.
    """
    df      = crisis_windows.copy()
    X       = df[CRISIS_FEATURE_COLS].dropna().values
    scaler  = StandardScaler()
    Xz      = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(Xz)

    df["archetype"] = labels
    df["archetype_name"] = [
        ARCHETYPE_NAMES.get(int(l), f"Cluster {l}") for l in labels
    ]

    return df, scaler, km


# =============================================================================
# COVERAGE METRICS
# =============================================================================

def coverage_at_epsilon(
    crisis_windows,
    synthetic_windows,
    eps_grid=None,
):
    """
    Coverage@ε: fraction of real crisis windows that have at least one
    synthetic neighbour within distance ε in normalised feature space.

    Parameters
    ──────────
    crisis_windows    : pd.DataFrame (real crisis only)
    synthetic_windows : pd.DataFrame (from synthetic paths)
    eps_grid          : list of ε values to evaluate

    Returns
    ───────
    pd.DataFrame with columns [epsilon, coverage, n_covered, n_total]
    """
    if eps_grid is None:
        eps_grid = list(np.arange(0.5, 5.5, 0.5))

    real_feats  = crisis_windows[CRISIS_FEATURE_COLS].dropna().values
    synth_feats = synthetic_windows[CRISIS_FEATURE_COLS].dropna().values

    if len(real_feats) == 0 or len(synth_feats) == 0:
        return pd.DataFrame(columns=["epsilon", "coverage", "n_covered", "n_total"])

    # Normalise jointly
    all_feats = np.vstack([real_feats, synth_feats])
    mu  = all_feats.mean(axis=0)
    sd  = all_feats.std(axis=0)
    sd[sd == 0] = 1.0

    rz = (real_feats  - mu) / sd
    sz = (synth_feats - mu) / sd

    D = cdist(rz, sz, metric="euclidean")
    min_dist = D.min(axis=1)  # nearest synthetic for each real crisis window

    rows = []
    for eps in eps_grid:
        n_covered = int((min_dist <= eps).sum())
        rows.append({
            "epsilon":   float(eps),
            "coverage":  float(n_covered / len(min_dist)),
            "n_covered": n_covered,
            "n_total":   len(min_dist),
        })

    return pd.DataFrame(rows)


def archetype_coverage(
    crisis_windows_with_archetypes,
    synthetic_windows,
    eps=2.0,
):
    """
    Coverage@ε broken down by crisis archetype.

    Returns
    ───────
    pd.DataFrame with one row per archetype, showing coverage@eps.
    """
    all_feats = pd.concat([
        crisis_windows_with_archetypes[CRISIS_FEATURE_COLS],
        synthetic_windows[CRISIS_FEATURE_COLS]
    ], ignore_index=True).dropna()

    mu = all_feats.mean().values
    sd = all_feats.std().values
    sd[sd == 0] = 1.0

    sz = (synthetic_windows[CRISIS_FEATURE_COLS].dropna().values - mu) / sd

    rows = []
    for archetype_id in sorted(crisis_windows_with_archetypes["archetype"].unique()):
        arch_df  = crisis_windows_with_archetypes[
            crisis_windows_with_archetypes["archetype"] == archetype_id
        ]
        arch_name = ARCHETYPE_NAMES.get(int(archetype_id), f"Cluster {archetype_id}")
        rz = (arch_df[CRISIS_FEATURE_COLS].dropna().values - mu) / sd

        if len(rz) == 0:
            continue

        D        = cdist(rz, sz, metric="euclidean")
        min_dist = D.min(axis=1)
        cov      = float(np.mean(min_dist <= eps))

        rows.append({
            "archetype_id":   int(archetype_id),
            "archetype_name": arch_name,
            "n_windows":      int(len(rz)),
            f"coverage_at_{eps}": cov,
            "mean_nearest_dist": float(min_dist.mean()),
            "verdict": "PASS" if cov >= 0.60 else "WARNING" if cov >= 0.40 else "FAIL",
        })

    return pd.DataFrame(rows)


def precision_at_epsilon(crisis_windows, synthetic_windows, eps=2.0):
    """
    Precision@ε: fraction of SYNTHETIC windows that are within ε of
    at least one real crisis window.

    High precision + low coverage = precision trap (TimeGAN failure mode).
    High coverage + low precision = diverse but unrealistic.
    """
    real_feats  = crisis_windows[CRISIS_FEATURE_COLS].dropna().values
    synth_feats = synthetic_windows[CRISIS_FEATURE_COLS].dropna().values

    if len(real_feats) == 0 or len(synth_feats) == 0:
        return np.nan, "INSUFFICIENT_DATA"

    all_f = np.vstack([real_feats, synth_feats])
    mu, sd = all_f.mean(0), all_f.std(0)
    sd[sd == 0] = 1.0

    rz = (real_feats  - mu) / sd
    sz = (synth_feats - mu) / sd

    D        = cdist(sz, rz, metric="euclidean")
    min_dist = D.min(axis=1)
    precision= float(np.mean(min_dist <= eps))

    return precision, ("PASS" if precision >= 0.40 else "WARNING" if precision >= 0.25 else "FAIL")


# =============================================================================
# FULL CRISIS COVERAGE REPORT
# =============================================================================

def crisis_coverage_report(
    real_returns,
    synthetic_paths,
    model_name="model",
    window=20,
    eps_main=2.0,
    n_clusters=5,
    eps_grid=None,
):
    """
    Full crisis coverage analysis for one model.

    Returns
    ───────
    dict with:
        coverage_curve    : pd.DataFrame, coverage@eps for eps in grid
        archetype_table   : pd.DataFrame, coverage by crisis archetype
        summary           : dict of scalar metrics
    """
    if eps_grid is None:
        eps_grid = list(np.arange(0.5, 5.5, 0.5))

    real  = np.asarray(real_returns).reshape(-1)
    paths = np.asarray(synthetic_paths)

    tail_thr = float(np.quantile(real, 0.05))

    # Real windows
    real_windows   = extract_window_features(real, window=window, tail_threshold=tail_thr)
    crisis_windows = identify_crisis_windows(real_windows)

    # Synthetic windows (from all paths)
    synth_flat = paths.reshape(-1)
    synth_windows = extract_window_features(synth_flat[:min(len(synth_flat), 50000)],
                                            window=window, tail_threshold=tail_thr)

    # Coverage curve
    cov_curve = coverage_at_epsilon(crisis_windows, synth_windows, eps_grid=eps_grid)

    # Precision
    prec, prec_verdict = precision_at_epsilon(crisis_windows, synth_windows, eps=eps_main)

    # Archetype coverage
    crisis_typed, scaler, km = cluster_crisis_archetypes(crisis_windows, n_clusters=n_clusters)
    arch_table = archetype_coverage(crisis_typed, synth_windows, eps=eps_main)

    # Summary
    cov_at_main = float(
        cov_curve.loc[cov_curve["epsilon"].round(1) == round(eps_main, 1), "coverage"].values[0]
    ) if len(cov_curve) > 0 else np.nan

    n_arch_pass = int((arch_table["verdict"] == "PASS").sum())   if len(arch_table) > 0 else 0
    n_arch_fail = int((arch_table["verdict"] == "FAIL").sum())   if len(arch_table) > 0 else 0
    n_arch_warn = int((arch_table["verdict"] == "WARNING").sum())if len(arch_table) > 0 else 0

    # Precision trap flag
    precision_trap = (prec >= 0.70) and (cov_at_main < 0.50)

    summary = {
        "model":                  model_name,
        f"coverage_at_{eps_main}":cov_at_main,
        f"precision_at_{eps_main}":prec,
        "precision_verdict":      prec_verdict,
        "n_crisis_windows":       int(len(crisis_windows)),
        "n_synthetic_windows":    int(len(synth_windows)),
        "n_archetypes_tested":    int(len(arch_table)),
        "n_archetypes_pass":      n_arch_pass,
        "n_archetypes_fail":      n_arch_fail,
        "n_archetypes_warning":   n_arch_warn,
        "precision_trap_detected":precision_trap,
        "precision_trap_flag":    "WARNING: precision trap" if precision_trap else "OK",
        "overall_verdict": (
            "PASS"    if cov_at_main >= 0.65 and not precision_trap else
            "WARNING" if cov_at_main >= 0.50 else
            "FAIL"
        ),
    }

    return {
        "model":          model_name,
        "coverage_curve": cov_curve,
        "archetype_table":arch_table,
        "summary":        summary,
    }


def crisis_coverage_multi_model(real_returns, models_dict, window=20, n_clusters=5):
    """
    Run crisis coverage for multiple models.

    Returns summary DataFrame and per-model detail dicts.
    """
    summaries = []
    details   = {}

    for name, paths in models_dict.items():
        report = crisis_coverage_report(
            real_returns, paths,
            model_name=name,
            window=window,
            n_clusters=n_clusters,
        )
        summaries.append(report["summary"])
        details[name] = report

    return pd.DataFrame(summaries), details
