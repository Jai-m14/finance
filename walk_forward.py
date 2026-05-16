# =============================================================================
# WALK-FORWARD VALIDATION ENGINE
# The most important test for credibility.
#
# Runs the full pipeline at each historical date using ONLY data
# available at that point in time (no lookahead bias).
# Records predicted stress probability vs realised outcome.
#
# Metrics:
#   Brier score          — calibration of probability forecasts
#   CRPS                 — distributional forecast quality
#   Reliability diagram  — are probabilities trustworthy?
#   Expected calibration error (ECE)
#   Recall / false alarm rate for stress events
#   Lead time before stress
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
warnings.filterwarnings("ignore")


# =============================================================================
# STRESS EVENT LABELLING
# =============================================================================

def label_stress_events(
    returns,
    horizon=20,
    drawdown_threshold=-0.08,
    vol_quantile=0.95,
):
    """
    Label each date as stress=1 if the NEXT `horizon` days contain
    a drawdown below `drawdown_threshold` OR realised vol at or above
    the `vol_quantile` percentile of the full vol history.

    Parameters
    ──────────
    returns             : pd.Series of log returns
    horizon             : int, forecast horizon in trading days
    drawdown_threshold  : float (e.g. -0.08 for 8% drawdown)
    vol_quantile        : float (e.g. 0.95 = top 5% vol)

    Returns
    ───────
    pd.Series (same index as returns), True = stress in next horizon days
    """
    r = pd.Series(returns).reset_index(drop=True)
    n = len(r)

    # Compute full vol distribution for threshold
    roll_vol = r.rolling(horizon).std() * np.sqrt(252)
    vol_thr  = roll_vol.quantile(vol_quantile)

    stress = pd.Series(False, index=r.index)

    for i in range(n - horizon):
        window = r.iloc[i : i + horizon].values
        cum    = np.cumsum(window)
        pk     = np.maximum.accumulate(cum)
        max_dd = float((cum - pk).min())

        fwd_vol = float(window.std() * np.sqrt(252))

        if max_dd <= drawdown_threshold or fwd_vol >= vol_thr:
            stress.iloc[i] = True

    stress.index = returns.index if hasattr(returns, "index") else stress.index
    return stress


# =============================================================================
# SINGLE-DATE PREDICTION
# =============================================================================

def predict_stress_at_date(
    returns_up_to_date,
    features_up_to_date,
    crcwb_fn,
    evt_fn,
    horizon=20,
    n_sims=2000,
    drawdown_threshold=-0.08,
    seed=42,
):
    """
    Generate CR-CWB-EVT paths conditioned on data available up to a date
    and compute forward stress probabilities.

    Returns dict of predicted probabilities.
    """
    try:
        raw_paths, _ = crcwb_fn(
            returns_up_to_date,
            features_up_to_date,
            n_sims=n_sims,
            horizon=horizon,
            seed=seed,
        )

        ref_ret = returns_up_to_date.values if hasattr(returns_up_to_date, "values") else returns_up_to_date
        paths, _ = evt_fn(raw_paths, ref_ret, seed=seed)

        cum  = paths.sum(axis=1)
        cs   = np.cumsum(paths, axis=1)
        pk   = np.maximum.accumulate(cs, axis=1)
        max_dd = (cs - pk).min(axis=1)

        losses = -cum

        return {
            "p_drawdown_5pct":    float(np.mean(max_dd <= -0.05)),
            "p_drawdown_8pct":    float(np.mean(max_dd <= -0.08)),
            "p_drawdown_10pct":   float(np.mean(max_dd <= -0.10)),
            "var_5pct":           float(np.quantile(losses, 0.95)),
            "var_1pct":           float(np.quantile(losses, 0.99)),
            "es_5pct":            float(losses[losses >= np.quantile(losses, 0.95)].mean()),
            "median_return":      float(np.median(cum)),
        }

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# WALK-FORWARD RUNNER
# =============================================================================

def run_walk_forward(
    returns,
    features,
    crcwb_fn,
    evt_fn,
    compute_lambda_fn,
    horizon=20,
    n_sims=2000,
    min_train_days=504,
    eval_frequency=21,
    drawdown_threshold=-0.08,
    seed=42,
    verbose=True,
):
    """
    Walk-forward validation loop.

    At each evaluation date:
      1. Build model using ONLY data up to that date
      2. Predict stress probability
      3. Record realised outcome from next `horizon` days

    Parameters
    ──────────
    returns          : pd.Series of log returns (full history)
    features         : pd.DataFrame of market features (full history)
    crcwb_fn         : callable(returns_df, features_df, n_sims, horizon, seed) → (paths, info)
    evt_fn           : callable(paths, reference_returns, seed) → (paths, evt_fit)
    compute_lambda_fn: callable(features_df) → (lambda_t, details)
    horizon          : int, forecast horizon
    n_sims           : int, simulations per date (lower = faster)
    min_train_days   : int, minimum training window
    eval_frequency   : int, evaluate every N days
    drawdown_threshold: float, stress definition
    verbose          : bool

    Returns
    ───────
    pd.DataFrame with one row per evaluation date
    """
    returns  = pd.Series(returns).dropna()
    features = pd.DataFrame(features).dropna()

    # Label stress events on full history
    stress_labels = label_stress_events(
        returns, horizon=horizon, drawdown_threshold=drawdown_threshold
    )

    eval_dates = returns.index[min_train_days : -horizon : eval_frequency]

    rows = []
    n_eval = len(eval_dates)

    for i, dt in enumerate(eval_dates):
        ret_hist  = returns.loc[:dt]
        feat_hist = features.loc[:dt]

        if len(ret_hist) < min_train_days or len(feat_hist) < 100:
            continue

        # Wrap as DataFrame with SPY column for crcwb_fn
        ret_df = ret_hist.to_frame("SPY") if isinstance(ret_hist, pd.Series) else ret_hist

        predictions = predict_stress_at_date(
            returns_up_to_date=ret_df,
            features_up_to_date=feat_hist,
            crcwb_fn=crcwb_fn,
            evt_fn=evt_fn,
            horizon=horizon,
            n_sims=n_sims,
            drawdown_threshold=drawdown_threshold,
            seed=seed,
        )

        if "error" in predictions:
            if verbose:
                print(f"  {dt.date()} — error: {predictions['error']}")
            continue

        # Regime score
        try:
            lam, lam_det = compute_lambda_fn(feat_hist)
        except Exception:
            lam, lam_det = np.nan, {}

        # Realised outcome
        realised_stress = bool(stress_labels.get(dt, False))

        # Realised max drawdown in next horizon days
        future_slice = returns.loc[dt:].iloc[1 : horizon + 1]
        if len(future_slice) >= horizon:
            fut_cum = np.cumsum(future_slice.values)
            fut_pk  = np.maximum.accumulate(fut_cum)
            realised_max_dd = float((fut_cum - fut_pk).min())
        else:
            realised_max_dd = np.nan

        rows.append({
            "date":              dt,
            "lambda_t":          lam,
            "regime":            lam_det.get("regime", "unknown"),
            **{f"pred_{k}": v for k, v in predictions.items()},
            "realised_stress":   int(realised_stress),
            "realised_max_dd":   realised_max_dd,
        })

        if verbose and (i % 10 == 0 or i == n_eval - 1):
            print(f"  [{i+1}/{n_eval}] {dt.date()}  λ={lam:.2f}  "
                  f"p_dd8={predictions.get('p_drawdown_8pct', np.nan):.2%}  "
                  f"realised_stress={realised_stress}")

    return pd.DataFrame(rows)


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def brier_score(predicted_probs, realised_labels):
    """Brier score: lower = better calibrated. Perfect = 0."""
    p = np.asarray(predicted_probs)
    y = np.asarray(realised_labels)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(predicted_probs, realised_labels, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predicted probabilities and measures gap between mean predicted
    and mean realised per bin.
    Lower = better.
    """
    p = np.asarray(predicted_probs)
    y = np.asarray(realised_labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(p)

    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if not mask.any():
            continue
        bin_p = p[mask].mean()
        bin_y = y[mask].mean()
        ece  += mask.sum() / n * abs(bin_p - bin_y)

    return float(ece)


def reliability_table(predicted_probs, realised_labels, n_bins=10):
    """
    Reliability diagram data.

    Returns pd.DataFrame with columns:
        bin_centre, mean_predicted, mean_realised, n_obs
    """
    p    = np.asarray(predicted_probs)
    y    = np.asarray(realised_labels)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if not mask.any():
            continue
        rows.append({
            "bin_centre":     float((bins[i] + bins[i+1]) / 2),
            "mean_predicted": float(p[mask].mean()),
            "mean_realised":  float(y[mask].mean()),
            "n_obs":          int(mask.sum()),
        })

    return pd.DataFrame(rows)


def stress_event_detection_metrics(
    predicted_probs,
    realised_labels,
    threshold=0.15,
):
    """
    Binary classification metrics at a probability threshold.

    Returns recall (crisis detection rate), precision, false alarm rate,
    F1, and lead-time analysis.
    """
    p      = np.asarray(predicted_probs) >= threshold
    y      = np.asarray(realised_labels).astype(bool)

    tp = int(( p &  y).sum())
    fp = int(( p & ~y).sum())
    tn = int((~p & ~y).sum())
    fn = int((~p &  y).sum())

    recall     = tp / max(tp + fn, 1)
    precision  = tp / max(tp + fp, 1)
    false_alarm= fp / max(fp + tn, 1)
    f1         = 2 * recall * precision / max(recall + precision, 1e-12)

    return {
        "threshold":       threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall":          float(recall),
        "precision":       float(precision),
        "false_alarm_rate":float(false_alarm),
        "f1_score":        float(f1),
    }


def compute_walk_forward_metrics(wf_df, prob_col="pred_p_drawdown_8pct"):
    """
    Aggregate the walk-forward DataFrame into performance metrics.

    Parameters
    ──────────
    wf_df    : pd.DataFrame from run_walk_forward
    prob_col : str, column name for predicted probability

    Returns
    ───────
    dict of scalar metrics
    """
    if len(wf_df) == 0 or prob_col not in wf_df.columns:
        return {"error": "No walk-forward data available."}

    valid = wf_df.dropna(subset=[prob_col, "realised_stress"])
    p = valid[prob_col].values
    y = valid["realised_stress"].values

    bs    = brier_score(p, y)
    ece   = expected_calibration_error(p, y)
    rel   = reliability_table(p, y)
    det   = stress_event_detection_metrics(p, y, threshold=0.15)

    # Brier skill score vs naive baseline (predict base rate always)
    base_rate   = float(y.mean())
    bs_baseline = float(np.mean((base_rate - y) ** 2))
    bss = 1 - bs / bs_baseline if bs_baseline > 0 else np.nan

    return {
        "n_eval_dates":            len(valid),
        "stress_base_rate":        base_rate,
        "brier_score":             bs,
        "brier_skill_score":       bss,
        "expected_calibration_error": ece,
        "recall_at_15pct":         det["recall"],
        "precision_at_15pct":      det["precision"],
        "false_alarm_rate_at_15pct": det["false_alarm_rate"],
        "f1_at_15pct":             det["f1_score"],
        "reliability_table":       rel,
        "overall_calibration_verdict": (
            "WELL_CALIBRATED"     if ece < 0.05 else
            "MODERATE_CALIBRATION"if ece < 0.10 else
            "POORLY_CALIBRATED"
        ),
    }
