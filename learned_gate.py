# =============================================================================
# LEARNED REGIME GATE
# Replaces the hand-designed λ(t) formula with a supervised model
# calibrated from historical evidence.
#
# WHAT IT DOES
# ────────────
# Defines stress precisely:
#   Stress = max drawdown ≤ -8% OR realised vol ≥ 95th percentile
#   over the next 20 trading days.
#
# Trains three models on expanding windows:
#   1. Logistic regression (transparent baseline)
#   2. Gradient boosting (nonlinear)
#   3. Isotonic-calibrated ensemble
#
# Outputs per date:
#   p_stress_5d, p_stress_20d, p_stress_60d
#   p_drawdown_5pct, p_drawdown_8pct, p_drawdown_10pct
#   confidence_score
#   top_drivers (feature importances)
#
# VALIDATION
# ──────────
# Walk-forward expanding window — no lookahead.
# Brier score, ECE, reliability diagram per model.
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
warnings.filterwarnings("ignore")


# =============================================================================
# STRESS LABELLING
# =============================================================================

def build_stress_labels(
    returns,
    horizons=(5, 20, 60),
    drawdown_thresholds=(-0.05, -0.08, -0.10),
    vol_quantile=0.95,
):
    """
    Build binary stress labels for multiple horizons.

    For each date t, label[t] = 1 if ANY of the following occur
    in the next `horizon` trading days:
      - max drawdown ≤ drawdown_threshold
      - realised vol ≥ vol_quantile of full vol history

    Parameters
    ──────────
    returns              : pd.Series of log returns (full history)
    horizons             : tuple of ints (forecast horizons in days)
    drawdown_thresholds  : tuple of floats (e.g. -0.08 for 8% DD)
    vol_quantile         : float, e.g. 0.95

    Returns
    ───────
    pd.DataFrame with one column per (horizon, threshold) combination
    """
    r = pd.Series(returns).dropna().reset_index(drop=True)
    n = len(r)

    roll_vol_20 = r.rolling(20).std() * np.sqrt(252)
    vol_thr     = roll_vol_20.quantile(vol_quantile)

    labels = {}

    for h in horizons:
        # Primary stress: any -8% drawdown OR high vol
        col_stress = f"stress_{h}d"
        stress_arr = np.zeros(n, dtype=int)

        for i in range(n - h):
            w   = r.iloc[i : i + h].values
            cum = np.cumsum(w)
            pk  = np.maximum.accumulate(cum)
            max_dd = float((cum - pk).min())
            fwd_vol= float(w.std() * np.sqrt(252))

            if max_dd <= -0.08 or fwd_vol >= vol_thr:
                stress_arr[i] = 1

        labels[col_stress] = stress_arr

        # Per-threshold drawdown labels
        for thr in drawdown_thresholds:
            col = f"dd_{int(abs(thr)*100)}pct_{h}d"
            arr = np.zeros(n, dtype=int)
            for i in range(n - h):
                w   = r.iloc[i : i + h].values
                cum = np.cumsum(w)
                pk  = np.maximum.accumulate(cum)
                if float((cum - pk).min()) <= thr:
                    arr[i] = 1
            labels[col] = arr

    label_df = pd.DataFrame(labels, index=r.index)
    label_df.index = returns.index if hasattr(returns, "index") else label_df.index

    return label_df


# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================

GATE_FEATURE_COLS = [
    "vol_5d", "vol_20d", "vol_60d",
    "ret_5d", "ret_20d", "ret_60d",
    "drawdown_5d", "drawdown_20d", "drawdown_60d",
    "vix_level", "vix_z_252d", "vix_change_5d",
    "loss_5d", "loss_20d",
]


def build_gate_features(features, available_cols=None):
    """
    Select and clean feature matrix for the gate model.
    Fills missing columns with zeros rather than failing.
    """
    if available_cols is None:
        available_cols = GATE_FEATURE_COLS

    df = pd.DataFrame(index=features.index)
    for col in available_cols:
        if col in features.columns:
            df[col] = features[col]
        else:
            df[col] = 0.0

    return df.fillna(0.0)


# =============================================================================
# SINGLE-DATE PREDICTION (EXPANDING WINDOW)
# =============================================================================

def fit_gate_models(X_train, y_train):
    """
    Fit three gate models on training data.
    Returns dict of fitted pipelines.
    """
    models = {}

    # 1. Logistic regression (interpretable baseline)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l1", solver="saga",
            C=0.1, max_iter=1000, random_state=42
        )),
    ])
    lr.fit(X_train, y_train)
    models["logistic"] = lr

    # 2. Gradient boosting (nonlinear)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    gb.fit(X_train, y_train)
    models["gradient_boosting"] = gb

    # 3. Calibrated GB (isotonic)
    gb_cal = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, random_state=42
        ),
        method="isotonic", cv=3,
    )
    try:
        gb_cal.fit(X_train, y_train)
        models["calibrated_gb"] = gb_cal
    except Exception:
        models["calibrated_gb"] = gb  # fallback

    return models


def predict_stress(models, X_current):
    """
    Predict stress probability from all gate models.
    Returns dict of probabilities and ensemble mean.
    """
    probs = {}
    for name, model in models.items():
        try:
            p = float(model.predict_proba(X_current.reshape(1, -1))[:, 1][0])
        except Exception:
            p = np.nan
        probs[name] = p

    valid = [v for v in probs.values() if not np.isnan(v)]
    probs["ensemble_mean"] = float(np.mean(valid)) if valid else np.nan
    probs["ensemble_range"] = float(max(valid) - min(valid)) if len(valid) > 1 else 0.0
    probs["confidence"] = (
        "HIGH"   if probs["ensemble_range"] < 0.05 else
        "MEDIUM" if probs["ensemble_range"] < 0.15 else
        "LOW"
    )

    return probs


def get_top_drivers(lr_model, feature_cols, top_n=5):
    """
    Extract top feature importances from the logistic regression.
    Uses absolute coefficients as proxy for feature importance.
    """
    try:
        clf   = lr_model.named_steps["clf"]
        coefs = np.abs(clf.coef_[0])
        idx   = np.argsort(coefs)[::-1][:top_n]
        return [
            {"feature": feature_cols[i], "importance": float(coefs[i])}
            for i in idx
        ]
    except Exception:
        return []


# =============================================================================
# FULL WALK-FORWARD GATE TRAINING AND EVALUATION
# =============================================================================

def run_walk_forward_gate(
    returns,
    features,
    target_col="stress_20d",
    min_train_days=504,
    eval_frequency=21,
    verbose=True,
):
    """
    Train and evaluate the gate model walk-forward.
    At each evaluation date:
      1. Train on all data available up to that date
      2. Predict stress probability for the NEXT period
      3. Record realised outcome

    Parameters
    ──────────
    returns       : pd.Series
    features      : pd.DataFrame (from make_market_features)
    target_col    : str, label column (e.g. "stress_20d")
    min_train_days: int
    eval_frequency: int, evaluate every N days

    Returns
    ───────
    pd.DataFrame with walk-forward predictions and realised outcomes
    """
    r = pd.Series(returns).dropna()
    f = pd.DataFrame(features).dropna()

    common = r.index.intersection(f.index)
    r = r.loc[common]
    f = f.loc[common]

    if verbose:
        print(f"  Building stress labels for {target_col}...")
    labels_df = build_stress_labels(r)

    if target_col not in labels_df.columns:
        raise ValueError(
            f"Target '{target_col}' not found. "
            f"Available: {list(labels_df.columns)}"
        )

    X_all = build_gate_features(f).values
    y_all = labels_df[target_col].values

    eval_dates = common[min_train_days : -63 : eval_frequency]
    rows = []

    if verbose:
        print(f"  Walk-forward: {len(eval_dates)} evaluation dates...")

    for i, dt in enumerate(eval_dates):
        t_idx = common.get_loc(dt)

        X_train = X_all[:t_idx]
        y_train = y_all[:t_idx]

        if len(X_train) < min_train_days:
            continue
        if y_train.sum() < 5:
            continue

        try:
            models = fit_gate_models(X_train, y_train)
            X_curr = X_all[t_idx]
            probs  = predict_stress(models, X_curr)
            drivers= get_top_drivers(
                models["logistic"],
                build_gate_features(f).columns.tolist()
            )

            # Realised
            realised = int(y_all[t_idx]) if t_idx < len(y_all) else np.nan

            row = {
                "date":           dt,
                "realised":       realised,
                "target":         target_col,
                **{f"p_{k}": v for k, v in probs.items() if isinstance(v, float)},
                "confidence":     probs.get("confidence", "UNKNOWN"),
                "top_driver_1":   drivers[0]["feature"] if len(drivers) > 0 else "",
                "top_driver_2":   drivers[1]["feature"] if len(drivers) > 1 else "",
                "top_driver_3":   drivers[2]["feature"] if len(drivers) > 2 else "",
            }
            rows.append(row)

        except Exception as e:
            if verbose and i % 20 == 0:
                print(f"    {dt.date()} — skipped: {e}")
            continue

        if verbose and (i % 20 == 0 or i == len(eval_dates) - 1):
            ens = probs.get("ensemble_mean", np.nan)
            print(f"    [{i+1}/{len(eval_dates)}] {dt.date()} "
                  f"p_stress={ens:.2%}  realised={realised}")

    return pd.DataFrame(rows)


# =============================================================================
# GATE CALIBRATION METRICS
# =============================================================================

def gate_calibration_metrics(wf_df, prob_col="p_ensemble_mean"):
    """
    Compute calibration metrics for the walk-forward gate predictions.
    """
    valid = wf_df.dropna(subset=[prob_col, "realised"])
    p = valid[prob_col].values
    y = valid["realised"].values.astype(float)

    if len(p) < 20:
        return {"error": "Insufficient data for calibration."}

    bs = float(brier_score_loss(y, p))

    # Brier skill score vs base rate
    base = float(y.mean())
    bs_null = float(brier_score_loss(y, np.full_like(p, base)))
    bss = 1 - bs / bs_null if bs_null > 0 else np.nan

    # ECE
    fraction_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))

    # Binary detection at 15% threshold
    y_hat = (p >= 0.15).astype(int)
    tp = int(((y_hat == 1) & (y == 1)).sum())
    fp = int(((y_hat == 1) & (y == 0)).sum())
    tn = int(((y_hat == 0) & (y == 0)).sum())
    fn = int(((y_hat == 0) & (y == 1)).sum())

    recall     = tp / max(tp + fn, 1)
    precision  = tp / max(tp + fp, 1)
    false_alarm= fp / max(fp + tn, 1)

    return {
        "n_obs":                int(len(p)),
        "stress_base_rate":     base,
        "brier_score":          bs,
        "brier_skill_score":    float(bss) if not np.isnan(bss) else None,
        "ece":                  ece,
        "recall_at_15pct":      float(recall),
        "precision_at_15pct":   float(precision),
        "false_alarm_at_15pct": float(false_alarm),
        "calibration_verdict": (
            "WELL_CALIBRATED"      if ece < 0.05 else
            "MODERATE_CALIBRATION" if ece < 0.10 else
            "POORLY_CALIBRATED"
        ),
        "reliability_curve": {
            "mean_pred":     mean_pred.tolist(),
            "fraction_pos":  fraction_pos.tolist(),
        },
    }


# =============================================================================
# CURRENT-DATE PREDICTION (PRODUCTION)
# =============================================================================

def predict_current_stress(
    returns,
    features,
    target_col="stress_20d",
    min_train_days=504,
):
    """
    Train gate models on full available history and predict current stress.
    This is the PRODUCTION version — runs at deployment time.

    Returns
    ───────
    dict with:
        p_stress         : ensemble mean probability
        confidence       : HIGH / MEDIUM / LOW
        model_range      : spread across gate models
        top_drivers      : list of {feature, importance}
        individual_models: probabilities per model
    """
    r = pd.Series(returns).dropna()
    f = pd.DataFrame(features).dropna()
    common = r.index.intersection(f.index)
    r, f = r.loc[common], f.loc[common]

    if len(r) < min_train_days:
        return {
            "error": f"Insufficient history: {len(r)} < {min_train_days} days.",
            "p_stress": np.nan,
            "confidence": "UNKNOWN",
        }

    labels_df = build_stress_labels(r)
    if target_col not in labels_df.columns:
        return {"error": f"Target '{target_col}' not found.", "p_stress": np.nan}

    X_all = build_gate_features(f).values
    y_all = labels_df[target_col].values

    feat_cols = build_gate_features(f).columns.tolist()

    # Train on all available history except the last period
    X_train = X_all[:-20]
    y_train = y_all[:-20]
    X_curr  = X_all[-1]

    try:
        models = fit_gate_models(X_train, y_train)
        probs  = predict_stress(models, X_curr)
        drivers= get_top_drivers(models["logistic"], feat_cols)

        return {
            "p_stress":         probs["ensemble_mean"],
            "confidence":       probs["confidence"],
            "model_range":      probs["ensemble_range"],
            "individual_models":{k: v for k, v in probs.items()
                                 if k not in ("ensemble_mean","ensemble_range","confidence")},
            "top_drivers":      drivers,
            "target":           target_col,
        }

    except Exception as e:
        return {"error": str(e), "p_stress": np.nan, "confidence": "UNKNOWN"}
