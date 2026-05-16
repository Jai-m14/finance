# =============================================================================
# RISK ENGINE
# CR-CWB-EVT: Current-Regime Crisis-Weighted Bootstrap + EVT Tail Correction
# + Regime Gate λ(t)
#
# This is the physical scenario generator.
# It produces forward paths calibrated to the current market regime.
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.stats import genpareto, percentileofscore
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from config import (
    DEFAULT_START_DATE,
    TICKERS_UNIVERSE,
    DRAWDOWN_LEVELS,
    CRISIS_WEIGHT,
    REGIME_BANDWIDTH,
    EVT_REPLACE_ALPHA,
    EVT_THRESHOLD_ALPHA,
    LAMBDA_STEEPNESS,
    LAMBDA_CENTER,
    LAMBDA_WEIGHTS,
    RANDOM_SEED,
)


# =============================================================================
# DATA LAYER
# =============================================================================

def download_market_data(ticker="SPY", start=DEFAULT_START_DATE):
    """
    Download price history for the target ticker and supporting universe.
    Returns prices, log returns, and VIX series aligned to the same index.
    """
    universe = list(set([ticker] + TICKERS_UNIVERSE))
    raw = yf.download(universe, start=start, auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(ticker)

    raw = raw.dropna(how="all").ffill().dropna()

    prices  = raw.copy()
    returns = np.log(prices / prices.shift(1)).dropna()

    vix = (
        prices["^VIX"].reindex(returns.index).ffill()
        if "^VIX" in prices.columns
        else pd.Series(index=returns.index, data=np.nan)
    )

    return prices, returns, vix


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def max_drawdown_from_returns(x):
    x   = np.asarray(x).reshape(-1)
    cum = np.cumsum(x)
    pk  = np.maximum.accumulate(cum)
    return float(np.min(cum - pk))


def make_market_features(returns, vix, ticker="SPY"):
    """
    Build the feature matrix used for regime detection and CR-CWB weighting.
    Only uses information available at each point in time (no lookahead).
    """
    if ticker not in returns.columns:
        raise ValueError(f"{ticker} not in returns columns: {list(returns.columns)}")

    r  = returns[ticker]
    df = pd.DataFrame(index=returns.index)

    for w in [5, 10, 20, 60]:
        df[f"ret_{w}d"]      = r.rolling(w).sum()
        df[f"vol_{w}d"]      = r.rolling(w).std() * np.sqrt(252)
        df[f"drawdown_{w}d"] = r.rolling(w).apply(max_drawdown_from_returns, raw=True)

    # Cross-asset features
    for other in ["QQQ", "IWM", "TLT", "GLD"]:
        if other in returns.columns and other != ticker:
            col = other.lower()
            df[f"{col}_ret_20d"] = returns[other].rolling(20).sum()
            df[f"{col}_vol_20d"] = returns[other].rolling(20).std() * np.sqrt(252)
            df[f"{ticker.lower()}_{col}_corr_60d"] = r.rolling(60).corr(returns[other])

    # VIX features
    df["vix_level"]    = vix
    df["vix_change_5d"]  = vix.diff(5)
    df["vix_change_20d"] = vix.diff(20)
    df["vix_z_252d"]   = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

    # Loss momentum
    df["loss_5d"]  = (-r).rolling(5).sum()
    df["loss_20d"] = (-r).rolling(20).sum()

    return df.dropna()


# =============================================================================
# REGIME GATE λ(t)
# =============================================================================

def compute_lambda(features):
    """
    Compute current-regime crisis intensity score λ(t) ∈ [0, 1].

    Uses percentile-rank of current vol, VIX, drawdown, and return
    against full history available at computation time.

    Higher λ → more crisis-like → combined model shifts toward
    CR-CWB-EVT branch.

    Returns
    ───────
    lambda_t : float
    details  : dict of component scores
    """
    latest = features.iloc[-1]
    w      = LAMBDA_WEIGHTS

    vol_pct   = percentileofscore(features["vol_20d"].dropna(),  latest["vol_20d"])   / 100
    vix_pct   = percentileofscore(features["vix_level"].dropna(),latest["vix_level"]) / 100
    dd_stress = 1 - percentileofscore(features["drawdown_20d"].dropna(), latest["drawdown_20d"]) / 100
    ret_stress= 1 - percentileofscore(features["ret_20d"].dropna(),      latest["ret_20d"])       / 100

    raw_score = (
        w["vol_pct"]   * vol_pct +
        w["vix_pct"]   * vix_pct +
        w["dd_stress"] * dd_stress +
        w["ret_stress"]* ret_stress
    )

    lambda_t = float(1 / (1 + np.exp(-LAMBDA_STEEPNESS * (raw_score - LAMBDA_CENTER))))

    details = {
        "vol_pct":    vol_pct,
        "vix_pct":    vix_pct,
        "dd_stress":  dd_stress,
        "ret_stress": ret_stress,
        "raw_score":  raw_score,
        "lambda_t":   lambda_t,
        "regime":     (
            "crisis"   if lambda_t > 0.70 else
            "fragile"  if lambda_t > 0.30 else
            "calm"
        ),
    }

    return lambda_t, details


# =============================================================================
# EVT TAIL CORRECTION
# =============================================================================

def fit_evt_tail(reference_returns, threshold_alpha=EVT_THRESHOLD_ALPHA):
    """
    Fit a Generalised Pareto Distribution to the left tail of reference_returns.
    Falls back to a wider threshold if too few exceedances.
    """
    losses     = -np.asarray(reference_returns).reshape(-1)
    threshold  = np.quantile(losses, 1 - threshold_alpha)
    exceedances= losses[losses > threshold] - threshold

    if len(exceedances) < 20:
        threshold_alpha = 0.10
        threshold       = np.quantile(losses, 1 - threshold_alpha)
        exceedances     = losses[losses > threshold] - threshold

    shape, loc, scale = genpareto.fit(exceedances, floc=0)

    return {
        "threshold_alpha": float(threshold_alpha),
        "threshold":       float(threshold),
        "shape":           float(shape),
        "loc":             float(loc),
        "scale":           float(scale),
        "n_exceedances":   int(len(exceedances)),
    }


def evt_correct_paths(
    paths,
    reference_returns,
    replace_alpha=EVT_REPLACE_ALPHA,
    threshold_alpha=EVT_THRESHOLD_ALPHA,
    seed=RANDOM_SEED,
):
    """
    Replace the deepest synthetic losses with GPD-sampled tail draws.

    This corrects the block-bootstrap's tendency to underweight
    extreme events by anchoring the tail to the empirical GPD fit.
    """
    rng  = np.random.default_rng(seed)
    paths= np.asarray(paths).copy()
    flat = paths.reshape(-1)
    losses = -flat

    evt    = fit_evt_tail(reference_returns, threshold_alpha=threshold_alpha)
    cutoff = np.quantile(losses, 1 - replace_alpha)
    mask   = losses >= cutoff
    n      = int(mask.sum())

    if n > 0:
        sampled = genpareto.rvs(
            c=evt["shape"], loc=0, scale=evt["scale"],
            size=n, random_state=rng
        )
        corrected = flat.copy()
        corrected[mask] = -(evt["threshold"] + sampled)
        paths = corrected.reshape(paths.shape)

    return paths, evt


# =============================================================================
# HISTORICAL BLOCK LIBRARY
# =============================================================================

def build_block_library(returns, features, ticker="SPY", block_len=20):
    """
    Slice the historical return series into non-overlapping blocks
    and attach the market features that preceded each block.
    """
    r      = returns[ticker].reindex(features.index).dropna()
    common = features.index.intersection(r.index)
    r, feat= r.loc[common], features.loc[common]
    vals   = r.values

    blocks, rows = [], []
    for i in range(60, len(vals) - block_len + 1):
        b     = vals[i : i + block_len]
        f_row = feat.iloc[i]
        rows.append({
            "block_id":        len(blocks),
            "start_date":      common[i],
            "end_date":        common[i + block_len - 1],
            "block_return":    float(b.sum()),
            "block_vol_ann":   float(b.std() * np.sqrt(252)),
            "block_max_dd":    max_drawdown_from_returns(b),
            "block_worst_1d":  float(b.min()),
            "past_vol_20d":    f_row["vol_20d"],
            "past_drawdown_20d": f_row["drawdown_20d"],
            "past_ret_20d":    f_row["ret_20d"],
            "past_vix_level":  f_row["vix_level"],
            "past_vix_z_252d": f_row["vix_z_252d"],
        })
        blocks.append(b)

    return np.asarray(blocks), pd.DataFrame(rows)


# =============================================================================
# CR-CWB SAMPLER
# =============================================================================

def current_regime_crisis_weighted_bootstrap(
    returns,
    features,
    ticker="SPY",
    n_sims=10000,
    horizon=20,
    crisis_weight=CRISIS_WEIGHT,
    regime_bandwidth=REGIME_BANDWIDTH,
    seed=RANDOM_SEED,
):
    """
    Sample synthetic paths by:
      1. Measuring distance from current regime to each historical block
      2. Up-weighting blocks from crisis regimes
      3. Drawing paths with weights proportional to regime similarity × crisis flag

    This ensures the synthetic distribution is anchored to the current
    regime while oversampling crisis episodes for tail realism.
    """
    rng = np.random.default_rng(seed)

    blocks, block_df = build_block_library(
        returns, features, ticker=ticker, block_len=horizon
    )

    latest      = features.iloc[-1]
    regime_cols = [
        "past_vol_20d", "past_drawdown_20d",
        "past_ret_20d", "past_vix_level", "past_vix_z_252d",
    ]
    current_vec = np.array([
        latest["vol_20d"],  latest["drawdown_20d"],
        latest["ret_20d"],  latest["vix_level"],
        latest["vix_z_252d"],
    ], dtype=float)

    X       = block_df[regime_cols].values.astype(float)
    scaler  = StandardScaler()
    Xz      = scaler.fit_transform(X)
    curr_z  = scaler.transform(current_vec.reshape(1, -1))[0]

    dist          = np.linalg.norm(Xz - curr_z, axis=1)
    regime_weight = np.exp(-0.5 * (dist / regime_bandwidth) ** 2)

    vol_thr    = block_df["block_vol_ann"].quantile(0.95)
    dd_thr     = block_df["block_max_dd"].quantile(0.05)
    is_crisis  = (
        (block_df["block_vol_ann"] >= vol_thr) |
        (block_df["block_max_dd"]  <= dd_thr)
    ).values

    crisis_mult = np.where(is_crisis, crisis_weight, 1.0)
    weights     = regime_weight * crisis_mult
    weights     = weights / weights.sum() if weights.sum() > 0 else np.ones(len(block_df)) / len(block_df)

    chosen = rng.choice(len(blocks), size=n_sims, replace=True, p=weights)
    paths  = blocks[chosen]

    block_df["regime_distance"]      = dist
    block_df["regime_weight"]        = regime_weight
    block_df["is_crisis_block"]      = is_crisis
    block_df["final_sample_weight"]  = weights

    return paths, block_df


# =============================================================================
# GARCH FORWARD SIMULATION (optional branch)
# =============================================================================

def simulate_garch_t_paths(spy_returns, horizon=20, n_sims=10000):
    """
    GARCH(1,1)-t forward simulation.
    Returns None if arch library not available.
    """
    if not ARCH_AVAILABLE:
        return None

    r_pct = spy_returns.dropna() * 100
    am    = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res   = am.fit(disp="off")
    fc    = res.forecast(horizon=horizon, method="simulation", simulations=n_sims)
    sim   = fc.simulations.values[-1] / 100.0
    return sim


# =============================================================================
# PATH STATISTICS
# =============================================================================

def future_path_stats(paths):
    """
    Compute forward-looking risk metrics from a (n_sims × horizon) path matrix.
    All metrics are expressed in log-return space.
    """
    paths = np.asarray(paths)
    cum   = paths.sum(axis=1)
    cs    = np.cumsum(paths, axis=1)
    pk    = np.maximum.accumulate(cs, axis=1)
    max_dd= (cs - pk).min(axis=1)

    out = {}
    for level in DRAWDOWN_LEVELS:
        label = int(abs(level) * 100)
        out[f"prob_drawdown_below_{label}pct"] = float(np.mean(max_dd <= level))

    losses = -cum
    for alpha in [0.05, 0.01]:
        var = np.quantile(losses, 1 - alpha)
        es  = float(losses[losses >= var].mean()) if (losses >= var).any() else np.nan
        out[f"var_{int(alpha*100)}pct_loss"] = float(var)
        out[f"es_{int(alpha*100)}pct_loss"]  = es

    out.update({
        "median_return":        float(np.median(cum)),
        "mean_return":          float(np.mean(cum)),
        "p05_return":           float(np.quantile(cum, 0.05)),
        "p01_return":           float(np.quantile(cum, 0.01)),
        "median_max_drawdown":  float(np.median(max_dd)),
        "p05_max_drawdown":     float(np.quantile(max_dd, 0.05)),
        "p01_max_drawdown":     float(np.quantile(max_dd, 0.01)),
        "realised_vol_ann":     float(paths.std(axis=1).mean() * np.sqrt(252)),
    })

    return out


# =============================================================================
# SCENARIO NARRATIVES
# =============================================================================

def extract_worst_scenario_narratives(paths, block_info=None, top_n=10):
    """
    Identify the worst synthetic paths and describe their characteristics.
    If block_info is provided, attach the source historical dates.
    """
    paths = np.asarray(paths)
    cum   = paths.sum(axis=1)
    cs    = np.cumsum(paths, axis=1)
    pk    = np.maximum.accumulate(cs, axis=1)
    max_dd= (cs - pk).min(axis=1)

    worst_idx = np.argsort(cum)[:top_n]
    rows = []

    for rank, idx in enumerate(worst_idx, start=1):
        path = paths[idx]
        row  = {
            "rank":          rank,
            "path_index":    int(idx),
            "total_return":  float(cum[idx]),
            "max_drawdown":  float(max_dd[idx]),
            "worst_1d":      float(path.min()),
            "worst_5d":      float(pd.Series(path).rolling(5).sum().min()),
            "vol_ann":       float(path.std() * np.sqrt(252)),
            "negative_days": int((path < 0).sum()),
        }
        if block_info is not None and idx < len(block_info):
            bi = block_info.iloc[idx]
            row["source_block_start"] = str(bi.get("start_date", ""))
            row["source_block_end"]   = str(bi.get("end_date", ""))
            row["source_is_crisis"]   = bool(bi.get("is_crisis_block", False))
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_crcwb_evt_paths(
    ticker="SPY",
    horizon=20,
    n_sims=10000,
    seed=RANDOM_SEED,
):
    """
    Full pipeline: download → features → regime gate → CR-CWB → EVT.

    Returns a dict containing everything needed by the dashboard
    and scoring engine.

    Parameters
    ──────────
    ticker  : str
    horizon : int, trading days
    n_sims  : int
    seed    : int

    Returns
    ───────
    dict with keys:
        prices, returns, features, vix
        paths, raw_paths, block_info, evt_fit
        lambda_t, lambda_details
        stats, narratives
        spot
    """
    prices, returns, vix = download_market_data(ticker=ticker)
    features = make_market_features(returns, vix, ticker=ticker)

    lambda_t, lambda_details = compute_lambda(features)

    raw_paths, block_info = current_regime_crisis_weighted_bootstrap(
        returns, features, ticker=ticker,
        n_sims=n_sims, horizon=horizon,
        crisis_weight=CRISIS_WEIGHT,
        regime_bandwidth=REGIME_BANDWIDTH,
        seed=seed,
    )

    evt_paths, evt_fit = evt_correct_paths(
        raw_paths,
        reference_returns=returns[ticker].dropna().values,
        replace_alpha=EVT_REPLACE_ALPHA,
        threshold_alpha=EVT_THRESHOLD_ALPHA,
        seed=seed,
    )

    path_stats = future_path_stats(evt_paths)
    narratives = extract_worst_scenario_narratives(evt_paths, block_info, top_n=10)

    spot = float(prices[ticker].dropna().iloc[-1])

    return {
        "prices":         prices,
        "returns":        returns,
        "features":       features,
        "vix":            vix,
        "paths":          evt_paths,
        "raw_paths":      raw_paths,
        "block_info":     block_info,
        "evt_fit":        evt_fit,
        "lambda_t":       lambda_t,
        "lambda_details": lambda_details,
        "stats":          path_stats,
        "narratives":     narratives,
        "spot":           spot,
    }
