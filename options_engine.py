# =============================================================================
# OPTIONS ENGINE
# Fetches the options chain and computes Black-Scholes risk-neutral metrics.
#
# KEY FRAMING:
# The output of this module is the MARKET-IMPLIED risk-neutral distribution.
# This is not the same as the physical distribution from risk_engine.py.
# The gap between them may reflect risk premia, not mispricing.
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
warnings.filterwarnings("ignore")

from config import (
    RISK_FREE_RATE,
    MIN_OPEN_INTEREST,
    MIN_VOLUME,
    MAX_BID_ASK_PCT,
    MIN_OPTION_MID,
)


# =============================================================================
# BLACK-SCHOLES CORE
# =============================================================================

def bs_put_price(S, K, T, r, sigma):
    """Analytical Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return float(max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_call_price(S, K, T, r, sigma):
    """Analytical Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_risk_neutral_prob(S, K, T, r, sigma):
    """
    Risk-neutral probability P_Q(S_T < K) under Black-Scholes.

    This is N(-d2), not the physical probability of loss.
    It incorporates the risk premium embedded in option prices.
    """
    if T <= 0 or sigma <= 0:
        return float(S < K)
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(-d2))


def implied_vol_put(price, S, K, T, r):
    """
    Solve for implied volatility from an observed put price using Brent's method.
    Returns NaN if no solution exists (e.g. price below intrinsic value).
    """
    intrinsic = max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic * 0.999:
        return np.nan
    try:
        iv = brentq(
            lambda sigma: bs_put_price(S, K, T, r, sigma) - price,
            1e-5, 8.0,
            xtol=1e-6,
            maxiter=200,
        )
        return float(iv)
    except Exception:
        return np.nan


def bs_delta(S, K, T, r, sigma):
    """Put delta: dP/dS. Negative for puts."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1)


def bs_vega(S, K, T, r, sigma):
    """Vega: sensitivity to 1% move in implied vol."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T) * 0.01)


# =============================================================================
# OPTIONS CHAIN FETCHING
# =============================================================================

def get_option_expiries(ticker="SPY"):
    """Return list of available expiry dates from yfinance."""
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


def fetch_spot(ticker="SPY"):
    """Fetch current spot price."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def fetch_put_chain(ticker="SPY", expiry=None):
    """
    Fetch the put options chain for a given expiry.
    If expiry is None, uses the nearest available expiry.

    Returns raw DataFrame with mid, bid-ask, and time-to-expiry columns added.
    """
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)

    if not expiries:
        raise ValueError(f"No options data available for {ticker}.")

    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        raise ValueError(f"Expiry {expiry} not available. Choose from: {expiries}")

    chain = tk.option_chain(expiry)
    puts  = chain.puts.copy()

    spot     = fetch_spot(ticker) or float(puts["lastPrice"].median())
    expiry_dt= pd.to_datetime(expiry)
    today    = pd.Timestamp.today(tz=None).normalize()
    T        = max((expiry_dt - today).days / 365.0, 1 / 365.0)

    puts["ticker"]    = ticker
    puts["expiry"]    = expiry
    puts["spot"]      = spot
    puts["T"]         = T
    puts["mid"]       = (puts["bid"] + puts["ask"]) / 2
    puts["bid_ask"]   = puts["ask"] - puts["bid"]
    puts["moneyness"] = puts["strike"] / spot

    puts["bid_ask_pct"] = np.where(
        puts["mid"] > 0,
        puts["bid_ask"] / puts["mid"],
        np.nan
    )

    return puts


def fetch_call_chain(ticker="SPY", expiry=None):
    """Fetch call chain (same structure as puts)."""
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise ValueError(f"No options data for {ticker}.")
    if expiry is None:
        expiry = expiries[0]

    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()

    spot      = fetch_spot(ticker) or float(calls["lastPrice"].median())
    expiry_dt = pd.to_datetime(expiry)
    today     = pd.Timestamp.today(tz=None).normalize()
    T         = max((expiry_dt - today).days / 365.0, 1 / 365.0)

    calls["ticker"]    = ticker
    calls["expiry"]    = expiry
    calls["spot"]      = spot
    calls["T"]         = T
    calls["mid"]       = (calls["bid"] + calls["ask"]) / 2
    calls["bid_ask"]   = calls["ask"] - calls["bid"]
    calls["moneyness"] = calls["strike"] / spot
    calls["bid_ask_pct"] = np.where(
        calls["mid"] > 0,
        calls["bid_ask"] / calls["mid"],
        np.nan
    )
    return calls


# =============================================================================
# LIQUIDITY FILTERING
# =============================================================================

def clean_put_chain(puts):
    """
    Apply liquidity filters.
    Retains only puts with acceptable bid-ask spread, open interest,
    and positive mid price.
    """
    df = puts.copy()

    for col in ["openInterest", "volume"]:
        if col not in df.columns:
            df[col] = 0

    mask = (
        (df["mid"]           >= MIN_OPTION_MID)  &
        (df["bid"]           >  0)               &
        (df["ask"]           >  0)               &
        (df["ask"]           >= df["bid"])        &
        (df["bid_ask_pct"]   <= MAX_BID_ASK_PCT)  &
        (df["openInterest"].fillna(0) >= MIN_OPEN_INTEREST)
    )
    return df[mask].copy()


# =============================================================================
# BLACK-SCHOLES METRICS
# =============================================================================

def add_bs_metrics(puts, r=RISK_FREE_RATE):
    """
    Compute implied vol and risk-neutral ITM probability for each option.
    Drops rows where IV cannot be solved.

    Returns enriched DataFrame ready for disagreement scoring.
    """
    df = puts.copy()

    ivs, rn_probs, bs_prices, deltas, vegas = [], [], [], [], []

    for _, row in df.iterrows():
        S   = float(row["spot"])
        K   = float(row["strike"])
        T   = float(row["T"])
        mid = float(row["mid"])

        iv = implied_vol_put(mid, S, K, T, r)

        if np.isnan(iv):
            rn_probs.append(np.nan)
            bs_prices.append(np.nan)
            deltas.append(np.nan)
            vegas.append(np.nan)
        else:
            rn_probs.append(bs_put_risk_neutral_prob(S, K, T, r, iv))
            bs_prices.append(bs_put_price(S, K, T, r, iv))
            deltas.append(bs_delta(S, K, T, r, iv))
            vegas.append(bs_vega(S, K, T, r, iv))

        ivs.append(iv)

    df["implied_vol"]            = ivs
    df["risk_neutral_prob_ITM"]  = rn_probs
    df["bs_price_from_iv"]       = bs_prices
    df["delta"]                  = deltas
    df["vega_per_1pct_vol"]      = vegas

    return df.dropna(subset=["implied_vol", "risk_neutral_prob_ITM"])


# =============================================================================
# VOLATILITY SURFACE SUMMARY
# =============================================================================

def vol_surface_summary(puts_with_bs):
    """
    Summarise the implied vol surface across strikes.
    Useful for the dashboard header metrics.
    """
    df  = puts_with_bs.copy()
    atm = df.iloc[(df["moneyness"] - 1.0).abs().argsort()[:1]]

    atm_iv   = float(atm["implied_vol"].values[0]) if len(atm) > 0 else np.nan
    otm_5pct = df[df["moneyness"] <= 0.95]["implied_vol"].mean()
    skew     = (otm_5pct - atm_iv) if not np.isnan(otm_5pct) else np.nan

    return {
        "atm_implied_vol":       atm_iv,
        "otm_5pct_implied_vol":  float(otm_5pct) if not np.isnan(otm_5pct) else None,
        "vol_skew_5pct":         float(skew)     if not np.isnan(skew)     else None,
        "n_liquid_puts":         int(len(df)),
        "min_moneyness":         float(df["moneyness"].min()),
        "max_moneyness":         float(df["moneyness"].max()),
    }
