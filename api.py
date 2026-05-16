# =============================================================================
# FASTAPI BACKEND
# Exposes the tail-risk intelligence engine as a REST API.
#
# Run:
#   uvicorn api:app --reload --port 8000
#
# Endpoints:
#   GET  /                            health check
#   POST /risk                        model stress metrics only
#   POST /options-disagreement        full pipeline including options
#   GET  /regime/{ticker}             current regime score only (fast)
# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from risk_engine import (
    generate_crcwb_evt_paths,
    download_market_data,
    make_market_features,
    compute_lambda,
)
from options_engine import (
    get_option_expiries,
    fetch_put_chain,
    clean_put_chain,
    add_bs_metrics,
    vol_surface_summary,
)
from scoring import (
    add_model_physical_probabilities,
    score_model_market_disagreement,
    summarise_dashboard,
)


app = FastAPI(
    title="Tail-Risk Intelligence API",
    description=(
        "Compares model-implied physical downside risk with options-implied "
        "risk-neutral probabilities. Not a trading recommendations service."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class RiskRequest(BaseModel):
    ticker:  str = Field(default="SPY",   description="Ticker symbol")
    horizon: int = Field(default=20,      ge=5, le=60, description="Trading days")
    n_sims:  int = Field(default=5000,    ge=1000, le=20000, description="Simulation paths")

class OptionsRequest(RiskRequest):
    expiry: Optional[str] = Field(default=None, description="Option expiry date (YYYY-MM-DD)")


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    """Convert numpy scalars and NaN to JSON-safe Python floats."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else round(v, 6)
    except Exception:
        return None


def clean_dict(d):
    """Recursively make a dict JSON-serialisable."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [clean_dict(v) for v in d]
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating, float)):
        return safe_float(d)
    if isinstance(d, pd.Timestamp):
        return str(d)
    return d


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Tail-Risk Intelligence API",
        "note": (
            "Physical probability ≠ risk-neutral probability. "
            "Disagreements may reflect risk premia, not mispricing."
        ),
    }


@app.get("/regime/{ticker}", tags=["Regime"])
def get_regime(ticker: str):
    """
    Fast endpoint: current regime score λ(t) only.
    Does not run full simulation.
    """
    try:
        prices, returns, vix = download_market_data(ticker=ticker.upper())
        features = make_market_features(returns, vix, ticker=ticker.upper())
        lambda_t, details = compute_lambda(features)

        return {
            "ticker":          ticker.upper(),
            "lambda_t":        safe_float(lambda_t),
            "regime":          details["regime"],
            "vol_pct":         safe_float(details["vol_pct"]),
            "vix_pct":         safe_float(details["vix_pct"]),
            "dd_stress":       safe_float(details["dd_stress"]),
            "ret_stress":      safe_float(details["ret_stress"]),
            "raw_score":       safe_float(details["raw_score"]),
            "latest_date":     str(features.index[-1].date()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk", tags=["Risk"])
def run_risk(req: RiskRequest):
    """
    Generate CR-CWB-EVT paths and return model stress metrics.
    Does not fetch options data.
    """
    try:
        out = generate_crcwb_evt_paths(
            ticker=req.ticker.upper(),
            horizon=req.horizon,
            n_sims=req.n_sims,
        )

        return clean_dict({
            "ticker":          req.ticker.upper(),
            "horizon":         req.horizon,
            "n_sims":          req.n_sims,
            "spot":            out["spot"],
            "lambda_t":        out["lambda_t"],
            "regime":          out["lambda_details"]["regime"],
            "lambda_details":  out["lambda_details"],
            "stress_metrics":  out["stats"],
            "evt_fit":         out["evt_fit"],
            "top_narratives":  out["narratives"].to_dict(orient="records"),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/options-disagreement", tags=["Options"])
def run_options_disagreement(req: OptionsRequest):
    """
    Full pipeline: model paths + options chain + disagreement scoring.

    Returns:
    - Model stress metrics
    - Regime score
    - Options-implied metrics
    - Physical vs risk-neutral probability disagreement per strike
    - Dashboard summary

    Important: disagreements are informational, not trading signals.
    """
    try:
        # Model paths
        out    = generate_crcwb_evt_paths(
            ticker=req.ticker.upper(),
            horizon=req.horizon,
            n_sims=req.n_sims,
        )
        paths  = out["paths"]

        # Expiry selection
        expiries = get_option_expiries(req.ticker.upper())
        if not expiries:
            raise ValueError(f"No options data for {req.ticker.upper()}")

        expiry = req.expiry or expiries[min(1, len(expiries) - 1)]
        if expiry not in expiries:
            raise ValueError(f"Expiry {expiry} not available.")

        # Options chain
        raw_puts = fetch_put_chain(req.ticker.upper(), expiry=expiry)
        clean    = clean_put_chain(raw_puts)

        if len(clean) == 0:
            raise ValueError("No liquid puts after filtering.")

        bs_puts    = add_bs_metrics(clean)
        model_puts = add_model_physical_probabilities(bs_puts, paths)
        scored     = score_model_market_disagreement(model_puts)
        summary    = summarise_dashboard(scored)
        vol_summ   = vol_surface_summary(bs_puts)

        # Top rows
        top_cols = [
            "strike", "moneyness", "mid", "implied_vol",
            "physical_prob_ITM", "risk_neutral_prob_ITM",
            "probability_disagreement", "abs_probability_disagreement",
            "physical_expected_payoff", "payoff_minus_mid",
            "tail_disagreement_score", "interpretation", "confidence",
            "openInterest", "bid_ask_pct",
        ]
        top_cols = [c for c in top_cols if c in scored.columns]
        top_rows = scored[top_cols].head(20).to_dict(orient="records")

        return clean_dict({
            "ticker":              req.ticker.upper(),
            "horizon":             req.horizon,
            "expiry":              expiry,
            "spot":                out["spot"],
            "lambda_t":            out["lambda_t"],
            "regime":              out["lambda_details"]["regime"],
            "stress_metrics":      out["stats"],
            "vol_surface_summary": vol_summ,
            "disagreement_summary":summary,
            "top_disagreements":   top_rows,
            "disclaimer": (
                "Physical probability (model) differs from risk-neutral probability "
                "(options market). This gap may reflect risk premia, not mispricing. "
                "Not a trading recommendation."
            ),
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/expiries/{ticker}", tags=["Options"])
def list_expiries(ticker: str):
    """List available option expiry dates for a ticker."""
    try:
        expiries = get_option_expiries(ticker.upper())
        return {"ticker": ticker.upper(), "expiries": expiries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
