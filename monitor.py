# =============================================================================
# DAILY MONITOR LOGGER
# Run this on a schedule (cron, GitHub Actions, etc.) to build a
# time-series database of tail-risk and disagreement metrics.
#
# Usage:
#   python monitor.py
#   python monitor.py --ticker QQQ --horizon 20 --n_sims 5000
# =============================================================================

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from risk_engine import generate_crcwb_evt_paths
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
from config import OUT_DIR_MONITOR


os.makedirs(OUT_DIR_MONITOR, exist_ok=True)
MONITOR_FILE = os.path.join(OUT_DIR_MONITOR, "daily_tailrisk_monitor.csv")


def run_daily_monitor(
    ticker="SPY",
    horizon=20,
    n_sims=5000,
    verbose=True,
):
    """
    Run the full pipeline and append one row to the daily monitor log.

    Captures:
    - Regime score and components
    - Model stress metrics (drawdown probs, VaR, ES)
    - Options disagreement summary
    - EVT tail parameters
    - Timestamp and configuration

    Returns the new row as a dict.
    """
    ts = datetime.now(timezone.utc).isoformat()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Daily monitor: {ticker}  |  horizon={horizon}d  |  {ts}")
        print(f"{'='*60}")

    # ── Risk engine ───────────────────────────────────────────────────────────
    if verbose:
        print("Running CR-CWB-EVT scenario generation...")

    out = generate_crcwb_evt_paths(ticker=ticker, horizon=horizon, n_sims=n_sims)

    paths         = out["paths"]
    stats         = out["stats"]
    lambda_t      = out["lambda_t"]
    lambda_details= out["lambda_details"]
    evt_fit       = out["evt_fit"]
    spot          = out["spot"]

    row = {
        "run_timestamp": ts,
        "ticker":        ticker,
        "horizon":       horizon,
        "n_sims":        n_sims,
        "spot":          spot,
        "lambda_t":      lambda_t,
        "regime":        lambda_details["regime"],
        **{f"lambda_{k}": v for k, v in lambda_details.items()
           if k not in ("regime",)},
        **{f"model_{k}": v for k, v in stats.items()},
        "evt_shape":     evt_fit["shape"],
        "evt_scale":     evt_fit["scale"],
        "evt_threshold": evt_fit["threshold"],
    }

    # ── Options engine ────────────────────────────────────────────────────────
    try:
        expiries = get_option_expiries(ticker)
        if expiries:
            expiry   = expiries[min(1, len(expiries)-1)]
            raw_puts = fetch_put_chain(ticker, expiry=expiry)
            clean    = clean_put_chain(raw_puts)
            bs_puts  = add_bs_metrics(clean)
            model_puts = add_model_physical_probabilities(bs_puts, paths)
            scored     = score_model_market_disagreement(model_puts)
            summary    = summarise_dashboard(scored)
            vol_summ   = vol_surface_summary(bs_puts)

            row.update({
                "options_expiry":           expiry,
                "options_n_liquid_puts":    summary["n_options"],
                "options_atm_iv":           vol_summ.get("atm_implied_vol"),
                "options_vol_skew_5pct":    vol_summ.get("vol_skew_5pct"),
                "options_top_disagreement": summary.get("top_probability_disagreement"),
                "options_mean_disagreement":summary.get("mean_probability_disagreement"),
                "options_max_abs_disagree": summary.get("max_abs_disagreement"),
                "options_n_model_more":     summary.get("n_model_sees_more_downside"),
                "options_n_market_more":    summary.get("n_options_price_more_downside"),
                "options_top_interpretation":summary.get("top_interpretation"),
            })

            if verbose:
                print(f"\n  λ(t)         = {lambda_t:.4f}  ({lambda_details['regime']})")
                print(f"  P(>5% DD)    = {stats['prob_drawdown_below_5pct']:.2%}")
                print(f"  VaR 5%       = {stats['var_5pct_loss']:.2%}")
                print(f"  ATM IV       = {vol_summ.get('atm_implied_vol', float('nan')):.2%}")
                print(f"  Top gap      = {summary.get('top_probability_disagreement', float('nan')):.3f}")

        else:
            row["options_expiry"] = "N/A"
            if verbose:
                print("  No options data available.")

    except Exception as e:
        row["options_error"] = str(e)
        if verbose:
            print(f"  Options engine error: {e}")

    # ── Append to log ─────────────────────────────────────────────────────────
    new_df = pd.DataFrame([row])

    if os.path.exists(MONITOR_FILE):
        existing = pd.read_csv(MONITOR_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(MONITOR_FILE, index=False)

    if verbose:
        print(f"\n  Appended to: {MONITOR_FILE}")
        print(f"  Total log rows: {len(combined)}")

    return row


def print_monitor_summary(n_days=10):
    """Print the last N rows of the monitor log."""
    if not os.path.exists(MONITOR_FILE):
        print("No monitor log found. Run monitor.py first.")
        return

    df = pd.read_csv(MONITOR_FILE)
    cols = [c for c in [
        "run_timestamp", "ticker", "lambda_t", "regime",
        "model_prob_drawdown_below_5pct",
        "model_var_5pct_loss",
        "options_atm_iv",
        "options_top_disagreement",
        "options_top_interpretation",
    ] if c in df.columns]

    print(f"\nMonitor log (last {n_days} rows):")
    print(df[cols].tail(n_days).to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail-Risk Daily Monitor")
    parser.add_argument("--ticker",   default="SPY",    type=str)
    parser.add_argument("--horizon",  default=20,       type=int)
    parser.add_argument("--n_sims",   default=5000,     type=int)
    parser.add_argument("--summary",  action="store_true",
                        help="Print recent log summary instead of running")
    args = parser.parse_args()

    if args.summary:
        print_monitor_summary()
    else:
        run_daily_monitor(
            ticker=args.ticker,
            horizon=args.horizon,
            n_sims=args.n_sims,
        )
