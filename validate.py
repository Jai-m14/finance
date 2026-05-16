#!/usr/bin/env python3
# =============================================================================
# TAILGUARD VALIDATION CLI
# Run full risk-validity analysis on any synthetic dataset.
#
# Usage:
#   python validate.py --real real_returns.csv --synthetic paths.csv
#   python validate.py --ticker SPY --run-engine
#
# Output: validation_report.html  metrics.json  diagnostics.csv
# =============================================================================

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

from metrics.body_realism   import compute_body_realism
from metrics.var_backtest   import var_es_backtest_panel, pathwise_var_stability
from metrics.crisis_coverage import crisis_coverage_report
from metrics.model_health   import compute_model_health, health_report_to_dataframe

OUT_DIR = "validation_output"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_returns(path):
    """Load a CSV of returns. Accepts single column or multi-column with 'date' index."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.set_index("date")
    arr = df.iloc[:, 0].dropna().values
    print(f"  Loaded {len(arr)} return observations from {path}")
    return arr


def load_paths(path):
    """Load a CSV of synthetic paths (rows = simulations, cols = time steps)."""
    df = pd.read_csv(path)
    arr = df.values
    print(f"  Loaded {arr.shape[0]} paths × {arr.shape[1]} steps from {path}")
    return arr


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_html_report(
    model_name,
    health_df,
    body_row,
    var_df,
    crisis_summary,
    run_ts,
    out_path,
):
    """Produce a standalone HTML validation report."""

    def status_badge(status):
        colors = {
            "PASS": "#1D9E75", "FAIL": "#E24B4A", "WARNING": "#EF9F27",
            "UNKNOWN": "#888780", "LOW": "#1D9E75", "MODERATE": "#EF9F27",
            "HIGH": "#E24B4A", "NARROW": "#1D9E75", "WIDE": "#E24B4A",
            "MEDIUM": "#EF9F27",
        }
        color = colors.get(status, "#888780")
        return (
            f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;">{status}</span>'
        )

    rows_html = ""
    for _, row in health_df.iterrows():
        badge = status_badge(str(row["status"]))
        rows_html += f"""
        <tr>
          <td style="font-weight:500">{row['check']}</td>
          <td>{badge}</td>
          <td>{row['detail']}</td>
        </tr>"""

    var_rows = ""
    if var_df is not None and len(var_df) > 0:
        for _, row in var_df.iterrows():
            badge = status_badge(str(row["overall_verdict"]))
            var_rows += f"""
            <tr>
              <td>{row['alpha']:.1%}</td>
              <td>{row['n_breaches']}</td>
              <td>{row['expected_breaches']:.1f}</td>
              <td>{row['obs_exp_ratio']:.2f}</td>
              <td>{status_badge(str(row['kupiec_verdict']))}</td>
              <td>{status_badge(str(row['independence_verdict']))}</td>
              <td>{badge}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TailGuard Validation Report — {model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 24px;
         color: #2c2c2a; line-height: 1.6; }}
  h1 {{ font-size: 1.6rem; font-weight: 500; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; font-weight: 500; margin-top: 32px;
        border-bottom: 1px solid #e0dfd6; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.9rem; }}
  th {{ background: #f4f3ee; text-align: left; padding: 8px 12px;
        font-weight: 500; border-bottom: 2px solid #d3d1c7; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e8e7e0; vertical-align: top; }}
  tr:hover td {{ background: #fafaf6; }}
  .meta {{ color: #888780; font-size: 0.85rem; margin-bottom: 24px; }}
  .warning-box {{ background: #fff8e1; border-left: 3px solid #EF9F27;
                  padding: 12px 16px; border-radius: 4px; margin: 16px 0;
                  font-size: 0.9rem; }}
  .key-finding {{ background: #f0faf6; border-left: 3px solid #1D9E75;
                  padding: 12px 16px; border-radius: 4px; margin: 8px 0; }}
</style>
</head>
<body>

<h1>TailGuard Validation Report</h1>
<div class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  Generated: {run_ts}
</div>

<div class="warning-box">
  <strong>Body realism PASS does not imply risk validity.</strong>
  A model can have low KS/Wasserstein and still fail Kupiec, ES severity,
  or crisis coverage tests. All four validation layers must be assessed.
</div>

<h2>Model Health Summary</h2>
<table>
  <tr><th>Check</th><th>Status</th><th>Detail</th></tr>
  {rows_html}
</table>

<h2>Body Realism (Layer 1)</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>KS statistic</td><td>{body_row.get('ks_stat', 'N/A'):.4f}</td></tr>
  <tr><td>KS p-value</td><td>{body_row.get('ks_pval', 'N/A'):.4f}</td></tr>
  <tr><td>Wasserstein distance</td><td>{body_row.get('wasserstein', 'N/A'):.6f}</td></tr>
  <tr><td>ACF(returns) error</td><td>{body_row.get('acf_returns_error', 'N/A'):.4f}</td></tr>
  <tr><td>ACF(r²) error</td><td>{body_row.get('acf_sq_returns_error', 'N/A'):.4f}</td></tr>
  <tr><td>Tail mass ratio (0.5%)</td><td>{body_row.get('tail_mass_ratio_0_5pct', 'N/A'):.3f}</td></tr>
</table>

<h2>VaR / ES Backtest (Layer 2)</h2>
<table>
  <tr>
    <th>α</th><th>Breaches</th><th>Expected</th><th>Obs/Exp</th>
    <th>Kupiec</th><th>Independence</th><th>Overall</th>
  </tr>
  {var_rows if var_rows else '<tr><td colspan="7">Not computed.</td></tr>'}
</table>

<h2>Crisis Coverage (Layer 3)</h2>
{"".join([
    f'<div class="key-finding"><strong>{k}:</strong> {v}</div>'
    for k, v in (crisis_summary or {}).items()
    if not k.startswith("_")
])}

<h2>Important Interpretation Notes</h2>
<ul>
  <li>Physical probabilities are estimates under the model's assumptions, not guaranteed forecasts.</li>
  <li>VaR backtests use a single train/test split. Results may vary with different splits.</li>
  <li>Crisis coverage measures whether the synthetic manifold overlaps historical crisis regimes.
      It does not guarantee coverage of future crises.</li>
  <li>Options disagreement may reflect risk premia, not model error.</li>
</ul>

</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)

    print(f"  Report saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_validation(
    real_returns,
    synthetic_paths,
    model_name="model",
    features=None,
    train_fraction=0.80,
):
    """
    Run the full four-layer validation suite.

    Parameters
    ──────────
    real_returns    : 1D array
    synthetic_paths : 2D array (n_sims × horizon) or 1D
    model_name      : str
    features        : pd.DataFrame or None
    train_fraction  : float, fraction used as training data

    Returns
    ───────
    dict with all results
    """
    print(f"\n{'='*60}")
    print(f"TAILGUARD VALIDATION: {model_name}")
    print(f"{'='*60}")

    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    real   = np.asarray(real_returns).reshape(-1)
    paths  = np.asarray(synthetic_paths)
    n_real = len(real)
    n_train= int(n_real * train_fraction)

    r_train = real[:n_train]
    r_test  = real[n_train:]

    print(f"\n  Train: {n_train} obs | Test: {len(r_test)} obs")

    # ── Layer 1: Body realism ─────────────────────────────────────────────────
    print("\n  Layer 1: Body realism...")
    body_row = compute_body_realism(r_test, paths, model_name=model_name)
    tail_mass_ratio = body_row.get("tail_mass_ratio_0_5pct")

    # ── Layer 2: VaR / ES backtest ────────────────────────────────────────────
    print("  Layer 2: VaR/ES backtest...")
    var_df = var_es_backtest_panel(r_test, paths, model_name=model_name)

    stability = pathwise_var_stability(r_test, paths, model_name=model_name, alpha=0.05)

    # ── Layer 3: Crisis coverage ──────────────────────────────────────────────
    print("  Layer 3: Crisis coverage...")
    crisis_report = crisis_coverage_report(r_train, paths, model_name=model_name)
    crisis_summary = crisis_report["summary"]

    # ── Model health ──────────────────────────────────────────────────────────
    print("  Computing model health...")
    health = compute_model_health(
        var_backtest_df=var_df,
        tail_mass_ratio=tail_mass_ratio,
        crisis_summary=crisis_summary,
        stability_result=stability,
        features=features,
        model_name=model_name,
    )
    health_df = health_report_to_dataframe(health)

    # ── Save outputs ──────────────────────────────────────────────────────────
    body_df      = pd.DataFrame([body_row])
    metrics_path = os.path.join(OUT_DIR, f"{model_name}_metrics.json")
    diag_path    = os.path.join(OUT_DIR, f"{model_name}_diagnostics.csv")
    report_path  = os.path.join(OUT_DIR, f"{model_name}_validation_report.html")
    var_path     = os.path.join(OUT_DIR, f"{model_name}_var_backtest.csv")

    # JSON metrics
    metrics_out = {
        "model":          model_name,
        "run_timestamp":  run_ts,
        "body_realism":   {k: v for k, v in body_row.items() if k != "_warning"},
        "health":         {
            "overall_confidence": health["overall_confidence"],
            "overall_score":      health["overall_confidence_score"],
            "n_pass":             health["n_pass"],
            "n_warning":          health["n_warning"],
            "n_fail":             health["n_fail"],
            "headline":           health["headline"],
        },
        "crisis_coverage": {
            k: v for k, v in crisis_summary.items()
            if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    var_df.to_csv(var_path, index=False)
    health_df.to_csv(diag_path, index=False)

    generate_html_report(
        model_name=model_name,
        health_df=health_df,
        body_row=body_row,
        var_df=var_df,
        crisis_summary={k: v for k, v in crisis_summary.items()
                        if isinstance(v, (int, float, str, bool))},
        run_ts=run_ts,
        out_path=report_path,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  OVERALL CONFIDENCE: {health['overall_confidence']}")
    print(f"  {health['headline']}")
    print(f"\n  Outputs:")
    for p in [metrics_path, diag_path, var_path, report_path]:
        print(f"    {p}")

    return {
        "health":         health,
        "health_df":      health_df,
        "body_realism":   body_row,
        "var_backtest":   var_df,
        "crisis_report":  crisis_report,
        "stability":      stability,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TailGuard risk-validity validation suite."
    )
    parser.add_argument("--real",       type=str, help="CSV of real returns")
    parser.add_argument("--synthetic",  type=str, help="CSV of synthetic paths (rows=sims, cols=steps)")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--run-engine", action="store_true",
                        help="Generate synthetic paths using CR-CWB-EVT engine (requires --ticker)")
    parser.add_argument("--ticker",     type=str, default="SPY")
    parser.add_argument("--horizon",    type=int, default=20)
    parser.add_argument("--n-sims",     type=int, default=5000)
    args = parser.parse_args()

    if args.run_engine:
        print(f"Generating CR-CWB-EVT paths for {args.ticker}...")
        from risk_engine import generate_crcwb_evt_paths
        out = generate_crcwb_evt_paths(args.ticker, horizon=args.horizon, n_sims=args.n_sims)
        real_ret = out["returns"][args.ticker].dropna().values
        synth_paths = out["paths"]
        model_name = f"CR-CWB-EVT_{args.ticker}"
        features = out["features"]
    elif args.real and args.synthetic:
        real_ret    = load_returns(args.real)
        synth_paths = load_paths(args.synthetic)
        model_name  = args.model_name
        features    = None
    else:
        print("Provide --real and --synthetic, or use --run-engine --ticker SPY")
        sys.exit(1)

    run_validation(
        real_returns=real_ret,
        synthetic_paths=synth_paths,
        model_name=model_name,
        features=features,
    )
