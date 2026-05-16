# =============================================================================
# FINREGIMEBENCH
# A benchmark for testing whether financial synthetic data is risk-valid.
#
# This is the public-facing research contribution.
# It evaluates any synthetic data generator across four validation layers
# and produces a single comparable leaderboard score.
#
# LEADERBOARD COLUMNS
# ───────────────────
# Model
# KS statistic                   (body realism — lower = better)
# Wasserstein distance            (body realism — lower = better)
# ACF(r²) match error             (volatility clustering — lower = better)
# VaR 5% verdict                  (PASS / FAIL)
# VaR 1% verdict                  (PASS / FAIL)
# ES severity ratio               (ideal = 1.0)
# Crisis Coverage@2               (higher = better)
# Crisis Precision@2              (higher = better)
# Tail mass ratio                 (ideal = 1.0)
# Aggregation-sensitive flag      (YES = bad)
# Walk-forward Brier score        (lower = better, optional)
# Overall Risk Validity Score     (0–100, composite)
#
# HOW TO USE
# ──────────
# frb = FinRegimeBench(real_returns)
# frb.add_model("GARCH-t",  garch_paths)
# frb.add_model("CR-CWB-EVT", crcwb_paths)
# frb.add_model("TimeGAN",  tgan_paths)
# leaderboard = frb.compute_leaderboard()
# frb.save_html("finregimebench_results.html")
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from metrics.body_realism   import compute_body_realism
from metrics.var_backtest   import var_es_backtest_panel, pathwise_var_stability
from metrics.crisis_coverage import crisis_coverage_report
from metrics.model_health   import compute_model_health, health_report_to_dataframe


# =============================================================================
# SCORING RUBRIC
# =============================================================================

def risk_validity_score(
    ks_stat,
    wasserstein,
    acf_sq_error,
    var_5pct_pass,
    var_1pct_pass,
    es_severity_ratio,
    coverage_at_2,
    precision_at_2,
    tail_mass_ratio,
    aggregation_sensitive,
    brier_score=None,
):
    """
    Compute overall Risk Validity Score (0–100).

    SCORING PHILOSOPHY:
    Body realism contributes 20% — necessary but not sufficient.
    VaR/ES validity contributes 35% — core risk metric.
    Crisis coverage contributes 30% — differentiator.
    Pathwise stability contributes 15% — robustness.

    Brier score bonus: if walk-forward is available, up to +10 points.
    """
    score = 0.0

    # ── Body realism (max 20 pts) ──────────────────────────────────────────
    ks_pts    = max(0, 10 - ks_stat * 200)          # 10 pts if KS < 0.05
    wass_pts  = max(0, 5  - wasserstein * 2000)      # 5 pts if Wass < 0.002
    acf_pts   = max(0, 5  - acf_sq_error * 50)       # 5 pts if ACF error < 0.10
    score += min(ks_pts + wass_pts + acf_pts, 20)

    # ── VaR/ES validity (max 35 pts) ──────────────────────────────────────
    score += 12 if var_5pct_pass else 0
    score += 12 if var_1pct_pass else 0

    # ES severity: 11 pts if ratio in [0.7, 1.5], decaying outside
    if es_severity_ratio is not None and not np.isnan(es_severity_ratio):
        es_pts = max(0, 11 - 11 * abs(np.log(max(es_severity_ratio, 0.01))))
        score += min(es_pts, 11)

    # ── Crisis coverage (max 30 pts) ──────────────────────────────────────
    cov_pts  = min(coverage_at_2 * 30, 20)        # 20 pts max for coverage
    prec_pts = min(precision_at_2 * 10, 10)       # 10 pts for precision

    # Tail mass ratio: ideal = 1.0
    if tail_mass_ratio is not None and not np.isnan(tail_mass_ratio):
        tail_pts = max(0, 10 - 10 * abs(np.log(max(tail_mass_ratio, 0.01))))
    else:
        tail_pts = 0

    score += min(cov_pts + prec_pts, 30)

    # ── Pathwise stability (max 15 pts) ───────────────────────────────────
    score += 15 if not aggregation_sensitive else 5

    # ── Walk-forward bonus (max 10 pts) ───────────────────────────────────
    if brier_score is not None and not np.isnan(brier_score):
        bss_pts = max(0, 10 - brier_score * 100)
        score += min(bss_pts, 10)

    return round(min(score, 100), 1)


# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class FinRegimeBench:
    """
    FinRegimeBench: a reproducible benchmark for financial synthetic data
    risk-validity evaluation.

    Usage:
        frb = FinRegimeBench(real_returns)
        frb.add_model("GARCH-t",     garch_paths)
        frb.add_model("CR-CWB-EVT",  crcwb_paths)
        frb.compute_leaderboard()
        frb.save_html("leaderboard.html")
    """

    def __init__(
        self,
        real_returns,
        train_fraction=0.80,
        window=20,
        n_crisis_clusters=5,
        name="FinRegimeBench",
    ):
        """
        Parameters
        ──────────
        real_returns    : 1D array of historical log returns
        train_fraction  : fraction used as training data for crisis coverage
        window          : window length for crisis feature extraction
        n_crisis_clusters: number of crisis archetypes
        name            : benchmark name (for reports)
        """
        self.real_returns   = np.asarray(real_returns).reshape(-1)
        self.train_fraction = train_fraction
        self.window         = window
        self.n_clusters     = n_crisis_clusters
        self.name           = name
        self.models         = {}       # {name: paths_array}
        self.results        = {}       # {name: result_dict}
        self.timestamp      = datetime.utcnow().isoformat()

        n = len(self.real_returns)
        self._r_train = self.real_returns[:int(n * train_fraction)]
        self._r_test  = self.real_returns[int(n * train_fraction):]

        print(f"FinRegimeBench initialised.")
        print(f"  Total returns: {n} | Train: {len(self._r_train)} | Test: {len(self._r_test)}")

    def add_model(self, name, synthetic_paths, brier_score=None):
        """
        Register a model for evaluation.

        Parameters
        ──────────
        name            : str, model identifier
        synthetic_paths : np.ndarray, shape (n_sims, horizon) or 1D
        brier_score     : float or None (from walk-forward calibration)
        """
        self.models[name] = {
            "paths":       np.asarray(synthetic_paths),
            "brier_score": brier_score,
        }
        print(f"  Registered: {name} | paths shape: {np.asarray(synthetic_paths).shape}")

    def evaluate_model(self, name):
        """Run all benchmark tests for one model."""
        m       = self.models[name]
        paths   = m["paths"]
        bs      = m["brier_score"]

        print(f"\n  Evaluating: {name}...")

        # Body realism
        body = compute_body_realism(self._r_test, paths, model_name=name)

        # VaR/ES backtest
        var_df = var_es_backtest_panel(
            self._r_test, paths, model_name=name,
            alphas=[0.10, 0.05, 0.025, 0.01]
        )

        def _verdict(alpha):
            row = var_df[var_df["alpha"] == alpha]
            return row["overall_verdict"].values[0] if len(row) > 0 else "UNKNOWN"

        def _es_ratio(alpha=0.05):
            row = var_df[var_df["alpha"] == alpha]
            return float(row["es_severity_ratio"].values[0]) if len(row) > 0 else np.nan

        var_5_pass = _verdict(0.05) == "PASS"
        var_1_pass = _verdict(0.01) == "PASS"
        es_ratio   = _es_ratio(0.05)

        # Pathwise stability
        stab = pathwise_var_stability(
            self._r_test, paths, model_name=name, alpha=0.05
        )
        aggregation_sensitive = bool(stab.get("aggregation_sensitive", False))

        # Crisis coverage
        crisis = crisis_coverage_report(
            self._r_train, paths,
            model_name=name,
            window=self.window,
            n_clusters=self.n_clusters,
        )
        cov_at_2  = float(crisis["summary"].get("coverage_at_2.0", np.nan))
        prec_at_2 = float(crisis["summary"].get("precision_at_2.0", np.nan))
        tmr       = float(body.get("tail_mass_ratio_0_5pct", np.nan))

        # Composite score
        score = risk_validity_score(
            ks_stat=float(body["ks_stat"]),
            wasserstein=float(body["wasserstein"]),
            acf_sq_error=float(body["acf_sq_returns_error"]),
            var_5pct_pass=var_5_pass,
            var_1pct_pass=var_1_pass,
            es_severity_ratio=es_ratio,
            coverage_at_2=cov_at_2 if not np.isnan(cov_at_2) else 0,
            precision_at_2=prec_at_2 if not np.isnan(prec_at_2) else 0,
            tail_mass_ratio=tmr,
            aggregation_sensitive=aggregation_sensitive,
            brier_score=bs,
        )

        result = {
            "model":                   name,
            "ks_stat":                 float(body["ks_stat"]),
            "wasserstein":             float(body["wasserstein"]),
            "acf_sq_error":            float(body["acf_sq_returns_error"]),
            "tail_mass_ratio":         tmr,
            "var_5pct_verdict":        "PASS" if var_5_pass else "FAIL",
            "var_1pct_verdict":        "PASS" if var_1_pass else "FAIL",
            "es_severity_ratio":       es_ratio,
            "crisis_coverage_at_2":    cov_at_2,
            "crisis_precision_at_2":   prec_at_2,
            "aggregation_sensitive":   aggregation_sensitive,
            "walk_forward_brier":      bs,
            "risk_validity_score":     score,
            "precision_trap":          bool(crisis["summary"].get("precision_trap_detected", False)),
            "n_archetypes_pass":       int(crisis["summary"].get("n_archetypes_pass", 0)),
            "n_archetypes_fail":       int(crisis["summary"].get("n_archetypes_fail", 0)),
        }

        self.results[name] = result
        print(f"    Risk Validity Score: {score:.1f}/100")
        return result

    def compute_leaderboard(self):
        """Run evaluation for all registered models and return ranked leaderboard."""
        print(f"\n{'='*60}")
        print(f"FINREGIMEBENCH EVALUATION")
        print(f"{'='*60}")

        for name in self.models:
            self.evaluate_model(name)

        lb = (
            pd.DataFrame(list(self.results.values()))
            .sort_values("risk_validity_score", ascending=False)
            .reset_index(drop=True)
        )
        lb.index = lb.index + 1  # Rank starts at 1

        print(f"\n{'='*60}")
        print("LEADERBOARD")
        print(f"{'='*60}")
        print(lb[["model", "risk_validity_score", "var_5pct_verdict",
                   "crisis_coverage_at_2", "aggregation_sensitive"]].to_string())

        return lb

    def save_html(self, path="finregimebench_results.html"):
        """Save a self-contained HTML leaderboard report."""
        lb = pd.DataFrame(list(self.results.values())).sort_values(
            "risk_validity_score", ascending=False
        ).reset_index(drop=True)
        lb.index = lb.index + 1

        def badge(val, good_val=None, invert=False):
            if isinstance(val, bool):
                color = "#E24B4A" if val else "#1D9E75"
                return f'<span style="background:{color};color:#fff;padding:2px 7px;border-radius:4px;font-size:0.8em;">{"YES" if val else "NO"}</span>'
            if isinstance(val, str):
                colors = {"PASS":"#1D9E75","FAIL":"#E24B4A","WARNING":"#EF9F27","UNKNOWN":"#888780"}
                c = colors.get(val, "#888780")
                return f'<span style="background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:0.8em;">{val}</span>'
            return f"{val}"

        rows_html = ""
        for rank, (_, row) in enumerate(lb.iterrows(), start=1):
            score = row["risk_validity_score"]
            score_color = "#1D9E75" if score >= 70 else "#EF9F27" if score >= 50 else "#E24B4A"
            rows_html += f"""
            <tr>
              <td><strong>#{rank}</strong></td>
              <td><strong>{row['model']}</strong></td>
              <td style="color:{score_color};font-weight:500;font-size:1.1em">{score:.1f}</td>
              <td>{row['ks_stat']:.4f}</td>
              <td>{row['wasserstein']:.5f}</td>
              <td>{row['tail_mass_ratio']:.3f}</td>
              <td>{badge(row['var_5pct_verdict'])}</td>
              <td>{badge(row['var_1pct_verdict'])}</td>
              <td>{row['es_severity_ratio']:.2f if not np.isnan(row['es_severity_ratio']) else 'N/A'}</td>
              <td>{row['crisis_coverage_at_2']:.2f if not np.isnan(row['crisis_coverage_at_2']) else 'N/A'}</td>
              <td>{badge(bool(row['aggregation_sensitive']))}</td>
              <td>{"—" if row['walk_forward_brier'] is None or np.isnan(float(row['walk_forward_brier'] or float('nan'))) else f"{float(row['walk_forward_brier']):.3f}"}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FinRegimeBench Leaderboard</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          max-width:1100px;margin:40px auto;padding:0 24px;color:#2c2c2a;line-height:1.6 }}
  h1   {{ font-size:1.8rem;font-weight:500;margin-bottom:4px }}
  h2   {{ font-size:1.1rem;font-weight:500;margin-top:32px;
          border-bottom:1px solid #e0dfd6;padding-bottom:6px }}
  table{{ border-collapse:collapse;width:100%;margin-top:12px;font-size:0.88rem }}
  th   {{ background:#f4f3ee;text-align:left;padding:8px 12px;
          font-weight:500;border-bottom:2px solid #d3d1c7 }}
  td   {{ padding:8px 12px;border-bottom:1px solid #e8e7e0;vertical-align:middle }}
  tr:hover td {{ background:#fafaf6 }}
  .subtitle {{ color:#888780;font-size:0.9rem;margin-bottom:24px }}
  .note {{ background:#fff8e1;border-left:3px solid #EF9F27;
           padding:12px 16px;border-radius:4px;margin:20px 0;font-size:0.88rem }}
</style>
</head>
<body>
<h1>FinRegimeBench</h1>
<div class="subtitle">
  Financial Synthetic Data Risk-Validity Benchmark &nbsp;|&nbsp;
  Generated: {self.timestamp}
</div>

<div class="note">
  <strong>Body realism (KS, Wasserstein) is necessary but not sufficient for risk validity.</strong>
  A model can have low KS and still fail VaR backtests or crisis coverage.
  The Risk Validity Score weights distributional realism at 20%, VaR/ES validity at 35%,
  crisis coverage at 30%, and pathwise stability at 15%.
</div>

<h2>Leaderboard</h2>
<table>
  <tr>
    <th>Rank</th>
    <th>Model</th>
    <th>Risk Validity Score</th>
    <th>KS stat</th>
    <th>Wasserstein</th>
    <th>Tail mass ratio</th>
    <th>VaR 5% verdict</th>
    <th>VaR 1% verdict</th>
    <th>ES severity ratio</th>
    <th>Crisis Coverage@2</th>
    <th>Aggregation sensitive</th>
    <th>Walk-fwd Brier</th>
  </tr>
  {rows_html}
</table>

<h2>Score interpretation</h2>
<ul>
  <li><strong>≥ 70</strong>: Production-grade. Model passes across all four validation layers.</li>
  <li><strong>50–69</strong>: Research-grade. Useful with caveats. Document failure modes.</li>
  <li><strong>&lt; 50</strong>: Experimental. Do not use for risk decisions without further investigation.</li>
</ul>

<h2>Methodology</h2>
<p>
  Each model is evaluated on historical data split into train ({int(self.train_fraction*100)}%) and test ({int((1-self.train_fraction)*100)}%) periods.
  Body realism uses KS test, Wasserstein distance, and ACF matching.
  VaR/ES validity uses Kupiec, Christoffersen, and ES severity tests.
  Crisis coverage uses Coverage@2 on historical crisis windows clustered into archetypes.
  Pathwise stability tests whether VaR validity holds across four aggregation modes.
</p>
<p>
  Walk-forward Brier score, when available, is computed over the full test period
  using expanding-window training — no lookahead bias.
</p>

</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)
        print(f"\nLeaderboard saved: {path}")
        return lb

    def to_json(self, path="finregimebench_results.json"):
        """Save raw results as JSON."""
        with open(path, "w") as f:
            json.dump(
                {"benchmark": self.name, "timestamp": self.timestamp, "results": self.results},
                f, indent=2, default=str
            )
        print(f"Results saved: {path}")
