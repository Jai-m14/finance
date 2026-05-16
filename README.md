# Tail-Risk Intelligence Dashboard

A deployable dashboard that compares **model-implied physical downside risk**
with **options-implied risk-neutral downside probability** to monitor
tail-risk disagreements across strikes and market regimes.

---

## What this is

A crisis-aware scenario generator (CR-CWB-EVT) produces forward paths
anchored to the current market regime. For each simulated path, we compute
P(S_T < K) — the physical probability of finishing below each strike.

We compare this to P_Q(S_T < K) derived from live options prices via
Black-Scholes implied volatility — the market's risk-neutral view.

Where these two views disagree materially, the dashboard flags the size,
direction, liquidity quality, and historical context of that disagreement.

---

## What this is not

This is **not** an arbitrage engine or a trading signal generator.

Differences between physical and risk-neutral probabilities may reflect:

- Volatility risk premium (market pays a premium to be protected)
- Crash-risk premium (deep OTM puts are systematically expensive)
- Liquidity premium
- Dealer hedging flows
- Model uncertainty in either direction

The dashboard is designed for **risk monitoring and model validation**,
not deterministic crisis prediction or guaranteed trading signals.

---

## Architecture

```
tailrisk_app/
├── app.py              Streamlit dashboard (deployable UI)
├── api.py              FastAPI backend (REST endpoints)
├── risk_engine.py      CR-CWB-EVT scenario generator
├── options_engine.py   Options chain + Black-Scholes metrics
├── scoring.py          Disagreement scoring and interpretation
├── monitor.py          Daily automated logger (run on schedule)
├── config.py           All parameters in one place
├── requirements.txt
└── README.md
```

---

## Quick start

### Local Streamlit dashboard

```bash
cd tailrisk_app
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

### FastAPI backend

```bash
cd tailrisk_app
uvicorn api:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API explorer.

### Daily monitor (run on schedule)

```bash
python monitor.py --ticker SPY --horizon 20 --n_sims 5000
```

View recent log:

```bash
python monitor.py --summary
```

---

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to share.streamlit.io
3. Connect your repo, set `app.py` as the entry point
4. Deploy

---

## Key endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/regime/{ticker}` | Current regime score λ(t) only (fast) |
| POST | `/risk` | Model stress metrics (no options) |
| POST | `/options-disagreement` | Full pipeline with disagreement scoring |
| GET | `/expiries/{ticker}` | List available option expiry dates |

---

## Model methodology

**CR-CWB (Current-Regime Crisis-Weighted Bootstrap)**

1. Build a library of all 20-day historical return blocks
2. Measure distance between current market features and each block
3. Up-weight blocks from crisis regimes (high vol, deep drawdown)
4. Sample paths proportional to regime similarity × crisis weight

**EVT Tail Correction**

The bootstrap understates the probability of extreme losses.
We replace the deepest 1% of synthetic losses with draws from a
Generalised Pareto Distribution fitted to historical tail exceedances.

**Regime Gate λ(t)**

A sigmoid-transformed composite of:
- Realised vol percentile (35%)
- VIX level percentile (30%)
- Drawdown stress (20%)
- Return stress (15%)

λ(t) → 0: calm, RCHS body-realism dominant
λ(t) → 1: crisis, CR-CWB-EVT tail coverage dominant

---

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RISK_FREE_RATE` | 0.045 | Update periodically |
| `DEFAULT_HORIZON_DAYS` | 20 | Trading days |
| `DEFAULT_N_SIMS` | 10000 | Simulation paths |
| `CRISIS_WEIGHT` | 2.0 | Crisis block upweight factor |
| `MIN_PROB_GAP` | 0.03 | Minimum gap to flag as material |
| `MAX_BID_ASK_PCT` | 0.20 | Options liquidity filter |

---

## Important interpretation note

> Physical probability P(S_T < K) and risk-neutral probability P_Q(S_T < K)
> are different quantities derived from different measures. A difference
> between them is expected and normal. It becomes interesting when it is
> unusually large, directionally consistent across strikes, and observed
> in liquid options. Even then, it reflects an information signal — not a
> guaranteed profit opportunity.
