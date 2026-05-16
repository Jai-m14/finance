# =============================================================================
# TAIL-RISK INTELLIGENCE DASHBOARD — CONFIGURATION
# =============================================================================

# Data
DEFAULT_TICKER     = "SPY"
DEFAULT_START_DATE = "2000-01-01"
TICKERS_UNIVERSE   = ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]

# Risk-free rate (update periodically)
RISK_FREE_RATE = 0.045

# Simulation defaults
DEFAULT_HORIZON_DAYS = 20
DEFAULT_N_SIMS       = 10000
RANDOM_SEED          = 42

# Drawdown monitoring levels
DRAWDOWN_LEVELS = [-0.05, -0.08, -0.10]

# Options liquidity filters
MIN_OPEN_INTEREST = 100
MIN_VOLUME        = 1
MAX_BID_ASK_PCT   = 0.20
MIN_OPTION_MID    = 0.05

# Disagreement thresholds
# A gap below this is not reported as material
MIN_PROB_GAP = 0.03

# Crisis-weighted bootstrap settings
CRISIS_WEIGHT      = 2.0
REGIME_BANDWIDTH   = 1.0

# EVT tail correction settings
EVT_REPLACE_ALPHA   = 0.01
EVT_THRESHOLD_ALPHA = 0.05

# Regime gate sigmoid parameters
# λ = sigmoid(-8 * (raw_score - 0.60))
LAMBDA_STEEPNESS = 8.0
LAMBDA_CENTER    = 0.60

# Regime score feature weights
LAMBDA_WEIGHTS = {
    "vol_pct":   0.35,
    "vix_pct":   0.30,
    "dd_stress": 0.20,
    "ret_stress":0.15,
}

# Output paths
OUT_DIR_MONITOR   = "daily_monitor_logs"
OUT_DIR_BACKTEST  = "backtest_outputs"
OUT_DIR_SCENARIOS = "scenario_narratives"
