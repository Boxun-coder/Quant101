# AGENT.md - Statistical Arbitrage Project Execution Plan

## 1. Project Overview & Directives
**Objective:** Build a robust, end-to-end Statistical Arbitrage (Stat Arb) backtesting engine and research pipeline using daily CRSP equity data for S&P 500 constituents (2016-01 to 2024-12).
**Target Audience:** Autonomous AI Coding Agents (e.g., Aider, Cursor, Devin, AutoGPT).
**Core Directive:** You must strictly follow modular software engineering practices. Avoid look-ahead bias at all costs. Separate data processing, alpha generation, risk modeling, and backtest execution into distinct modules.

## 2. Dataset Context
**Input File Details:**
* **Format:** Standard CRSP output (CSV or Parquet).
* **Key Columns Available:**
    * `PERMNO`: Primary security identifier (Do NOT use `TICKER` as it is not point-in-time stable).
    * `date`: Trading date.
    * `PRC`: Closing price (Negative values indicate bid-ask midpoint if no trades occurred).
    * `VOL`: Trading volume.
    * `RET`: Daily return (includes dividends - USE THIS for alpha).
    * `RETX`: Daily return (excludes dividends).
    * `SHROUT`: Shares outstanding.
    * `sprtrn`: S&P 500 benchmark daily return.

## 3. Directory Structure
Agents must establish and populate the following directory structure:

    stat_arb_project/
    ├── data/
    │   ├── raw/                 # Put the raw CRSP dataset here
    │   └── clean/               # Output folder for processed matrices
    ├── src/
    │   ├── config.py            # Global parameters (dates, transaction costs)
    │   ├── data_pipeline.py     # CRSP data cleaner and matrix pivot logic
    │   ├── alphas.py            # Factor construction logic
    │   ├── risk_model.py        # Neutralization and PCA logic
    │   ├── portfolio.py         # Weight allocation
    │   └── backtester.py        # PnL simulation and metrics
    ├── run_pipeline.py          # Master execution script
    └── requirements.txt         # pandas, numpy, scipy, pyarrow

---

## 4. Execution Phases & Agent Instructions

### Phase 1: Data Engineering (`src/data_pipeline.py`)
**Goal:** Ingest raw CRSP data and convert it into clean, 2D time-series matrices (Dates as Index, PERMNO as Columns).
**Required Tasks:**
1.  **Read Data:** Load the raw CRSP file.
2.  **Fix Negative Prices:** Overwrite `PRC` using `abs(PRC)`.
3.  **Calculate Market Cap:** Create `MktCap = PRC * SHROUT`.
4.  **Filter/Align:** Ensure continuous time series. Forward-fill missing prices up to a limit (e.g., 5 days), but set `VOL` to 0 on those days.
5.  **Pivot:** Create individual DataFrames (matrices) for `PRC`, `RET`, `VOL`, and `MktCap`. Save these as `.parquet` files in `data/clean/`.

### Phase 2: Alpha Research (`src/alphas.py`)
**Goal:** Generate predictive statistical factors based on the cleaned matrices.
**Required Tasks:**
1.  **Short-Term Reversal (5-Day):**
    * Calculate rolling 5-day sum of `RET`.
    * Multiply by -1 (Reversal expects losers to win and winners to lose).
    * Cross-sectionally z-score the results daily.
2.  **Volume Shock:**
    * Calculate 20-day Simple Moving Average (SMA) of `VOL`.
    * Calculate ratio: `VOL_Shock = VOL_today / VOL_SMA_20`.
    * Cross-sectionally z-score the results daily.
3.  **Cross-Sectional Momentum (12M - 1M):**
    * Calculate 252-day cumulative return, lagging by 21 days (exclude the most recent month).
    * Cross-sectionally z-score the results daily.
4.  **Factor Aggregation:**
    * Create a method to combine these factors (e.g., equal weight: 33% Reversal, 33% Volume, 33% Momentum) to output a single `Raw_Alpha` matrix.

### Phase 3: Risk & Portfolio Construction (`src/risk_model.py` & `src/portfolio.py`)
**Goal:** Convert `Raw_Alpha` scores into tradable, market-neutral target weights.
**Required Tasks:**
1.  **Beta Neutralization:** Regress each stock's `RET` against `sprtrn` (rolling 60-day). Adjust `Raw_Alpha` so the portfolio's expected beta is 0.
2.  **Cross-Sectional Demeaning:** Ensure the sum of weights on any given day is exactly 0.0 (Market Neutral).
3.  **Leverage Scaling:** Scale the weights so the sum of the absolute weights (Gross Leverage) equals exactly 2.0 (i.e., 100% Long, 100% Short).
4.  **Output:** A `Target_Weights` matrix.

### Phase 4: Backtesting Engine (`src/backtester.py`)
**Goal:** Simulate historical performance securely, preventing look-ahead bias.
**Required Tasks:**
1.  **Weight Shifting (CRITICAL):** Shift the `Target_Weights` matrix by 1 row forward. Weights calculated at the end of Day $T$ can only capture the return on Day $T+1$.
    * Formula: `Daily_PnL = Shift(Target_Weights, 1) * RET`
2.  **Transaction Costs:** Calculate daily turnover: `Turnover = abs(Target_Weights - Target_Weights_yesterday)`.
    * Deduct 5 basis points (0.0005) per unit of turnover from the Daily PnL.
3.  **Metrics Calculation:** Compute the following and print a summary tear sheet:
    * Annualized Return (252 days)
    * Annualized Volatility
    * Sharpe Ratio (Return / Volatility)
    * Maximum Drawdown
    * Average Daily Turnover

## 5. Acceptance Criteria (Definition of Done)
* [ ] All `.py` files are created as per the directory structure.
* [ ] No `SettingWithCopyWarning` or look-ahead biases in pandas operations.
* [ ] The pipeline runs end-to-end via `python run_pipeline.py`.
* [ ] The backtester outputs a Sharpe Ratio > 1.0 after transaction costs.
* [ ] Portfolio is verifiably dollar-neutral every single day.
