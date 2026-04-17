from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
SRC_DIR = PROJECT_ROOT / "src"

RAW_DATA_FILE = RAW_DIR / "SP500_to202412.csv"

PRICE_FILE = CLEAN_DIR / "prc.parquet"
RETURN_FILE = CLEAN_DIR / "ret.parquet"
VOLUME_FILE = CLEAN_DIR / "vol.parquet"
MARKET_CAP_FILE = CLEAN_DIR / "mktcap.parquet"
BENCHMARK_FILE = CLEAN_DIR / "sprtrn.parquet"
RAW_ALPHA_FILE = CLEAN_DIR / "raw_alpha.parquet"
TARGET_WEIGHTS_FILE = CLEAN_DIR / "target_weights.parquet"
RESULTS_MATRIX_FILE = CLEAN_DIR / "results_matrix.parquet"
FINAL_REPORT_FILE = PROJECT_ROOT / "Final_Report.pdf"

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"

FORWARD_FILL_LIMIT = 5
BETA_LOOKBACK_DAYS = 60
REVERSAL_WINDOW_DAYS = 5
VOLUME_SMA_DAYS = 20
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
GROSS_LEVERAGE_TARGET = 2.0
TRANSACTION_COST_BPS = 5.0
TRADING_DAYS_PER_YEAR = 252

FACTOR_TAIL_FRACTIONS = {
    "reversal_5d": 0.05,
    "volume_shock": 0.02,
    "momentum_12m_1m": 0.02,
}

FACTOR_SMOOTHING_WINDOWS = {
    "reversal_5d": 20,
    "volume_shock": 60,
    "momentum_12m_1m": 120,
}

FACTOR_COMBINATION_WEIGHTS = {
    "reversal_5d": 0.1,
    "volume_shock": 3.0,
    "momentum_12m_1m": 0.5,
}

IN_SAMPLE_END = "2020-12-31"
OUT_OF_SAMPLE_START = "2021-01-01"
