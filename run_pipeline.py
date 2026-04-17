from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))

from src import config
from src.alphas import build_raw_alpha
from src.backtester import build_results_matrix, format_tear_sheet, run_backtest
from src.data_pipeline import build_clean_matrices
from src.portfolio import build_target_weights, check_dollar_neutrality
from src.report_generator import generate_report


def main() -> None:
    clean = build_clean_matrices()
    factors = build_raw_alpha(clean["ret"], clean["vol"])
    raw_alpha = factors["raw_alpha"]
    raw_alpha.to_parquet(config.RAW_ALPHA_FILE)

    target_weights, _ = build_target_weights(raw_alpha, clean["ret"], clean["sprtrn"])
    target_weights.to_parquet(config.TARGET_WEIGHTS_FILE)

    daily_pnl, turnover, metrics = run_backtest(target_weights, clean["ret"])
    results_matrix = build_results_matrix(target_weights, clean["ret"], daily_pnl, turnover)
    results_matrix.to_parquet(config.RESULTS_MATRIX_FILE)

    if not check_dollar_neutrality(target_weights.fillna(0.0)):
        raise ValueError("Target weights are not dollar-neutral on every day.")

    generate_report()

    print(format_tear_sheet(metrics))
    print(f"Daily observations     : {len(daily_pnl)}")
    print(f"Active trading days    : {(target_weights.abs().sum(axis=1) > 0).sum()}")
    print(f"Parquet output folder  : {config.CLEAN_DIR}")
    print(f"Final report           : {config.FINAL_REPORT_FILE}")


if __name__ == "__main__":
    main()
