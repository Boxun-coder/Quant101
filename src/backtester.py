from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def max_drawdown(returns: pd.Series) -> float:
    equity_curve = (1.0 + returns.fillna(0.0)).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve.div(running_peak).sub(1.0)
    return float(drawdown.min())


def compute_metrics(daily_returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    daily_returns = daily_returns.fillna(0.0)
    turnover = turnover.fillna(0.0)

    annualized_return = float((1.0 + daily_returns).prod() ** (config.TRADING_DAYS_PER_YEAR / len(daily_returns)) - 1.0)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(config.TRADING_DAYS_PER_YEAR))
    sharpe = float(np.nan if annualized_volatility == 0 else daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(config.TRADING_DAYS_PER_YEAR))

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(daily_returns),
        "average_daily_turnover": float(turnover.mean()),
    }


def run_backtest(target_weights: pd.DataFrame, ret: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    target_weights = target_weights.reindex(ret.index)
    executed_weights = target_weights.shift(1)

    gross_pnl = executed_weights.mul(ret).sum(axis=1, min_count=1)
    turnover = target_weights.diff().abs().sum(axis=1).fillna(0.0)
    realized_turnover = turnover.shift(1).fillna(0.0)
    net_pnl = gross_pnl.fillna(0.0) - (config.TRANSACTION_COST_BPS / 10000.0) * realized_turnover

    metrics = compute_metrics(net_pnl, realized_turnover)
    return net_pnl, realized_turnover, metrics


def build_results_matrix(
    target_weights: pd.DataFrame,
    ret: pd.DataFrame,
    daily_pnl: pd.Series,
    turnover: pd.Series,
) -> pd.DataFrame:
    target_weights = target_weights.reindex(ret.index)
    gross_exposure = target_weights.abs().sum(axis=1)
    net_exposure = target_weights.sum(axis=1)
    cumulative_return = (1.0 + daily_pnl.fillna(0.0)).cumprod() - 1.0
    running_peak = (1.0 + daily_pnl.fillna(0.0)).cumprod().cummax()
    drawdown = (1.0 + daily_pnl.fillna(0.0)).cumprod().div(running_peak).sub(1.0)

    results = pd.DataFrame(
        {
            "daily_return": daily_pnl.fillna(0.0),
            "turnover": turnover.fillna(0.0),
            "gross_exposure": gross_exposure.fillna(0.0),
            "net_exposure": net_exposure.fillna(0.0),
            "cumulative_return": cumulative_return.fillna(0.0),
            "drawdown": drawdown.fillna(0.0),
        },
        index=ret.index,
    )
    return results


def format_tear_sheet(metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            "Stat Arb Backtest Summary",
            "-------------------------",
            f"Annualized Return     : {metrics['annualized_return']:.2%}",
            f"Annualized Volatility : {metrics['annualized_volatility']:.2%}",
            f"Sharpe Ratio          : {metrics['sharpe_ratio']:.2f}",
            f"Maximum Drawdown      : {metrics['max_drawdown']:.2%}",
            f"Average Daily Turnover: {metrics['average_daily_turnover']:.4f}",
        ]
    )
