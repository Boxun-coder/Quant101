from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def rolling_beta(ret: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    benchmark = benchmark.reindex(ret.index)

    ret_mean = ret.rolling(config.BETA_LOOKBACK_DAYS, min_periods=config.BETA_LOOKBACK_DAYS).mean()
    benchmark_mean = benchmark.rolling(
        config.BETA_LOOKBACK_DAYS,
        min_periods=config.BETA_LOOKBACK_DAYS,
    ).mean()

    cross_mean = ret.mul(benchmark, axis=0).rolling(
        config.BETA_LOOKBACK_DAYS,
        min_periods=config.BETA_LOOKBACK_DAYS,
    ).mean()
    benchmark_var = benchmark.pow(2).rolling(
        config.BETA_LOOKBACK_DAYS,
        min_periods=config.BETA_LOOKBACK_DAYS,
    ).mean() - benchmark_mean.pow(2)

    covariance = cross_mean.sub(ret_mean.mul(benchmark_mean, axis=0))
    beta = covariance.div(benchmark_var.replace(0.0, np.nan), axis=0)
    return beta


def beta_neutralize(raw_alpha: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    beta_centered = beta.sub(beta.mean(axis=1), axis=0)
    alpha_centered = raw_alpha.sub(raw_alpha.mean(axis=1), axis=0)

    numerator = alpha_centered.mul(beta_centered).sum(axis=1, min_count=1)
    denominator = beta_centered.pow(2).sum(axis=1, min_count=1).replace(0.0, np.nan)
    slope = numerator.div(denominator)

    neutral_alpha = alpha_centered.sub(beta_centered.mul(slope, axis=0))
    return neutral_alpha.replace([np.inf, -np.inf], np.nan)
