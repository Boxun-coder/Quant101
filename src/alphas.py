from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src import config


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def short_term_reversal(ret: pd.DataFrame) -> pd.DataFrame:
    signal = -ret.rolling(config.REVERSAL_WINDOW_DAYS, min_periods=config.REVERSAL_WINDOW_DAYS).sum()
    return cross_sectional_zscore(signal)


def volume_shock(vol: pd.DataFrame) -> pd.DataFrame:
    sma = vol.rolling(config.VOLUME_SMA_DAYS, min_periods=config.VOLUME_SMA_DAYS).mean()
    signal = vol.div(sma)
    return cross_sectional_zscore(signal)


def momentum_12m_1m(ret: pd.DataFrame) -> pd.DataFrame:
    bounded = ret.clip(lower=-0.999999)
    log_ret = np.log1p(bounded)
    signal = np.expm1(
        log_ret.shift(config.MOMENTUM_SKIP_DAYS).rolling(
            config.MOMENTUM_LOOKBACK_DAYS,
            min_periods=config.MOMENTUM_LOOKBACK_DAYS,
        ).sum()
    )
    return cross_sectional_zscore(signal)


def tail_concentrate(frame: pd.DataFrame, tail_fraction: float) -> pd.DataFrame:
    ranks = frame.rank(axis=1, pct=True, method="average")
    tail_mask = (ranks <= tail_fraction) | (ranks >= 1.0 - tail_fraction)
    return frame.where(tail_mask, 0.0)


def smooth_signal(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=1).mean()


def preprocess_factor(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    focused = tail_concentrate(frame, config.FACTOR_TAIL_FRACTIONS[name])
    return smooth_signal(focused, config.FACTOR_SMOOTHING_WINDOWS[name])


def aggregate_factors(
    factors: Dict[str, pd.DataFrame],
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or {name: 1.0 for name in factors}
    combined = None
    total_weight = 0.0

    for name, frame in factors.items():
        weight = weights[name]
        combined = frame.mul(weight) if combined is None else combined.add(frame.mul(weight), fill_value=0.0)
        total_weight += weight

    if combined is None or total_weight == 0.0:
        raise ValueError("At least one factor with a non-zero weight is required.")

    return combined.div(total_weight)


def build_raw_alpha(ret: pd.DataFrame, vol: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base_factors = {
        "reversal_5d": short_term_reversal(ret),
        "volume_shock": volume_shock(vol),
        "momentum_12m_1m": momentum_12m_1m(ret),
    }
    processed_factors = {
        name: preprocess_factor(name, factor)
        for name, factor in base_factors.items()
    }
    raw_alpha = aggregate_factors(processed_factors, config.FACTOR_COMBINATION_WEIGHTS)

    return {
        **base_factors,
        "raw_alpha": raw_alpha,
    }
