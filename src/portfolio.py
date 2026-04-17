from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.risk_model import beta_neutralize, rolling_beta


def demean_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


def scale_to_gross(frame: pd.DataFrame, gross_target: float = config.GROSS_LEVERAGE_TARGET) -> pd.DataFrame:
    gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
    scaled = frame.div(gross, axis=0).mul(gross_target)
    return scaled.replace([np.inf, -np.inf], np.nan)


def build_target_weights(raw_alpha: pd.DataFrame, ret: pd.DataFrame, benchmark: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta = rolling_beta(ret, benchmark)
    neutral_alpha = beta_neutralize(raw_alpha, beta)
    demeaned = demean_cross_section(neutral_alpha)
    weights = scale_to_gross(demeaned)
    return weights, beta


def check_dollar_neutrality(weights: pd.DataFrame, tolerance: float = 1e-10) -> bool:
    net = weights.sum(axis=1).fillna(0.0)
    return bool((net.abs() <= tolerance).all())
