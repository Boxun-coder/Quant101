from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src import config


REQUIRED_COLUMNS = ["PERMNO", "date", "PRC", "VOL", "RET", "RETX", "SHROUT", "sprtrn"]


def load_raw_crsp(path: Path | str = config.RAW_DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    else:
        frame = pd.read_csv(
            path,
            usecols=REQUIRED_COLUMNS,
            parse_dates=["date"],
            engine="python",
        )

    for column in ["PRC", "VOL", "RET", "RETX", "SHROUT", "sprtrn"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="raise").astype(int)
    frame = frame.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    return frame


def _align_panel(frame: pd.DataFrame, all_dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    price = frame.pivot_table(index="date", columns="PERMNO", values="PRC", aggfunc="last")
    ret = frame.pivot_table(index="date", columns="PERMNO", values="RET", aggfunc="last")
    vol = frame.pivot_table(index="date", columns="PERMNO", values="VOL", aggfunc="last")
    shrout = frame.pivot_table(index="date", columns="PERMNO", values="SHROUT", aggfunc="last")

    price = price.reindex(all_dates).sort_index()
    ret = ret.reindex(all_dates).sort_index()
    vol = vol.reindex(all_dates).sort_index()
    shrout = shrout.reindex(all_dates).sort_index()

    price_missing = price.isna()
    price = price.ffill(limit=config.FORWARD_FILL_LIMIT)
    shrout = shrout.ffill(limit=config.FORWARD_FILL_LIMIT)

    filled_price_mask = price_missing & price.notna()
    vol = vol.mask(filled_price_mask, 0.0)
    ret = ret.mask(filled_price_mask, 0.0)

    market_cap = price * shrout

    return {
        "prc": price,
        "ret": ret,
        "vol": vol,
        "mktcap": market_cap,
    }


def _write_parquet(frame: pd.DataFrame | pd.Series, path: Path) -> None:
    try:
        frame.to_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install `pyarrow` to satisfy the project requirements."
        ) from exc


def build_clean_matrices(
    raw_path: Path | str = config.RAW_DATA_FILE,
    output_dir: Path | str = config.CLEAN_DIR,
) -> Dict[str, pd.DataFrame | pd.Series]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_raw_crsp(raw_path).copy()
    frame["PRC"] = frame["PRC"].abs()
    frame["MktCap"] = frame["PRC"] * frame["SHROUT"]

    all_dates = pd.DatetimeIndex(sorted(frame["date"].dropna().unique()))
    benchmark = frame.groupby("date", sort=True)["sprtrn"].first().reindex(all_dates)
    panels = _align_panel(frame, all_dates)

    _write_parquet(panels["prc"], config.PRICE_FILE)
    _write_parquet(panels["ret"], config.RETURN_FILE)
    _write_parquet(panels["vol"], config.VOLUME_FILE)
    _write_parquet(panels["mktcap"], config.MARKET_CAP_FILE)
    _write_parquet(benchmark.to_frame(name="sprtrn"), config.BENCHMARK_FILE)

    return {
        **panels,
        "sprtrn": benchmark.rename("sprtrn"),
    }
