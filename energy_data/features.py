"""Feature engineering functions for WTI spot price regime modeling."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_log_returns(
    price_df: pd.DataFrame,
    price_col: str = "price",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute daily log returns from a price series.

    Args:
        price_df: Input DataFrame with date and price columns.
        price_col: Name of the price column.
        date_col: Name of the date column.

    Returns:
        DataFrame with columns [date, price, log_return] and first row dropped.
    """
    missing = {date_col, price_col} - set(price_df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {sorted(missing)}")

    df = price_df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)

    # EIA occasionally includes missing/placeholder values; drop non-positive prices.
    df = df[df[price_col] > 0].copy()
    if df.shape[0] < 10:
        raise ValueError("Need at least 10 positive price observations to compute log returns.")

    df["log_return"] = np.log(df[price_col]).diff()
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    df = df.rename(columns={date_col: "date", price_col: "price"})
    return df


def time_based_split(
    values: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 1D array into contiguous train/validation/test partitions."""
    if values.ndim != 1:
        raise ValueError("time_based_split expects a 1D array.")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")

    n = values.shape[0]
    if n < 10:
        raise ValueError("Need at least 10 observations for train/val/test split.")

    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    val_end = min(val_end, n - 1)

    train = values[:train_end]
    val = values[train_end:val_end]
    test = values[val_end:]

    if val.size == 0 or test.size == 0:
        raise ValueError("Split produced empty validation/test partition.")

    return train, val, test
