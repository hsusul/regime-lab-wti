from __future__ import annotations

import numpy as np
import pandas as pd

from energy_data.features import compute_log_returns


def test_compute_log_returns_values() -> None:
    prices = [100.0]
    for _ in range(10):
        prices.append(prices[-1] * 1.1)

    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=len(prices), freq="D"),
            "price": prices,
        }
    )

    out = compute_log_returns(df)
    expected = np.full(len(prices) - 1, np.log(1.1), dtype=np.float64)

    assert out.shape[0] == len(prices) - 1
    assert np.allclose(out["log_return"].to_numpy(), expected)
