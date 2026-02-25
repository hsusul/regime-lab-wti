from __future__ import annotations

import numpy as np
import pandas as pd

from energy_data.features import compute_log_returns


def test_compute_log_returns_values() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "price": [100.0, 110.0, 121.0],
        }
    )

    out = compute_log_returns(df)
    expected = np.array([np.log(1.1), np.log(1.1)])

    assert out.shape[0] == 2
    assert np.allclose(out["log_return"].to_numpy(), expected)
