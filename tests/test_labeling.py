from __future__ import annotations

import numpy as np

from models.infer import assign_regime_labels


def test_assign_regime_labels_deterministic_by_sigma() -> None:
    mu = np.array([0.001, -0.002, 0.0005], dtype=np.float64)
    sigma = np.array([0.03, 0.01, 0.10], dtype=np.float64)
    transition = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.07, 0.03, 0.90],
        ],
        dtype=np.float64,
    )

    mapping_first = assign_regime_labels(mu=mu, sigma=sigma, transition_matrix=transition)
    mapping_second = assign_regime_labels(mu=mu, sigma=sigma, transition_matrix=transition)

    assert mapping_first == mapping_second
    assert mapping_first == {
        "1": "low_vol",
        "0": "mid_vol",
        "2": "shock",
    }
