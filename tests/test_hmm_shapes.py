from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")

from models.hmm_tfp import GaussianHMMTFP, HMMTrainingConfig
from models.infer import forward_filter_probs


def test_hmm_shapes_and_probabilities() -> None:
    rng = np.random.default_rng(42)
    observations = rng.normal(loc=0.0, scale=0.01, size=120)

    model = GaussianHMMTFP(n_states=3, seed=42)
    history = model.fit(
        train_observations=observations[:80],
        val_observations=observations[80:100],
        config=HMMTrainingConfig(
            learning_rate=0.03,
            max_epochs=5,
            patience=2,
            occupancy_entropy_weight=0.1,
        ),
    )

    assert history["epochs_ran"] >= 1

    params = model.get_params()
    transition = params["transition_matrix"]

    assert transition.shape == (3, 3)
    assert np.allclose(transition.sum(axis=1), np.ones(3), atol=1e-5)

    probs = forward_filter_probs(
        observations=observations,
        initial_probs=params["initial_probs"],
        transition_matrix=transition,
        mu=params["mu"],
        sigma=params["sigma"],
    )

    assert probs.shape == (observations.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), np.ones(observations.shape[0]), atol=1e-6)
