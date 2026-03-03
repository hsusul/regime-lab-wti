"""TensorFlow Probability Gaussian HMM implementation for 3-regime modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

tf = None
tfp = None
tfd = None


def _require_tensorflow() -> None:
    """Import TensorFlow/TFP lazily so artifact-first paths do not require them."""
    global tf, tfp, tfd
    if tf is not None and tfp is not None and tfd is not None:
        return
    try:
        import tensorflow as _tf
        import tensorflow_probability as _tfp
    except Exception as exc:
        raise ImportError(
            "TensorFlow + TensorFlow Probability are required for HMM training/inference."
        ) from exc

    tf = _tf
    tfp = _tfp
    tfd = _tfp.distributions


@dataclass
class HMMTrainingConfig:
    """Training hyperparameters for GaussianHMMTFP."""

    learning_rate: float = 0.05
    max_epochs: int = 300
    patience: int = 10
    min_delta: float = 1e-3
    occupancy_entropy_weight: float = 1.0


class GaussianHMMTFP:
    """Trainable Gaussian Hidden Markov Model using TensorFlow Probability."""

    def __init__(self, n_states: int = 3, seed: int = 42) -> None:
        _require_tensorflow()
        if n_states <= 0:
            raise ValueError("n_states must be > 0")

        self.n_states = n_states
        self.seed = seed
        self.dtype = tf.float64

        tf.keras.utils.set_random_seed(seed)

        # Bias logits toward persistence via identity initialization.
        self.initial_logits = tf.Variable(
            tf.zeros([self.n_states], dtype=self.dtype), name="initial_logits"
        )
        self.transition_logits = tf.Variable(
            tf.eye(self.n_states, dtype=self.dtype), name="transition_logits"
        )
        self.loc = tf.Variable(
            tf.linspace(
                tf.constant(-0.01, dtype=self.dtype),
                tf.constant(0.01, dtype=self.dtype),
                self.n_states,
            ),
            dtype=self.dtype,
            name="loc",
        )
        self.unconstrained_scale = tf.Variable(
            tf.zeros([self.n_states], dtype=self.dtype), name="unconstrained_scale"
        )

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [
            self.initial_logits,
            self.transition_logits,
            self.loc,
            self.unconstrained_scale,
        ]

    def _scale(self) -> tf.Tensor:
        # Keep strictly positive scale; avoid a hard ceiling that encourages collapse.
        return tf.nn.softplus(self.unconstrained_scale) + tf.constant(1e-4, self.dtype)

    def distribution(self, num_steps: int) -> tfd.HiddenMarkovModel:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        return tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(logits=self.initial_logits),
            transition_distribution=tfd.Categorical(logits=self.transition_logits),
            observation_distribution=tfd.Normal(loc=self.loc, scale=self._scale()),
            num_steps=num_steps,
        )

    def log_prob(self, observations: np.ndarray | tf.Tensor) -> tf.Tensor:
        obs = tf.convert_to_tensor(observations, dtype=self.dtype)
        if obs.shape.rank != 1:
            obs = tf.reshape(obs, [-1])
        steps = int(obs.shape[0])
        hmm = self.distribution(steps)
        return hmm.log_prob(obs)

    def fit(
        self,
        train_observations: np.ndarray,
        val_observations: Optional[np.ndarray] = None,
        config: Optional[HMMTrainingConfig] = None,
    ) -> Dict[str, Any]:
        cfg = config or HMMTrainingConfig()
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

        train_obs = np.asarray(train_observations, dtype=np.float64)
        if train_obs.ndim != 1 or train_obs.size < 2:
            raise ValueError("train_observations must be a 1D array with at least 2 points")

        val_obs: Optional[np.ndarray]
        if val_observations is not None:
            val_obs = np.asarray(val_observations, dtype=np.float64)
            if val_obs.ndim != 1 or val_obs.size < 1:
                raise ValueError("val_observations must be a non-empty 1D array")
        else:
            val_obs = None

        history: Dict[str, Any] = {
            "train_log_likelihood": [],
            "val_log_likelihood": [],
        }

        best_metric = -np.inf
        best_weights = self._snapshot()
        stale_epochs = 0
        stopped_early = False

        train_tensor = tf.convert_to_tensor(train_obs, dtype=self.dtype)
        val_tensor = (
            tf.convert_to_tensor(val_obs, dtype=self.dtype) if val_obs is not None else None
        )

        for epoch in range(cfg.max_epochs):
            with tf.GradientTape() as tape:
                neg_log_likelihood = -self.log_prob(train_tensor)
                sigma = self._scale()
                # Softly discourage extremely large sigmas (prevents an "outlier sink" regime).
                # 0.05 is already very high for daily returns; penalize above it.
                sigma_penalty = tf.reduce_sum(
                    tf.square(tf.nn.relu(sigma - tf.constant(0.05, dtype=self.dtype)))
                )
                neg_log_likelihood = neg_log_likelihood + tf.constant(
                    50.0, dtype=self.dtype
                ) * sigma_penalty
                if cfg.occupancy_entropy_weight > 0.0:
                    hmm = self.distribution(num_steps=int(train_tensor.shape[0]))
                    marg = hmm.posterior_marginals(train_tensor).probs_parameter()
                    avg_occ = tf.reduce_mean(marg, axis=0)
                    entropy = -tf.reduce_sum(avg_occ * tf.math.log(avg_occ + 1e-8))
                    neg_log_likelihood = (
                        neg_log_likelihood - cfg.occupancy_entropy_weight * entropy
                    )

            grads = tape.gradient(neg_log_likelihood, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            train_ll = float(self.log_prob(train_tensor).numpy())
            val_ll = (
                float(self.log_prob(val_tensor).numpy()) if val_tensor is not None else train_ll
            )

            history["train_log_likelihood"].append(train_ll)
            history["val_log_likelihood"].append(val_ll)

            metric = val_ll
            if metric > best_metric + cfg.min_delta:
                best_metric = metric
                best_weights = self._snapshot()
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= cfg.patience:
                stopped_early = True
                break

        self._restore(best_weights)

        history["best_val_log_likelihood"] = best_metric
        history["epochs_ran"] = len(history["train_log_likelihood"])
        history["stopped_early"] = stopped_early
        return history

    def get_params(self) -> Dict[str, np.ndarray]:
        transition_matrix = tf.nn.softmax(self.transition_logits, axis=-1).numpy()
        initial_probs = tf.nn.softmax(self.initial_logits, axis=-1).numpy()
        mu = self.loc.numpy()
        sigma = self._scale().numpy()
        return {
            "n_states": np.array([self.n_states], dtype=np.int64),
            "initial_probs": initial_probs,
            "transition_matrix": transition_matrix,
            "mu": mu,
            "sigma": sigma,
            "initial_logits": self.initial_logits.numpy(),
            "transition_logits": self.transition_logits.numpy(),
            "unconstrained_scale": self.unconstrained_scale.numpy(),
        }

    def save(self, output_path: Path) -> None:
        params = self.get_params()
        np.savez(output_path, **params)

    @classmethod
    def load(cls, output_path: Path) -> "GaussianHMMTFP":
        params = np.load(output_path)
        n_states = int(params["n_states"][0])
        model = cls(n_states=n_states)
        model.initial_logits.assign(params["initial_logits"])
        model.transition_logits.assign(params["transition_logits"])
        model.loc.assign(params["mu"])
        model.unconstrained_scale.assign(params["unconstrained_scale"])
        return model

    def posterior_marginals(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        hmm = self.distribution(obs.shape[0])
        posterior = hmm.posterior_marginals(obs)
        return posterior.probs_parameter().numpy()

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Compute most likely hidden state sequence for observations."""
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("observations must be a non-empty 1D array")

        try:
            hmm = self.distribution(obs.shape[0])
            posterior_mode = hmm.posterior_mode(obs)
            mode_arr = np.asarray(
                posterior_mode.numpy() if hasattr(posterior_mode, "numpy") else posterior_mode,
                dtype=np.int64,
            )
            return mode_arr.reshape(-1)
        except Exception:
            return self._viterbi_numpy(obs)

    def _viterbi_numpy(self, observations: np.ndarray) -> np.ndarray:
        """Numpy fallback Viterbi decoder from current model parameters."""
        params = self.get_params()

        init = np.asarray(params["initial_probs"], dtype=np.float64)
        trans = np.asarray(params["transition_matrix"], dtype=np.float64)
        mu = np.asarray(params["mu"], dtype=np.float64)
        sigma = np.maximum(np.asarray(params["sigma"], dtype=np.float64), 1e-8)

        n_steps = observations.shape[0]
        n_states = init.shape[0]
        if trans.shape != (n_states, n_states):
            raise ValueError("transition matrix shape mismatch")

        eps = 1e-12
        log_init = np.log(np.clip(init, eps, 1.0))
        log_trans = np.log(np.clip(trans, eps, 1.0))

        log_norm = -0.5 * np.log(2.0 * np.pi)
        delta = np.full((n_steps, n_states), -np.inf, dtype=np.float64)
        psi = np.zeros((n_steps, n_states), dtype=np.int64)

        log_emit0 = log_norm - np.log(sigma) - 0.5 * np.square((observations[0] - mu) / sigma)
        delta[0] = log_init + log_emit0

        for t in range(1, n_steps):
            log_emit_t = (
                log_norm - np.log(sigma) - 0.5 * np.square((observations[t] - mu) / sigma)
            )
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(n_states)] + log_emit_t

        states = np.zeros(n_steps, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_steps - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def _snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "initial_logits": self.initial_logits.numpy().copy(),
            "transition_logits": self.transition_logits.numpy().copy(),
            "loc": self.loc.numpy().copy(),
            "unconstrained_scale": self.unconstrained_scale.numpy().copy(),
        }

    def _restore(self, weights: Dict[str, np.ndarray]) -> None:
        self.initial_logits.assign(weights["initial_logits"])
        self.transition_logits.assign(weights["transition_logits"])
        self.loc.assign(weights["loc"])
        self.unconstrained_scale.assign(weights["unconstrained_scale"])
