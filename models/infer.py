"""Inference helpers for filtering probabilities and forecasting from Gaussian HMM parameters."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import pandas as pd


def assign_regime_labels(
    mu: np.ndarray, sigma: np.ndarray, transition_matrix: np.ndarray
) -> dict[str, str]:
    """Assign deterministic semantic labels to regimes.

    Labels are assigned by ascending volatility (sigma):
    smallest -> low_vol, middle -> mid_vol, largest -> shock.
    """
    _ = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    trans = np.asarray(transition_matrix, dtype=np.float64)

    if sigma_arr.ndim != 1:
        raise ValueError("sigma must be a 1D array")
    n_states = sigma_arr.shape[0]
    if trans.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape mismatch for labels")

    mu_arr = np.asarray(mu, dtype=np.float64)
    if mu_arr.shape != sigma_arr.shape:
        raise ValueError("mu/sigma shape mismatch for labels")

    # Primary: sigma ascending.
    # Tie-breaker: abs(mu) ascending so higher |mu| receives the "shock" rank for tied sigma.
    # Final tie-breaker: state index ascending for deterministic output.
    state_idx = np.arange(n_states, dtype=np.int64)
    sorted_idx = np.lexsort((state_idx, np.abs(mu_arr), sigma_arr))

    rank_names = ["low_vol", "mid_vol", "shock"]
    rank_to_name: list[str] = []
    for rank in range(n_states):
        if rank < len(rank_names):
            rank_to_name.append(rank_names[rank])
        else:
            rank_to_name.append(f"regime_rank_{rank}")

    mapping: dict[str, str] = {}
    for rank, state_idx in enumerate(sorted_idx):
        mapping[str(int(state_idx))] = rank_to_name[rank]
    return mapping


def stationary_distribution(
    transition_matrix: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000
) -> np.ndarray:
    """Compute stationary distribution via power iteration."""
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    if a_mat.ndim != 2 or a_mat.shape[0] != a_mat.shape[1]:
        raise ValueError("transition_matrix must be square")

    n_states = a_mat.shape[0]
    dist = np.full(n_states, 1.0 / n_states, dtype=np.float64)

    for _ in range(max_iter):
        nxt = dist @ a_mat
        if np.max(np.abs(nxt - dist)) < tol:
            dist = nxt
            break
        dist = nxt

    total = float(np.sum(dist))
    if total <= 0:
        return np.full(n_states, 1.0 / n_states, dtype=np.float64)
    return dist / total


def one_step_predictive_regime_distribution(
    current_probs: np.ndarray, transition_matrix: np.ndarray
) -> np.ndarray:
    """Propagate current state probabilities forward one step."""
    probs = np.asarray(current_probs, dtype=np.float64)
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("current_probs must be 1D")
    if a_mat.shape != (probs.shape[0], probs.shape[0]):
        raise ValueError("transition_matrix shape mismatch for one-step prediction")

    nxt = probs @ a_mat
    total = float(np.sum(nxt))
    if total <= 0:
        return np.full_like(probs, 1.0 / probs.shape[0], dtype=np.float64)
    return nxt / total


def probability_weighted_moments(
    state_probs: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> tuple[float, float]:
    """Compute mixture expected return and volatility from state probabilities."""
    probs = np.asarray(state_probs, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    if probs.ndim != 1:
        raise ValueError("state_probs must be 1D")
    if mu_arr.shape != probs.shape or sigma_arr.shape != probs.shape:
        raise ValueError("mu/sigma shape mismatch for weighted moments")

    mean = float(np.dot(probs, mu_arr))
    second_moment = float(np.dot(probs, np.square(sigma_arr) + np.square(mu_arr)))
    variance = max(second_moment - mean * mean, 1e-12)
    return mean, float(np.sqrt(variance))


def load_model_params_json(path: Path) -> dict[str, np.ndarray]:
    """Load compact model parameters needed for artifact-first inference."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required = {"transition_matrix", "mu", "sigma"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"model_params.json missing fields: {sorted(missing)}")

    transition_matrix = np.asarray(payload["transition_matrix"], dtype=np.float64)
    mu = np.asarray(payload["mu"], dtype=np.float64)
    sigma = np.asarray(payload["sigma"], dtype=np.float64)

    if mu.ndim != 1 or sigma.ndim != 1:
        raise ValueError("mu and sigma must be 1D arrays")
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have matching shapes")
    n_states = mu.shape[0]
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape mismatch with mu/sigma")

    initial_probs_raw = payload.get("initial_probs")
    initial_logits_raw = payload.get("initial_logits")
    if initial_probs_raw is not None:
        initial_probs = np.asarray(initial_probs_raw, dtype=np.float64)
        if initial_probs.shape != (n_states,):
            raise ValueError("initial_probs shape mismatch")
        total = float(np.sum(initial_probs))
        if total <= 0:
            raise ValueError("initial_probs must sum to a positive value")
        initial_probs = initial_probs / total
    elif initial_logits_raw is not None:
        logits = np.asarray(initial_logits_raw, dtype=np.float64)
        if logits.shape != (n_states,):
            raise ValueError("initial_logits shape mismatch")
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        initial_probs = exp_logits / np.sum(exp_logits)
    else:
        initial_probs = stationary_distribution(transition_matrix)

    return {
        "n_states": np.asarray([n_states], dtype=np.int64),
        "transition_matrix": transition_matrix,
        "mu": mu,
        "sigma": sigma,
        "initial_probs": initial_probs,
    }


def _normal_pdf(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-8)
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * np.square((x - mu) / sigma)
    return coeff * np.exp(exponent)


def forward_filter_probs(
    observations: np.ndarray,
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute filtering probabilities p(z_t | r_1:t) via forward recursion."""
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 1:
        raise ValueError("observations must be a 1D array")

    init = np.asarray(initial_probs, dtype=np.float64)
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    n_states = init.shape[0]
    if a_mat.shape != (n_states, n_states):
        raise ValueError("transition_matrix shape mismatch")
    if mu.shape[0] != n_states or sigma.shape[0] != n_states:
        raise ValueError("mu/sigma shape mismatch")

    probs = np.zeros((obs.shape[0], n_states), dtype=np.float64)

    emission = _normal_pdf(float(obs[0]), mu, sigma)
    alpha = init * emission
    total = np.sum(alpha)
    alpha = np.full_like(alpha, 1.0 / n_states) if total <= 0 else alpha / total
    probs[0] = alpha

    for t in range(1, obs.shape[0]):
        prior = probs[t - 1] @ a_mat
        emission = _normal_pdf(float(obs[t]), mu, sigma)
        alpha = prior * emission
        total = np.sum(alpha)
        probs[t] = np.full(n_states, 1.0 / n_states) if total <= 0 else alpha / total

    return probs


def transition_counts_from_states(states: np.ndarray, n_states: int) -> np.ndarray:
    """Transition counts from a hard state sequence."""
    seq = np.asarray(states, dtype=np.int64)
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    if seq.size < 2:
        return counts

    for i in range(seq.size - 1):
        counts[seq[i], seq[i + 1]] += 1
    return counts


def implied_avg_duration_days(transition_matrix: np.ndarray) -> list[float | None]:
    """Implied expected duration 1 / (1 - p_kk) for each regime."""
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    durations: list[float | None] = []
    for k in range(a_mat.shape[0]):
        p_stay = float(a_mat[k, k])
        if p_stay >= 0.999999:
            durations.append(None)
        else:
            durations.append(float(1.0 / (1.0 - p_stay)))
    return durations


def build_regime_summary(
    dates: Sequence[str],
    returns: np.ndarray,
    filter_probs: np.ndarray,
    transition_matrix: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    regime_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build per-regime and global summary metrics."""
    returns = np.asarray(returns, dtype=np.float64)
    probs = np.asarray(filter_probs, dtype=np.float64)

    if probs.shape[0] != returns.shape[0]:
        raise ValueError("filter_probs must align with returns length")

    n_states = probs.shape[1]
    hard_states = np.argmax(probs, axis=1)
    durations = implied_avg_duration_days(transition_matrix)
    trans_counts = transition_counts_from_states(hard_states, n_states)

    regime_rows: list[dict[str, Any]] = []
    for k in range(n_states):
        idx = hard_states == k
        assigned = int(np.sum(idx))

        emp_mean = float(np.mean(returns[idx])) if assigned > 0 else None
        emp_vol = float(np.std(returns[idx], ddof=1)) if assigned > 1 else None

        regime_rows.append(
            {
                "regime": int(k),
                "label": (
                    regime_labels.get(str(k), f"regime_{k}")
                    if regime_labels is not None
                    else f"regime_{k}"
                ),
                "mu": float(mu[k]),
                "sigma": float(sigma[k]),
                "avg_posterior_probability": float(np.mean(probs[:, k])),
                "observations_assigned": assigned,
                "empirical_mean_return": emp_mean,
                "empirical_volatility": emp_vol,
                "implied_avg_duration_days": durations[k],
            }
        )

    summary: dict[str, Any] = {
        "start_date": str(dates[0]) if dates else None,
        "end_date": str(dates[-1]) if dates else None,
        "n_observations": int(returns.shape[0]),
        "state_occupancy": [float(x) for x in np.mean(probs, axis=0)],
        "transition_counts": trans_counts.tolist(),
        "regimes": regime_rows,
    }
    return summary


def predict_proba_payload(
    dates: Sequence[str], returns: np.ndarray, filter_probs: np.ndarray
) -> dict[str, Any]:
    """Build JSON-serializable payload for probability endpoint/artifact."""
    probs = np.asarray(filter_probs, dtype=np.float64)
    return {
        "dates": [str(d) for d in dates],
        "returns": [float(x) for x in returns],
        "regime_probabilities": [[float(p) for p in row] for row in probs],
    }


def forecast_predictive_distribution(
    last_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    interval: float = 0.95,
    last_date: str | None = None,
) -> dict[str, Any]:
    """Forecast return distribution by propagating posterior state probabilities."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not (0.0 < interval < 1.0):
        raise ValueError("interval must be in (0, 1)")

    probs = np.asarray(last_posterior, dtype=np.float64)
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    z = NormalDist().inv_cdf(0.5 + 0.5 * interval)

    out_rows: list[dict[str, Any]] = []
    current = probs.copy()

    for h in range(1, horizon + 1):
        current = current @ a_mat

        mean_h = float(np.dot(current, mu))
        second_moment = float(np.dot(current, np.square(sigma) + np.square(mu)))
        var_h = max(second_moment - mean_h * mean_h, 1e-12)
        std_h = float(np.sqrt(var_h))

        out_rows.append(
            {
                "horizon": h,
                "mean": mean_h,
                "std": std_h,
                "lower": mean_h - z * std_h,
                "upper": mean_h + z * std_h,
                "state_probabilities": [float(x) for x in current],
            }
        )

    forecast_dates: list[str] = []
    if last_date is not None:
        start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
        forecast_dates = [
            ts.strftime("%Y-%m-%d")
            for ts in pd.bdate_range(start=start, periods=horizon, freq="B")
        ]

    return {
        "interval": interval,
        "horizon": horizon,
        "forecast_dates": forecast_dates,
        "forecast": out_rows,
    }
