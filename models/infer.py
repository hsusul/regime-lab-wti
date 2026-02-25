"""Inference helpers for filtering probabilities and forecasting from Gaussian HMM parameters."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import pandas as pd


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
