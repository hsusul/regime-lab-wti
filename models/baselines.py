"""Lightweight baseline models for return-regime comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1e-12


@dataclass
class GaussianParams:
    """Simple univariate Gaussian parameters."""

    mu: float
    sigma: float


def _as_1d_float(observations: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(observations, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("observations must be a 1D array")
    if arr.size == 0:
        raise ValueError("observations must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observations must be finite")
    return arr


def _gaussian_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma_clipped = max(float(sigma), 1e-6)
    var = sigma_clipped * sigma_clipped
    return -0.5 * (np.log(2.0 * np.pi * var) + np.square((x - float(mu)) / sigma_clipped))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    vmax = np.max(values, axis=axis, keepdims=True)
    stable = values - vmax
    summed = np.sum(np.exp(stable), axis=axis, keepdims=True)
    out = vmax + np.log(np.maximum(summed, _EPS))
    return np.squeeze(out, axis=axis)


def single_regime_gaussian_fit(observations: np.ndarray | list[float]) -> dict[str, Any]:
    """Fit a single Gaussian baseline from observations."""
    obs = _as_1d_float(observations)
    sigma = float(np.std(obs, ddof=1)) if obs.size > 1 else 1e-6
    sigma = max(sigma, 1e-6)
    return {
        "mu": float(np.mean(obs)),
        "sigma": sigma,
        "n_obs": int(obs.size),
    }


def score_ll(
    observations: np.ndarray | list[float],
    *,
    mu: float,
    sigma: float,
) -> dict[str, float | int]:
    """Score univariate Gaussian log-likelihood metrics."""
    obs = _as_1d_float(observations)
    ll_vec = _gaussian_logpdf(obs, mu=float(mu), sigma=float(sigma))
    total = float(np.sum(ll_vec))
    return {
        "log_likelihood": total,
        "avg_log_likelihood": float(total / float(obs.size)),
        "n_obs": int(obs.size),
    }


def gmm_baseline_fit(
    observations: np.ndarray | list[float],
    n_components: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a lightweight 1D diagonal GMM via EM."""
    obs = _as_1d_float(observations)
    n = int(obs.size)
    k = int(max(1, n_components))

    if n < k:
        # Stable fallback when data is too short for requested components.
        k = n

    rng = np.random.default_rng(seed)
    quantile_grid = np.linspace(0.05, 0.95, num=k)
    means = np.quantile(obs, quantile_grid).astype(np.float64)
    if np.allclose(means, means[0]):
        means = means + rng.normal(loc=0.0, scale=1e-4, size=k)

    global_var = float(np.var(obs, ddof=1)) if n > 1 else 1e-4
    global_var = max(global_var, 1e-6)
    variances = np.full(k, global_var, dtype=np.float64)
    weights = np.full(k, 1.0 / float(k), dtype=np.float64)

    prev_ll: float | None = None
    converged = False
    n_iter = 0

    x = obs[:, None]
    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        log_prob = np.zeros((n, k), dtype=np.float64)
        for comp in range(k):
            sigma = float(np.sqrt(max(variances[comp], 1e-8)))
            log_prob[:, comp] = np.log(max(weights[comp], _EPS)) + _gaussian_logpdf(
                obs, mu=float(means[comp]), sigma=sigma
            )

        log_norm = _logsumexp(log_prob, axis=1)
        ll_total = float(np.sum(log_norm))

        responsibilities = np.exp(log_prob - log_norm[:, None])
        nk = np.sum(responsibilities, axis=0)
        nk = np.maximum(nk, _EPS)

        weights = nk / float(n)
        means = np.sum(responsibilities * x, axis=0) / nk
        centered_sq = np.square(x - means[None, :])
        variances = np.sum(responsibilities * centered_sq, axis=0) / nk
        variances = np.maximum(variances, 1e-8)

        if prev_ll is not None and abs(ll_total - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll_total

    sigma = np.sqrt(variances)
    return {
        "n_components": int(k),
        "weights": [float(v) for v in weights],
        "mu": [float(v) for v in means],
        "sigma": [float(v) for v in sigma],
        "n_iter": int(n_iter),
        "converged": bool(converged),
    }


def gmm_score_ll(observations: np.ndarray | list[float], gmm_params: dict[str, Any]) -> dict[str, float | int]:
    """Score log-likelihood for a fitted 1D GMM baseline."""
    obs = _as_1d_float(observations)
    mu = np.asarray(gmm_params.get("mu", []), dtype=np.float64)
    sigma = np.asarray(gmm_params.get("sigma", []), dtype=np.float64)
    weights = np.asarray(gmm_params.get("weights", []), dtype=np.float64)

    if mu.ndim != 1 or sigma.ndim != 1 or weights.ndim != 1:
        raise ValueError("gmm params must be 1D arrays")
    if not (mu.shape == sigma.shape == weights.shape):
        raise ValueError("gmm params shape mismatch")

    k = int(mu.shape[0])
    if k == 0:
        raise ValueError("gmm params must contain at least one component")

    log_prob = np.zeros((obs.size, k), dtype=np.float64)
    for comp in range(k):
        log_prob[:, comp] = np.log(max(float(weights[comp]), _EPS)) + _gaussian_logpdf(
            obs,
            mu=float(mu[comp]),
            sigma=float(sigma[comp]),
        )
    log_norm = _logsumexp(log_prob, axis=1)
    total = float(np.sum(log_norm))
    return {
        "log_likelihood": total,
        "avg_log_likelihood": float(total / float(obs.size)),
        "n_obs": int(obs.size),
    }


def _rolling_realized_vol(
    observations: np.ndarray,
    window: int,
    history: np.ndarray | None = None,
) -> np.ndarray:
    hist = np.asarray(history, dtype=np.float64) if history is not None else np.asarray([], dtype=np.float64)
    if hist.ndim != 1:
        raise ValueError("history must be 1D")

    combined = np.concatenate([hist, observations])
    vol = np.full(combined.shape[0], np.nan, dtype=np.float64)
    if combined.size == 0:
        return np.asarray([], dtype=np.float64)

    w = int(max(2, window))
    for idx in range(w - 1, combined.shape[0]):
        segment = combined[idx - w + 1 : idx + 1]
        vol[idx] = float(np.std(segment, ddof=1)) if segment.size > 1 else 0.0

    obs_slice = vol[hist.shape[0] :]
    if np.all(np.isnan(obs_slice)):
        return np.zeros_like(observations, dtype=np.float64)

    fill_value = float(np.nanmedian(obs_slice))
    obs_slice = np.where(np.isnan(obs_slice), fill_value, obs_slice)
    return obs_slice


def _assign_vol_regimes(vol: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    states = np.zeros(vol.shape[0], dtype=np.int64)
    states[vol > q_low] = 1
    states[vol > q_high] = 2
    return states


def rule_based_vol_regime_fit(
    observations: np.ndarray | list[float],
    window: int = 20,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> dict[str, Any]:
    """Fit a simple volatility-threshold regime baseline."""
    obs = _as_1d_float(observations)
    realized_vol = _rolling_realized_vol(obs, window=int(window), history=None)

    q_low = float(np.quantile(realized_vol, float(low_quantile)))
    q_high = float(np.quantile(realized_vol, float(high_quantile)))
    q_high = max(q_high, q_low + 1e-8)

    states = _assign_vol_regimes(realized_vol, q_low=q_low, q_high=q_high)
    global_mu = float(np.mean(obs))
    global_sigma = float(np.std(obs, ddof=1)) if obs.size > 1 else 1e-6
    global_sigma = max(global_sigma, 1e-6)

    state_mu: list[float] = []
    state_sigma: list[float] = []
    state_counts: list[int] = []
    for state in range(3):
        mask = states == state
        state_counts.append(int(np.sum(mask)))
        if int(np.sum(mask)) == 0:
            state_mu.append(global_mu)
            state_sigma.append(global_sigma)
            continue
        state_obs = obs[mask]
        state_mu.append(float(np.mean(state_obs)))
        sigma = float(np.std(state_obs, ddof=1)) if state_obs.size > 1 else global_sigma
        state_sigma.append(max(sigma, 1e-6))

    return {
        "window": int(window),
        "thresholds": {"q_low": q_low, "q_high": q_high},
        "state_mu": state_mu,
        "state_sigma": state_sigma,
        "state_counts": state_counts,
    }


def rule_based_vol_regime_score(
    observations: np.ndarray | list[float],
    model_params: dict[str, Any],
    history: np.ndarray | list[float] | None = None,
) -> dict[str, float | int]:
    """Score rule-based vol regime baseline with Gaussian state emissions."""
    obs = _as_1d_float(observations)
    hist_arr = None if history is None else _as_1d_float(history)

    thresholds = model_params.get("thresholds", {})
    q_low = float(thresholds.get("q_low", 0.0))
    q_high = float(thresholds.get("q_high", 1.0))
    window = int(model_params.get("window", 20))

    state_mu = np.asarray(model_params.get("state_mu", []), dtype=np.float64)
    state_sigma = np.asarray(model_params.get("state_sigma", []), dtype=np.float64)
    if state_mu.shape != (3,) or state_sigma.shape != (3,):
        raise ValueError("rule-based parameters must contain 3 state_mu/state_sigma values")

    realized_vol = _rolling_realized_vol(obs, window=window, history=hist_arr)
    states = _assign_vol_regimes(realized_vol, q_low=q_low, q_high=q_high)
    ll = 0.0
    for idx in range(obs.size):
        s = int(states[idx])
        ll += float(_gaussian_logpdf(np.asarray([obs[idx]]), state_mu[s], state_sigma[s])[0])

    return {
        "log_likelihood": float(ll),
        "avg_log_likelihood": float(ll / float(obs.size)),
        "n_obs": int(obs.size),
    }


def evaluate_baselines(
    train_obs: np.ndarray,
    val_obs: np.ndarray,
    test_obs: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit and score all baseline models on train/val/test splits."""
    train_arr = _as_1d_float(train_obs)
    val_arr = _as_1d_float(val_obs)
    test_arr = _as_1d_float(test_obs)

    single = single_regime_gaussian_fit(train_arr)
    gmm = gmm_baseline_fit(train_arr, n_components=3, seed=seed)
    rule = rule_based_vol_regime_fit(train_arr)

    val_hist = train_arr
    test_hist = np.concatenate([train_arr, val_arr])

    out: dict[str, Any] = {
        "single_regime_gaussian": {
            "fit": single,
            "val": score_ll(val_arr, mu=float(single["mu"]), sigma=float(single["sigma"])),
            "test": score_ll(test_arr, mu=float(single["mu"]), sigma=float(single["sigma"])),
        },
        "gmm_diag_k3": {
            "fit": gmm,
            "val": gmm_score_ll(val_arr, gmm),
            "test": gmm_score_ll(test_arr, gmm),
        },
        "rule_based_vol_regime": {
            "fit": rule,
            "val": rule_based_vol_regime_score(val_arr, rule, history=val_hist),
            "test": rule_based_vol_regime_score(test_arr, rule, history=test_hist),
        },
    }

    best_name = None
    best_test_avg = None
    for name, payload in out.items():
        test_avg = float(payload.get("test", {}).get("avg_log_likelihood", float("-inf")))
        if best_test_avg is None or test_avg > best_test_avg:
            best_test_avg = test_avg
            best_name = name

    return {
        "baselines": out,
        "summary": {
            "best_baseline_name": best_name,
            "best_baseline_test_avg_log_likelihood": best_test_avg,
        },
    }
