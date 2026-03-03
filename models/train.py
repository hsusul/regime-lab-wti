"""Training pipeline for WTI 3-regime Gaussian HMM using TensorFlow Probability."""

from __future__ import annotations

import json
import random
import subprocess
import hashlib
from statistics import NormalDist
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from energy_data.eia_client import EIAClientConfig, EIAWTIClient
from energy_data.features import compute_log_returns, time_based_split
from models.baselines import evaluate_baselines
from models.infer import (
    assign_regime_labels,
    build_regime_summary,
    forecast_predictive_distribution,
    forward_filter_probs,
    probability_weighted_moments,
    predict_proba_payload,
)
from models.provenance import write_manifest_with_provenance


@dataclass
class PipelineConfig:
    """Normalized pipeline settings loaded from YAML."""

    run_dir: str
    data: dict[str, Any]
    split: dict[str, Any]
    model: dict[str, Any]
    forecast: dict[str, Any]


DEFAULT_CONFIG: dict[str, Any] = {
    "run_dir": "runs",
    "data": {
        "raw_dir": "data/raw",
        "cache_filename": "wti_cushing_daily.csv",
        "start_date": None,
        "end_date": None,
    },
    "split": {
        "train_frac": 0.70,
        "val_frac": 0.15,
    },
    "model": {
        "n_states": 3,
        "learning_rate": 0.05,
        "max_epochs": 300,
        "patience": 10,
        "min_delta": 1e-3,
        "seed": 42,
    },
    "forecast": {
        "default_horizon": 10,
        "interval": 0.95,
        "include_quantiles": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULT_CONFIG, raw_cfg)
    return PipelineConfig(
        run_dir=str(merged["run_dir"]),
        data=dict(merged["data"]),
        split=dict(merged["split"]),
        model=dict(merged["model"]),
        forecast=dict(merged["forecast"]),
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        # Keep non-TF paths (tests, artifact-only workflows) importable.
        pass


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{uuid4().hex[:8]}"


def _safe_git_commit_hash(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        commit = completed.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


def _config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _log_return_short_series_message(price_df: pd.DataFrame) -> str:
    prices = pd.to_numeric(price_df.get("price"), errors="coerce")
    original_count = int(price_df.shape[0])
    filtered_count = int((prices > 0).sum())
    dropped_non_positive_or_nan = original_count - filtered_count
    min_price = float(prices.min()) if prices.notna().any() else None
    return (
        "Insufficient positive price history after filtering for log returns. "
        f"original_count={original_count}, "
        f"filtered_count={filtered_count}, "
        f"min_price={min_price}, "
        f"dropped_non_positive_or_nan={dropped_non_positive_or_nan}"
    )


def _transition_entropy(transition_matrix: np.ndarray) -> float:
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    clipped = np.clip(a_mat, 1e-12, 1.0)
    row_entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return float(np.mean(row_entropy))


def _viterbi_segment_stats(states: np.ndarray) -> dict[str, float | int | None]:
    seq = np.asarray(states, dtype=np.int64)
    if seq.ndim != 1 or seq.size == 0:
        return {"segment_count": 0, "mean_duration": None, "max_duration": None}

    durations: list[int] = []
    current = int(seq[0])
    count = 1
    for state in seq[1:]:
        state_i = int(state)
        if state_i == current:
            count += 1
        else:
            durations.append(count)
            current = state_i
            count = 1
    durations.append(count)

    return {
        "segment_count": int(len(durations)),
        "mean_duration": float(np.mean(durations)),
        "max_duration": int(np.max(durations)),
    }


def _build_data_meta_payload(price_df: pd.DataFrame, features_df: pd.DataFrame) -> dict[str, Any]:
    prices = price_df.copy()
    prices["date"] = pd.to_datetime(prices.get("date"), errors="coerce")
    prices["price"] = pd.to_numeric(prices.get("price"), errors="coerce")

    features = features_df.copy()
    features["date"] = pd.to_datetime(features.get("date"), errors="coerce")
    features["log_return"] = pd.to_numeric(features.get("log_return"), errors="coerce")

    canonical = features[["date", "log_return"]].dropna().copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values("date").reset_index(drop=True)
    dataset_bytes = canonical.to_csv(index=False, float_format="%.12g").encode("utf-8")

    return {
        "dataset_hash": _sha256_bytes(dataset_bytes),
        "start_date": (
            str(canonical["date"].iloc[0]) if not canonical.empty else None
        ),
        "end_date": (
            str(canonical["date"].iloc[-1]) if not canonical.empty else None
        ),
        "row_counts": {
            "raw_rows": int(prices.shape[0]),
            "feature_rows": int(features.shape[0]),
            "hash_rows": int(canonical.shape[0]),
        },
        "missingness": {
            "raw_missing_price_rows": int(prices["price"].isna().sum()),
            "raw_non_positive_price_rows": int((prices["price"] <= 0).fillna(False).sum()),
            "feature_missing_return_rows": int(features["log_return"].isna().sum()),
        },
    }


def _robust_sigma_by_regime(
    returns: np.ndarray,
    states: np.ndarray,
    fallback_sigma: np.ndarray,
) -> list[float]:
    obs = np.asarray(returns, dtype=np.float64)
    seq = np.asarray(states, dtype=np.int64)
    sigma = np.asarray(fallback_sigma, dtype=np.float64)
    if obs.shape[0] != seq.shape[0]:
        raise ValueError("returns and states must align for robust sigma computation")

    robust_sigma: list[float] = []
    for state in range(sigma.shape[0]):
        state_obs = obs[seq == state]
        if state_obs.size < 2:
            robust_sigma.append(float(max(sigma[state], 1e-6)))
            continue
        median = float(np.median(state_obs))
        mad = float(np.median(np.abs(state_obs - median)))
        robust = max(1.4826 * mad, 1e-6)
        if not np.isfinite(robust):
            robust = float(max(sigma[state], 1e-6))
        robust_sigma.append(float(robust))
    return robust_sigma


def _regime_stability_metrics(
    states: np.ndarray,
    labels_map: dict[str, str],
    *,
    window_days: int = 60,
) -> dict[str, Any]:
    seq = np.asarray(states, dtype=np.int64)
    if seq.ndim != 1 or seq.size == 0:
        return {
            "window_days": int(window_days),
            "label_flip_rate": None,
            "label_flip_rate_last_n_days": None,
            "avg_segment_length_by_label": {},
        }

    n_window = min(max(int(window_days), 2), int(seq.size))
    tail = seq[-n_window:]
    flips = int(np.sum(tail[1:] != tail[:-1])) if tail.size > 1 else 0
    flip_rate = float(flips / max(int(tail.size) - 1, 1))

    durations_by_label: dict[str, list[int]] = {}
    current = int(seq[0])
    count = 1
    for state in seq[1:]:
        state_i = int(state)
        if state_i == current:
            count += 1
            continue
        label = labels_map.get(str(current), f"regime_{current}")
        durations_by_label.setdefault(label, []).append(int(count))
        current = state_i
        count = 1
    label = labels_map.get(str(current), f"regime_{current}")
    durations_by_label.setdefault(label, []).append(int(count))

    avg_segment_length_by_label = {
        regime_label: float(np.mean(np.asarray(durations, dtype=np.float64)))
        for regime_label, durations in durations_by_label.items()
        if durations
    }

    return {
        "window_days": int(n_window),
        "label_flip_rate": flip_rate,
        "label_flip_rate_last_n_days": flip_rate,
        "avg_segment_length_by_label": avg_segment_length_by_label,
    }


def _infer_shock_state(regime_labels: dict[str, str], sigma: np.ndarray) -> int:
    for key, value in regime_labels.items():
        if str(value) == "shock":
            return int(key)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    return int(np.argmax(sigma_arr))


def _max_drawdown_proxy(log_returns: np.ndarray) -> float | None:
    obs = np.asarray(log_returns, dtype=np.float64)
    if obs.size == 0:
        return None
    cumulative = np.exp(np.cumsum(obs))
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(drawdown))


def _annualized_vol_proxy(log_returns: np.ndarray) -> float | None:
    obs = np.asarray(log_returns, dtype=np.float64)
    if obs.size < 2:
        return None
    return float(np.std(obs, ddof=1) * np.sqrt(252.0))


def _build_backtest_payload(
    run_id: str,
    created_at_utc: str,
    returns: np.ndarray,
    states: np.ndarray,
    labels_map: dict[str, str],
    sigma: np.ndarray,
) -> dict[str, Any]:
    obs = np.asarray(returns, dtype=np.float64)
    seq = np.asarray(states, dtype=np.int64)
    shock_state = _infer_shock_state(labels_map, sigma=sigma)
    active = (seq != shock_state).astype(np.float64)
    strategy_returns = obs * active

    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "strategy_name": "avoid_shock_regime",
        "shock_state": int(shock_state),
        "shock_label": labels_map.get(str(shock_state), f"regime_{shock_state}"),
        "n_days": int(obs.size),
        "days_avoided": int(np.sum(seq == shock_state)),
        "participation_rate": float(np.mean(active)),
        "annualized_vol_proxy": _annualized_vol_proxy(strategy_returns),
        "max_drawdown_proxy": _max_drawdown_proxy(strategy_returns),
        "buy_hold_annualized_vol_proxy": _annualized_vol_proxy(obs),
        "buy_hold_max_drawdown_proxy": _max_drawdown_proxy(obs),
        "avg_daily_return_proxy": float(np.mean(strategy_returns)),
    }


def _baseline_comparison_summary(
    baseline_metrics: dict[str, Any],
    *,
    hmm_val_avg_ll: float,
    hmm_test_avg_ll: float,
) -> dict[str, Any]:
    baselines = baseline_metrics.get("baselines", {})
    best_name = baseline_metrics.get("summary", {}).get("best_baseline_name")
    best_payload = baselines.get(best_name, {}) if isinstance(baselines, dict) and best_name else {}
    best_val = (
        float(best_payload.get("val", {}).get("avg_log_likelihood"))
        if isinstance(best_payload, dict) and best_payload.get("val", {}).get("avg_log_likelihood") is not None
        else None
    )
    best_test = (
        float(best_payload.get("test", {}).get("avg_log_likelihood"))
        if isinstance(best_payload, dict) and best_payload.get("test", {}).get("avg_log_likelihood") is not None
        else None
    )
    return {
        "best_baseline_name": best_name,
        "best_baseline_val_avg_log_likelihood": best_val,
        "best_baseline_test_avg_log_likelihood": best_test,
        "hmm_val_avg_log_likelihood": float(hmm_val_avg_ll),
        "hmm_test_avg_log_likelihood": float(hmm_test_avg_ll),
        "delta_vs_hmm_val_avg_log_likelihood": (
            None if best_val is None else float(best_val - float(hmm_val_avg_ll))
        ),
        "delta_vs_hmm_test_avg_log_likelihood": (
            None if best_test is None else float(best_test - float(hmm_test_avg_ll))
        ),
    }


def _build_events_payload(
    run_id: str,
    dates: list[str],
    returns: np.ndarray,
    states: np.ndarray,
    labels_map: dict[str, str],
) -> dict[str, Any]:
    if len(dates) != states.shape[0] or returns.shape[0] != states.shape[0]:
        raise ValueError("Events payload inputs must align on length.")
    if states.shape[0] == 0:
        return {"run_id": run_id, "n_events": 0, "events": []}

    seq_states = states.astype(np.int64)
    seq_returns = returns.astype(np.float64)
    events: list[dict[str, Any]] = []
    start_idx = 0
    current = int(seq_states[0])
    segment_index = 0

    for idx in range(1, seq_states.shape[0]):
        state_i = int(seq_states[idx])
        if state_i != current:
            end_idx = idx - 1
            segment_returns = seq_returns[start_idx : end_idx + 1]
            length = int(end_idx - start_idx + 1)
            events.append(
                {
                    "segment_index": int(segment_index),
                    "state": int(current),
                    "label": labels_map.get(str(current), f"regime_{current}"),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_date": str(dates[start_idx]),
                    "end_date": str(dates[end_idx]),
                    "length": length,
                    "duration_days": length,
                    "cumulative_log_return": float(np.sum(segment_returns)),
                    "mean_return": float(np.mean(segment_returns)),
                    "realized_vol": (
                        float(np.std(segment_returns, ddof=1))
                        if segment_returns.shape[0] > 1
                        else 0.0
                    ),
                }
            )
            segment_index += 1
            start_idx = idx
            current = state_i

    end_idx = seq_states.shape[0] - 1
    segment_returns = seq_returns[start_idx : end_idx + 1]
    length = int(end_idx - start_idx + 1)
    events.append(
        {
            "segment_index": int(segment_index),
            "state": int(current),
            "label": labels_map.get(str(current), f"regime_{current}"),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_date": str(dates[start_idx]),
            "end_date": str(dates[end_idx]),
            "length": length,
            "duration_days": length,
            "cumulative_log_return": float(np.sum(segment_returns)),
            "mean_return": float(np.mean(segment_returns)),
            "realized_vol": (
                float(np.std(segment_returns, ddof=1))
                if segment_returns.shape[0] > 1
                else 0.0
            ),
        }
    )
    return {"run_id": run_id, "n_events": int(len(events)), "events": events}


def _event_classifier_summary(events_payload: dict[str, Any]) -> dict[str, Any]:
    events = events_payload.get("events", [])
    if not isinstance(events, list):
        events = []

    counts: dict[str, int] = {}
    duration_accumulator: dict[str, list[float]] = {}
    longest: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        label = str(event.get("label", "unknown"))
        duration = float(event.get("duration_days", event.get("length", 0)))
        counts[label] = counts.get(label, 0) + 1
        duration_accumulator.setdefault(label, []).append(duration)
        longest.append(
            {
                "label": label,
                "start_date": event.get("start_date"),
                "end_date": event.get("end_date"),
                "duration_days": duration,
            }
        )

    avg_duration: dict[str, float] = {}
    for label, durations in duration_accumulator.items():
        avg_duration[label] = float(np.mean(np.asarray(durations, dtype=np.float64)))

    longest_sorted = sorted(longest, key=lambda row: float(row.get("duration_days", 0.0)), reverse=True)

    return {
        "event_counts_by_label": counts,
        "avg_event_duration_days_by_label": avg_duration,
        "top_5_longest_events": longest_sorted[:5],
    }


def _build_forecast_eval_payload(
    run_id: str,
    created_at_utc: str,
    returns: np.ndarray,
    filter_probs: np.ndarray,
    transition_matrix: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    interval: float,
    include_quantiles: bool = False,
    window: int = 120,
    max_horizon: int = 5,
) -> dict[str, Any]:
    """Build rolling forecast evaluation metrics from in-memory artifacts."""
    obs = np.asarray(returns, dtype=np.float64)
    probs = np.asarray(filter_probs, dtype=np.float64)
    a_mat = np.asarray(transition_matrix, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    z = NormalDist().inv_cdf(0.5 + 0.5 * float(interval))
    z05 = NormalDist().inv_cdf(0.05)
    z95 = NormalDist().inv_cdf(0.95)
    horizon_rows: list[dict[str, Any]] = []

    n_obs = int(obs.shape[0])
    for horizon in range(1, max_horizon + 1):
        if n_obs <= horizon:
            horizon_rows.append(
                {
                    "horizon": int(horizon),
                    "n_samples": 0,
                    "mae": None,
                    "coverage": None,
                }
            )
            continue

        origin_end = n_obs - horizon
        origin_start = max(0, origin_end - int(window))
        abs_errors: list[float] = []
        coverages: list[float] = []
        quantile_coverages: list[float] = []

        for t in range(origin_start, origin_end):
            state_probs = np.asarray(probs[t], dtype=np.float64)
            for _ in range(horizon):
                state_probs = state_probs @ a_mat

            exp_return, exp_vol = probability_weighted_moments(
                state_probs=state_probs,
                mu=mu_arr,
                sigma=sigma_arr,
            )
            realized = float(obs[t + horizon])
            abs_errors.append(abs(realized - exp_return))
            lower = exp_return - z * exp_vol
            upper = exp_return + z * exp_vol
            coverages.append(1.0 if lower <= realized <= upper else 0.0)
            if include_quantiles:
                p05 = exp_return + z05 * exp_vol
                p95 = exp_return + z95 * exp_vol
                quantile_coverages.append(1.0 if p05 <= realized <= p95 else 0.0)

        row = {
            "horizon": int(horizon),
            "n_samples": int(len(abs_errors)),
            "mae": (float(np.mean(abs_errors)) if abs_errors else None),
            "coverage": (float(np.mean(coverages)) if coverages else None),
        }
        if include_quantiles:
            row["quantile_coverage_p05_p95"] = (
                float(np.mean(quantile_coverages)) if quantile_coverages else None
            )
        horizon_rows.append(row)

    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "window": int(window),
        "interval": float(interval),
        "include_quantiles": bool(include_quantiles),
        "horizons": horizon_rows,
    }


def resolve_run_dir(run_id: Optional[str], runs_root: str | Path = "runs") -> Path:
    root = Path(runs_root)
    if run_id:
        path = root / run_id
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return path

    latest_path = root / "latest_run.txt"
    if not latest_path.exists():
        raise FileNotFoundError(
            "No latest run pointer found. Train a model with POST /fit first."
        )

    latest_run_id = latest_path.read_text(encoding="utf-8").strip()
    path = root / latest_run_id
    if not path.exists():
        raise FileNotFoundError(f"Latest run directory missing: {latest_run_id}")
    return path


def train_model_run(
    config_path: str | Path = "configs/default.yaml",
    force_refresh: bool = False,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Train a new model run and persist all artifacts under runs/<run_id>/."""
    cfg = load_config(config_path)

    n_states = int(cfg.model.get("n_states", 3))
    if n_states != 3:
        raise ValueError("This project defaults to and currently enforces n_states=3.")

    seed = int(cfg.model["seed"])
    set_global_seed(seed)

    client = EIAWTIClient(
        EIAClientConfig(
            cache_dir=Path(cfg.data["raw_dir"]),
            cache_filename=str(cfg.data.get("cache_filename", "wti_cushing_daily.csv")),
        )
    )

    prices = client.fetch_daily_wti(
        force_refresh=force_refresh,
        start_date=cfg.data.get("start_date"),
        end_date=cfg.data.get("end_date"),
    )
    try:
        features = compute_log_returns(prices)
    except ValueError as exc:
        if "Need at least 10 positive price observations" in str(exc):
            raise ValueError(_log_return_short_series_message(prices)) from exc
        raise

    data_meta_payload = _build_data_meta_payload(prices, features)

    dates = [d.strftime("%Y-%m-%d") for d in features["date"]]
    returns = features["log_return"].to_numpy(dtype=np.float64)

    train_obs, val_obs, test_obs = time_based_split(
        returns,
        train_frac=float(cfg.split["train_frac"]),
        val_frac=float(cfg.split["val_frac"]),
    )

    from models.hmm_tfp import GaussianHMMTFP, HMMTrainingConfig

    model = GaussianHMMTFP(n_states=n_states, seed=seed)
    history = model.fit(
        train_observations=train_obs,
        val_observations=val_obs,
        config=HMMTrainingConfig(
            learning_rate=float(cfg.model["learning_rate"]),
            max_epochs=int(cfg.model["max_epochs"]),
            patience=int(cfg.model["patience"]),
            min_delta=float(cfg.model["min_delta"]),
        ),
    )

    train_ll = float(model.log_prob(train_obs).numpy())
    val_ll = float(model.log_prob(val_obs).numpy())
    test_ll = float(model.log_prob(test_obs).numpy())

    params = model.get_params()
    model_params_payload = {
        "n_states": int(n_states),
        "initial_probs": [float(x) for x in params["initial_probs"]],
        "initial_logits": [float(x) for x in params["initial_logits"]],
        "transition_matrix": [
            [float(x) for x in row] for row in params["transition_matrix"]
        ],
        "mu": [float(x) for x in params["mu"]],
        "sigma": [float(x) for x in params["sigma"]],
    }
    regime_labels = assign_regime_labels(
        mu=params["mu"],
        sigma=params["sigma"],
        transition_matrix=params["transition_matrix"],
    )
    sorted_state_idx = np.argsort(np.asarray(params["sigma"], dtype=np.float64), kind="stable")
    ordered_labels = [regime_labels.get(str(int(i)), f"regime_{int(i)}") for i in sorted_state_idx]

    filter_probs = forward_filter_probs(
        observations=returns,
        initial_probs=params["initial_probs"],
        transition_matrix=params["transition_matrix"],
        mu=params["mu"],
        sigma=params["sigma"],
    )
    viterbi_states = model.viterbi(returns)
    if viterbi_states.shape[0] != returns.shape[0]:
        raise ValueError("Viterbi decode output length does not match returns length.")
    robust_sigma = _robust_sigma_by_regime(
        returns=returns,
        states=viterbi_states,
        fallback_sigma=params["sigma"],
    )
    model_params_payload["robust_sigma"] = [float(x) for x in robust_sigma]

    predict_payload = predict_proba_payload(dates=dates, returns=returns, filter_probs=filter_probs)
    viterbi_payload = {
        "dates": dates,
        "states": [int(s) for s in viterbi_states],
        "labels": [regime_labels.get(str(int(s)), f"regime_{int(s)}") for s in viterbi_states],
        "label_mapping": regime_labels,
    }
    regime_summary = build_regime_summary(
        dates=dates,
        returns=returns,
        filter_probs=filter_probs,
        transition_matrix=params["transition_matrix"],
        mu=params["mu"],
        sigma=params["sigma"],
        regime_labels=regime_labels,
    )

    transition_payload = {
        "transition_matrix": params["transition_matrix"].tolist(),
        "state_labels": ordered_labels,
        "label_mapping": regime_labels,
    }

    occupancy = np.asarray(regime_summary.get("state_occupancy", []), dtype=np.float64)
    regime_diagnostics: list[dict[str, Any]] = []
    for row in regime_summary.get("regimes", []):
        regime_idx = int(row["regime"])
        occ = (
            float(occupancy[regime_idx])
            if occupancy.ndim == 1 and regime_idx < occupancy.shape[0]
            else float(row.get("avg_posterior_probability", 0.0))
        )
        regime_diagnostics.append(
            {
                "regime": regime_idx,
                "label": row.get("label", f"regime_{regime_idx}"),
                "occupancy": occ,
                "avg_posterior_probability": float(row.get("avg_posterior_probability", occ)),
                "implied_avg_duration_days": row.get("implied_avg_duration_days"),
            }
        )

    default_forecast = forecast_predictive_distribution(
        last_posterior=filter_probs[-1],
        transition_matrix=params["transition_matrix"],
        mu=params["mu"],
        sigma=params["sigma"],
        horizon=int(cfg.forecast.get("default_horizon", 10)),
        interval=float(cfg.forecast.get("interval", 0.95)),
        last_date=dates[-1],
    )

    run_id_final = run_id or _make_run_id()
    run_root = Path(cfg.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    run_path = run_root / run_id_final
    run_path.mkdir(parents=True, exist_ok=True)

    run_created_at = datetime.now(timezone.utc).isoformat()
    baseline_metrics = evaluate_baselines(
        train_obs=train_obs,
        val_obs=val_obs,
        test_obs=test_obs,
        seed=seed,
    )
    baseline_comparison = _baseline_comparison_summary(
        baseline_metrics,
        hmm_val_avg_ll=float(val_ll / float(val_obs.size)),
        hmm_test_avg_ll=float(test_ll / float(test_obs.size)),
    )
    baselines_payload = {
        "run_id": run_id_final,
        "created_at_utc": run_created_at,
        "split_sizes": {
            "train": int(train_obs.size),
            "val": int(val_obs.size),
            "test": int(test_obs.size),
        },
        "baselines": baseline_metrics.get("baselines", {}),
        "summary": {
            **baseline_metrics.get("summary", {}),
            **baseline_comparison,
        },
    }

    events_payload = _build_events_payload(
        run_id=run_id_final,
        dates=dates,
        returns=returns,
        states=viterbi_states,
        labels_map=regime_labels,
    )
    forecast_eval_payload = _build_forecast_eval_payload(
        run_id=run_id_final,
        created_at_utc=run_created_at,
        returns=returns,
        filter_probs=filter_probs,
        transition_matrix=params["transition_matrix"],
        mu=params["mu"],
        sigma=params["sigma"],
        interval=float(cfg.forecast.get("interval", 0.95)),
        include_quantiles=bool(cfg.forecast.get("include_quantiles", False)),
        window=120,
        max_horizon=5,
    )
    backtest_payload = _build_backtest_payload(
        run_id=run_id_final,
        created_at_utc=run_created_at,
        returns=returns,
        states=viterbi_states,
        labels_map=regime_labels,
        sigma=params["sigma"],
    )

    metrics_payload = {
        "train_log_likelihood": train_ll,
        "val_log_likelihood": val_ll,
        "test_log_likelihood": test_ll,
        "train_avg_log_likelihood": train_ll / float(train_obs.size),
        "val_avg_log_likelihood": val_ll / float(val_obs.size),
        "test_avg_log_likelihood": test_ll / float(test_obs.size),
        "history": history,
        "split_sizes": {
            "train": int(train_obs.size),
            "val": int(val_obs.size),
            "test": int(test_obs.size),
        },
        "created_at_utc": run_created_at,
    }

    evaluation_payload = {
        "run_id": run_id_final,
        "created_at_utc": run_created_at,
        "n_obs": int(returns.shape[0]),
        "n_states": int(n_states),
        "metrics": {
            "best_val_log_likelihood": float(history.get("best_val_log_likelihood", val_ll)),
            "test_avg_log_likelihood": float(test_ll / float(test_obs.size)),
            "epochs_ran": int(history.get("epochs_ran", len(history.get("train_log_likelihood", [])))),
            "stopped_early": bool(history.get("stopped_early", False)),
        },
        "regime_diagnostics": {
            "implied_avg_duration_days": [
                row.get("implied_avg_duration_days") for row in regime_summary.get("regimes", [])
            ],
            "transition_entropy": _transition_entropy(params["transition_matrix"]),
            "regimes": regime_diagnostics,
            "viterbi_segment_stats": _viterbi_segment_stats(viterbi_states),
        },
        "event_classifier_summary": _event_classifier_summary(events_payload),
        "regime_stability": _regime_stability_metrics(
            states=viterbi_states,
            labels_map=regime_labels,
            window_days=60,
        ),
        "baseline_comparison": baseline_comparison,
    }

    config_payload = {
        "run_id": run_id_final,
        "config_path": str(config_path),
        "run_dir": str(run_path),
        "loaded_config": {
            "run_dir": cfg.run_dir,
            "data": cfg.data,
            "split": cfg.split,
            "model": cfg.model,
            "forecast": cfg.forecast,
        },
    }

    artifact_filenames = [
        "config.json",
        "model_params.npz",
        "model_params.json",
        "evaluation.json",
        "metrics.json",
        "transition_matrix.json",
        "regime_labels.json",
        "regime_summary.json",
        "predict_proba.json",
        "viterbi_states.json",
        "events.json",
        "forecast_default.json",
        "forecast_eval.json",
        "baselines.json",
        "backtest.json",
        "data_meta.json",
        "manifest.json",
    ]

    manifest_payload = {
        "run_id": run_id_final,
        "created_at_utc": run_created_at,
        "config_hash": _config_hash(config_payload["loaded_config"]),
        "git_commit_hash": _safe_git_commit_hash(Path(__file__).resolve().parents[1]),
        "n_states": int(n_states),
        "seed": int(seed),
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "data_hash": data_meta_payload.get("dataset_hash"),
        "n_observations": int(returns.shape[0]),
        "artifacts": artifact_filenames,
    }

    _write_json(run_path / "config.json", config_payload)
    model.save(run_path / "model_params.npz")
    _write_json(run_path / "model_params.json", model_params_payload)
    _write_json(run_path / "evaluation.json", evaluation_payload)
    _write_json(run_path / "metrics.json", metrics_payload)
    _write_json(run_path / "transition_matrix.json", transition_payload)
    _write_json(run_path / "regime_labels.json", {"label_mapping": regime_labels})
    _write_json(run_path / "regime_summary.json", regime_summary)
    _write_json(run_path / "predict_proba.json", predict_payload)
    _write_json(run_path / "viterbi_states.json", viterbi_payload)
    _write_json(run_path / "events.json", events_payload)
    _write_json(run_path / "forecast_default.json", default_forecast)
    _write_json(run_path / "forecast_eval.json", forecast_eval_payload)
    _write_json(run_path / "baselines.json", baselines_payload)
    _write_json(run_path / "backtest.json", backtest_payload)
    _write_json(run_path / "data_meta.json", data_meta_payload)
    write_manifest_with_provenance(run_dir=run_path, manifest_payload=manifest_payload)

    (run_root / "latest_run.txt").write_text(run_id_final, encoding="utf-8")

    return {
        "run_id": run_id_final,
        "run_dir": str(run_path),
        "metrics": metrics_payload,
        "transition_matrix": transition_payload,
        "regime_summary": regime_summary,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
