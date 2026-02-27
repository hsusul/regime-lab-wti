"""Training pipeline for WTI 3-regime Gaussian HMM using TensorFlow Probability."""

from __future__ import annotations

import json
import random
import subprocess
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from energy_data.eia_client import EIAClientConfig, EIAWTIClient
from energy_data.features import compute_log_returns, time_based_split
from models.hmm_tfp import GaussianHMMTFP, HMMTrainingConfig
from models.infer import (
    assign_regime_labels,
    build_regime_summary,
    forecast_predictive_distribution,
    forward_filter_probs,
    predict_proba_payload,
)


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
    tf.keras.utils.set_random_seed(seed)


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

    dates = [d.strftime("%Y-%m-%d") for d in features["date"]]
    returns = features["log_return"].to_numpy(dtype=np.float64)

    train_obs, val_obs, test_obs = time_based_split(
        returns,
        train_frac=float(cfg.split["train_frac"]),
        val_frac=float(cfg.split["val_frac"]),
    )

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
        "metrics.json",
        "transition_matrix.json",
        "regime_labels.json",
        "regime_summary.json",
        "predict_proba.json",
        "viterbi_states.json",
        "forecast_default.json",
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
        "n_observations": int(returns.shape[0]),
        "artifacts": artifact_filenames,
    }

    _write_json(run_path / "config.json", config_payload)
    model.save(run_path / "model_params.npz")
    _write_json(run_path / "metrics.json", metrics_payload)
    _write_json(run_path / "transition_matrix.json", transition_payload)
    _write_json(run_path / "regime_labels.json", {"label_mapping": regime_labels})
    _write_json(run_path / "regime_summary.json", regime_summary)
    _write_json(run_path / "predict_proba.json", predict_payload)
    _write_json(run_path / "viterbi_states.json", viterbi_payload)
    _write_json(run_path / "forecast_default.json", default_forecast)
    _write_json(run_path / "manifest.json", manifest_payload)

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
