"""API routes for fitting and querying model run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from models.infer import forecast_predictive_distribution
from models.train import resolve_run_dir, train_model_run

router = APIRouter()
RUNS_ROOT = Path("runs")


class FitRequest(BaseModel):
    """Request payload for training endpoint."""

    config_path: str = Field(default="configs/default.yaml")
    force_refresh: bool = Field(default=False)
    run_id: Optional[str] = Field(default=None)


@router.post("/fit")
def fit_model(request: FitRequest) -> dict[str, Any]:
    """Train model and persist artifacts under runs/<run_id>/."""
    try:
        result = train_model_run(
            config_path=request.config_path,
            force_refresh=request.force_refresh,
            run_id=request.run_id,
        )
        return {
            "run_id": result["run_id"],
            "run_dir": result["run_dir"],
            "metrics": result["metrics"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    """List available run directories under runs/ sorted newest-first."""
    if not RUNS_ROOT.exists():
        return {"runs": []}

    run_ids = sorted(
        [p.name for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_")],
        reverse=True,
    )
    return {"runs": run_ids}


@router.get("/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    """Return regime summary for a specific run id."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "regime_summary.json")


@router.get("/runs/{run_id}/plot")
def run_plot_path(run_id: str) -> dict[str, str]:
    """Return filesystem path to plot artifact for a specific run id."""
    run_path = _resolve_run_path(run_id)
    plot_path = run_path / "regimes.html"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {plot_path.name}")
    return {"run_id": run_id, "plot_path": str(plot_path)}


@router.get("/predict_proba")
def predict_proba(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return dates, observed returns, and per-regime filtering probabilities."""
    run_path = _resolve_run_path(run_id)
    payload_path = run_path / "predict_proba.json"
    return _read_json(payload_path)


@router.get("/transition_matrix")
def transition_matrix(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return the learned transition matrix."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "transition_matrix.json")


@router.get("/regime_summary")
def regime_summary(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return per-regime summary statistics for the requested run."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "regime_summary.json")


@router.get("/forecast")
def forecast(
    run_id: Optional[str] = Query(default=None),
    horizon: int = Query(default=10, ge=1, le=365),
    interval: float = Query(default=0.95, gt=0.0, lt=1.0),
) -> dict[str, Any]:
    """Return horizon-step predictive mean and uncertainty intervals."""
    run_path = _resolve_run_path(run_id)

    model_params = np.load(run_path / "model_params.npz")
    probs_payload = _read_json(run_path / "predict_proba.json")

    regime_probs = np.asarray(probs_payload["regime_probabilities"], dtype=np.float64)
    dates = probs_payload["dates"]
    if regime_probs.size == 0:
        raise HTTPException(status_code=500, detail="Empty regime probabilities in run.")

    payload = forecast_predictive_distribution(
        last_posterior=regime_probs[-1],
        transition_matrix=model_params["transition_matrix"],
        mu=model_params["mu"],
        sigma=model_params["sigma"],
        horizon=horizon,
        interval=interval,
        last_date=dates[-1],
    )
    payload["run_id"] = run_path.name
    return payload


def _resolve_run_path(run_id: Optional[str]) -> Path:
    try:
        return resolve_run_dir(run_id=run_id, runs_root=RUNS_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
