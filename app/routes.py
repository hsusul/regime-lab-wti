"""API routes for fitting and querying model run artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from models.infer import (
    forecast_predictive_distribution,
    load_model_params_json,
    probability_weighted_moments,
)
from models.train import resolve_run_dir, train_model_run

router = APIRouter()
RUNS_ROOT = Path("runs")
PINNED_RUN_FILENAME = "pinned_run.txt"


class FitRequest(BaseModel):
    """Request payload for training endpoint."""

    config_path: str = Field(default="configs/default.yaml")
    force_refresh: bool = Field(default=False)
    run_id: Optional[str] = Field(default=None)


class PredictCurrentRequest(BaseModel):
    """Request payload for current regime endpoint."""

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
    return {"runs": _list_run_ids()}


@router.get("/runs/pinned")
def pinned_run() -> dict[str, str]:
    """Return the currently pinned run id and path."""
    run_id = _read_pinned_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return {"run_id": run_id, "run_dir": str(run_path)}


@router.post("/runs/{run_id}/pin")
def pin_run(run_id: str) -> dict[str, str]:
    """Pin a specific run id for production-style workflows."""
    _validate_run_id_for_pin(run_id)
    run_path = RUNS_ROOT / run_id
    if not run_path.exists() or not run_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found for pinning: {run_id}")

    _pinned_run_file().write_text(run_id, encoding="utf-8")
    return {"pinned_run_id": run_id}


@router.get("/runs/latest")
def latest_run() -> dict[str, Any]:
    """Return lightweight metadata for the newest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return _latest_run_payload(run_id, run_path)


@router.get("/runs/latest/summary")
def latest_run_summary() -> dict[str, Any]:
    """Return regime summary for latest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "regime_summary.json", run_id=run_id)


@router.get("/runs/latest/evaluation")
def latest_run_evaluation() -> dict[str, Any]:
    """Return evaluation artifact for latest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "evaluation.json", run_id=run_id)


@router.get("/runs/latest/plot")
def latest_run_plot_path() -> dict[str, str]:
    """Return plot path for latest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    plot_path = run_path / "regimes.html"
    if not plot_path.exists():
        raise _artifact_not_found(run_id, [plot_path.name])
    return {"run_id": run_id, "plot_path": str(plot_path)}


@router.get("/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    """Return regime summary for a specific run id."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "regime_summary.json", run_id=run_id)


@router.get("/runs/{run_id}/evaluation")
def run_evaluation(run_id: str) -> dict[str, Any]:
    """Return evaluation artifact for a specific run id."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "evaluation.json", run_id=run_id)


@router.get("/runs/{run_id}/plot")
def run_plot_path(run_id: str) -> dict[str, str]:
    """Return filesystem path to plot artifact for a specific run id."""
    run_path = _resolve_run_path(run_id)
    plot_path = run_path / "regimes.html"
    if not plot_path.exists():
        raise _artifact_not_found(run_id, [plot_path.name])
    return {"run_id": run_id, "plot_path": str(plot_path)}


@router.get("/runs/{run_id}/plot/html")
def run_plot_html(run_id: str) -> Response:
    """Return plot HTML artifact bytes for a specific run id."""
    run_path = _resolve_run_path(run_id)
    plot_path = run_path / "regimes.html"
    if not plot_path.exists():
        raise _artifact_not_found(run_id, [plot_path.name])
    html = plot_path.read_text(encoding="utf-8")
    return Response(content=html, media_type="text/html; charset=utf-8")


@router.get("/runs/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, Any]:
    """Return available artifact filenames for a run."""
    run_path = _resolve_run_path(run_id)
    artifacts = sorted([p.name for p in run_path.iterdir() if p.is_file()])
    return {"run_id": run_id, "artifacts": artifacts}


@router.get("/runs/{run_id}/manifest")
def run_manifest(run_id: str) -> dict[str, Any]:
    """Return manifest payload for a run."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "manifest.json", run_id=run_id)


@router.get("/runs/{run_id}/model")
def run_model(run_id: str) -> dict[str, Any]:
    """Return compact model parameters for a run."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "model_params.json", run_id=run_id)


@router.get("/ui")
def ui_page() -> HTMLResponse:
    """Simple demo UI for browsing runs and regime outputs."""
    title = html.escape("WTI Regime Monitor")
    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #111; background: #f9fafb; }}
    h1 {{ margin: 0 0 12px 0; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    select, button {{ padding: 6px 10px; font-size: 14px; }}
    .box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    #currentRegime {{ font-size: 16px; font-weight: 600; }}
    #status {{ color: #444; font-size: 14px; }}
    #summaryPre {{ margin: 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    iframe {{ width: 100%; height: 760px; border: 1px solid #ccc; border-radius: 8px; background: #fff; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div id="pinnedInfo"><strong>Pinned run:</strong> none</div>
  <div id="status">Loading runs...</div>
  <div class="row">
    <label for="runSelect"><strong>Run:</strong></label>
    <select id="runSelect"></select>
    <button id="refreshBtn" type="button">Refresh</button>
    <button id="pinBtn" type="button">Pin this run</button>
  </div>
  <div class="box">
    <div id="currentRegime">Current regime: N/A</div>
    <div id="currentMeta"></div>
  </div>
  <div class="box">
    <strong>Run summary</strong>
    <pre id="summaryPre">N/A</pre>
  </div>
  <iframe id="plotFrame" src="about:blank" title="Regime Plot"></iframe>

  <script>
    const runSelect = document.getElementById("runSelect");
    const refreshBtn = document.getElementById("refreshBtn");
    const pinBtn = document.getElementById("pinBtn");
    const statusEl = document.getElementById("status");
    const pinnedInfoEl = document.getElementById("pinnedInfo");
    const currentRegimeEl = document.getElementById("currentRegime");
    const currentMetaEl = document.getElementById("currentMeta");
    const summaryPre = document.getElementById("summaryPre");
    const plotFrame = document.getElementById("plotFrame");

    async function fetchJson(url, options) {{
      const res = await fetch(url, options || {{}});
      const data = await res.json().catch(() => ({{}}));
      if (!res.ok) {{
        const detail = data && data.detail ? data.detail : ("HTTP " + res.status);
        throw new Error(detail);
      }}
      return data;
    }}

    async function loadPinned() {{
      try {{
        const pinned = await fetchJson("/runs/pinned");
        pinnedInfoEl.innerHTML = "<strong>Pinned run:</strong> " + pinned.run_id;
      }} catch (err) {{
        pinnedInfoEl.innerHTML = "<strong>Pinned run:</strong> none";
      }}
    }}

    function formatSummary(summary) {{
      const start = summary.start_date || "N/A";
      const end = summary.end_date || "N/A";
      const regimes = Array.isArray(summary.regimes) ? summary.regimes : [];
      const rows = regimes.map(r => {{
        const label = r.label || ("regime_" + r.regime);
        const sigma = (typeof r.sigma === "number") ? r.sigma.toFixed(4) : "N/A";
        return `- ${{label}}: sigma=${{sigma}}`;
      }});
      return `start: ${{start}}\\nend:   ${{end}}\\n${{rows.join("\\n")}}`;
    }}

    async function loadRun(runId) {{
      statusEl.textContent = "Loading run " + runId + "...";
      try {{
        const [summary, current] = await Promise.all([
          fetchJson("/runs/" + encodeURIComponent(runId) + "/summary"),
          fetchJson("/predict_current?include_probs=true", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ run_id: runId }})
          }})
        ]);

        summaryPre.textContent = formatSummary(summary);

        const label = current.label || ("regime_" + current.state);
        currentRegimeEl.textContent = "Current regime: " + label + " (" + current.source + ")";
        const pLabel = (current.p_label === null || current.p_label === undefined)
          ? "N/A"
          : current.p_label.toFixed(4);
        currentMetaEl.textContent = "as_of=" + current.as_of_date + " | p_label=" + pLabel;

        plotFrame.src = "/runs/" + encodeURIComponent(runId) + "/plot/html";
        statusEl.textContent = "Loaded run " + runId;
      }} catch (err) {{
        statusEl.textContent = "Error: " + err.message;
      }}
    }}

    async function refreshRuns() {{
      try {{
        const payload = await fetchJson("/runs");
        const runs = Array.isArray(payload.runs) ? payload.runs : [];
        await loadPinned();
        runSelect.innerHTML = "";
        if (runs.length === 0) {{
          statusEl.textContent = "No runs available. Train a model first.";
          summaryPre.textContent = "N/A";
          currentRegimeEl.textContent = "Current regime: N/A";
          currentMetaEl.textContent = "";
          plotFrame.src = "about:blank";
          return;
        }}

        for (const runId of runs) {{
          const opt = document.createElement("option");
          opt.value = runId;
          opt.textContent = runId;
          runSelect.appendChild(opt);
        }}
        await loadRun(runs[0]);
      }} catch (err) {{
        statusEl.textContent = "Error loading runs: " + err.message;
      }}
    }}

    runSelect.addEventListener("change", () => {{
      if (runSelect.value) {{
        loadRun(runSelect.value);
      }}
    }});
    pinBtn.addEventListener("click", async () => {{
      if (!runSelect.value) {{
        return;
      }}
      try {{
        const runId = runSelect.value;
        await fetchJson("/runs/" + encodeURIComponent(runId) + "/pin", {{
          method: "POST"
        }});
        await loadPinned();
        statusEl.textContent = "Pinned run " + runId;
      }} catch (err) {{
        statusEl.textContent = "Error pinning run: " + err.message;
      }}
    }});
    refreshBtn.addEventListener("click", refreshRuns);
    refreshRuns();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_body)


@router.post("/predict_current")
def predict_current_with_options(
    request: Optional[PredictCurrentRequest] = None,
    include_probs: bool = Query(default=False),
) -> dict[str, Any]:
    """Return current regime for the last observation date with optional probs."""
    return _predict_current_impl(request=request, include_probs=include_probs)


def _predict_current_impl(
    request: Optional[PredictCurrentRequest], include_probs: bool
) -> dict[str, Any]:
    """Shared implementation for /predict_current."""
    run_id = request.run_id if request is not None else None
    if run_id is None:
        run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)

    labels_map = _read_label_mapping(run_path)
    viterbi_path = run_path / "viterbi_states.json"
    proba_path = run_path / "predict_proba.json"
    model_params = _load_model_params_or_none(run_id=run_id, run_path=run_path)

    viterbi_exists = viterbi_path.exists()
    should_load_probs = include_probs or not viterbi_exists
    prob_info = (
        _load_last_prob_info(run_id=run_id, proba_path=proba_path, labels_map=labels_map)
        if should_load_probs
        else None
    )
    n_states_hint: int | None = None
    if model_params is not None and "transition_matrix" in model_params:
        n_states_hint = int(np.asarray(model_params["transition_matrix"]).shape[0])
    elif prob_info is not None:
        n_states_hint = int(len(prob_info["raw_state_probs"]))
    if n_states_hint is not None:
        labels_map = _with_default_labels(labels_map, n_states_hint)

    if viterbi_exists:
        viterbi = _read_json(viterbi_path, run_id=run_id)
        dates = viterbi.get("dates", [])
        states = viterbi.get("states", [])
        labels = viterbi.get("labels", [])
        if not dates or not states:
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_id} has empty artifact: viterbi_states.json",
            )
        if not labels_map:
            labels_map = _with_default_labels(labels_map, int(max(states)) + 1)
        state = int(states[-1])
        label = (
            str(labels[-1])
            if len(labels) == len(states)
            else labels_map.get(str(state), f"regime_{state}")
        )
        expected_return = None
        expected_vol = None
        if prob_info is not None and model_params is not None:
            expected_return, expected_vol = probability_weighted_moments(
                state_probs=np.asarray(prob_info["raw_state_probs"], dtype=np.float64),
                mu=model_params["mu"],
                sigma=model_params["sigma"],
            )

        payload: dict[str, Any] = {
            "run_id": run_id,
            "as_of_date": str(dates[-1]),
            "as_of": str(dates[-1]),
            "state": state,
            "label": label,
            "p_label": None,
            "source": "viterbi",
            "label_mapping": labels_map,
            "expected_return": expected_return,
            "expected_vol": expected_vol,
        }
        if prob_info is not None and state < len(prob_info["raw_state_probs"]):
            payload["p_label"] = float(prob_info["raw_state_probs"][state])
        if include_probs:
            payload["probs"] = (
                prob_info["label_probs"] if prob_info is not None else None
            )
            payload["raw_state_probs"] = (
                prob_info["raw_state_probs"] if prob_info is not None else None
            )
        return payload

    if prob_info is not None:
        state = int(prob_info["state"])
        expected_return = None
        expected_vol = None
        if model_params is not None:
            expected_return, expected_vol = probability_weighted_moments(
                state_probs=np.asarray(prob_info["raw_state_probs"], dtype=np.float64),
                mu=model_params["mu"],
                sigma=model_params["sigma"],
            )

        payload = {
            "run_id": run_id,
            "as_of_date": str(prob_info["as_of_date"]),
            "as_of": str(prob_info["as_of_date"]),
            "state": state,
            "label": labels_map.get(str(state), f"regime_{state}"),
            "p_label": float(prob_info["p_label"]),
            "source": "filtering_argmax",
            "label_mapping": labels_map,
            "expected_return": expected_return,
            "expected_vol": expected_vol,
        }
        if include_probs:
            payload["probs"] = prob_info["label_probs"]
            payload["raw_state_probs"] = prob_info["raw_state_probs"]
        return payload

    raise _artifact_not_found(run_id, ["viterbi_states.json", "predict_proba.json"])


@router.get("/predict_proba")
def predict_proba(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return dates, observed returns, and per-regime filtering probabilities."""
    run_path = _resolve_run_path(run_id)
    payload_path = run_path / "predict_proba.json"
    return _read_json(payload_path, run_id=run_path.name)


@router.get("/transition_matrix")
def transition_matrix(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return the learned transition matrix."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "transition_matrix.json", run_id=run_path.name)


@router.get("/regime_summary")
def regime_summary(run_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Return per-regime summary statistics for the requested run."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "regime_summary.json", run_id=run_path.name)


@router.get("/forecast")
def forecast(
    run_id: Optional[str] = Query(default=None),
    horizon: int = Query(default=10, ge=1, le=365),
    interval: float = Query(default=0.95, gt=0.0, lt=1.0),
) -> dict[str, Any]:
    """Return horizon-step predictive mean and uncertainty intervals."""
    run_path = _resolve_run_path(run_id)

    model_params = _load_forecast_model_params(run_path=run_path)
    probs_payload = _read_json(run_path / "predict_proba.json", run_id=run_path.name)

    regime_probs = np.asarray(probs_payload["regime_probabilities"], dtype=np.float64)
    dates = probs_payload["dates"]
    if regime_probs.size == 0:
        raise HTTPException(status_code=500, detail="Empty regime probabilities in run.")

    payload = forecast_predictive_distribution(
        last_posterior=regime_probs[-1],
        transition_matrix=np.asarray(model_params["transition_matrix"], dtype=np.float64),
        mu=np.asarray(model_params["mu"], dtype=np.float64),
        sigma=np.asarray(model_params["sigma"], dtype=np.float64),
        horizon=horizon,
        interval=interval,
        last_date=dates[-1],
    )
    payload["run_id"] = run_path.name
    return payload


@router.get("/forecast_v2")
def forecast_v2(
    run_id: Optional[str] = Query(default=None),
    use_pinned: bool = Query(default=False),
    horizon: int = Query(default=10, ge=1, le=365),
) -> dict[str, Any]:
    """Return structured forecast moments with semantic state probabilities."""
    resolved_run_id = _resolve_effective_run_id(run_id=run_id, use_pinned=use_pinned)
    run_path = _resolve_run_path(resolved_run_id)

    model_params = _load_forecast_model_params(run_path=run_path)
    probs_payload = _read_json(run_path / "predict_proba.json", run_id=run_path.name)
    regime_probabilities = np.asarray(probs_payload.get("regime_probabilities", []), dtype=np.float64)
    dates = probs_payload.get("dates", [])
    if regime_probabilities.ndim != 2 or regime_probabilities.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_path.name} has malformed artifact: predict_proba.json",
        )
    if len(dates) != regime_probabilities.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_path.name} has malformed artifact: predict_proba.json",
        )

    transition_matrix = np.asarray(model_params["transition_matrix"], dtype=np.float64)
    mu = np.asarray(model_params["mu"], dtype=np.float64)
    sigma = np.asarray(model_params["sigma"], dtype=np.float64)
    current = np.asarray(regime_probabilities[-1], dtype=np.float64)

    n_states = int(current.shape[0])
    labels_map = _with_default_labels(_read_label_mapping(run_path), n_states)
    expected_index = 1.0
    rows: list[dict[str, Any]] = []

    for step in range(1, horizon + 1):
        current = current @ transition_matrix
        exp_return, exp_vol = probability_weighted_moments(
            state_probs=current,
            mu=mu,
            sigma=sigma,
        )
        expected_index *= float(np.exp(exp_return))

        rows.append(
            {
                "horizon": int(step),
                "state_probs": {
                    labels_map.get(str(i), f"regime_{i}"): float(current[i])
                    for i in range(n_states)
                },
                "expected_return": float(exp_return),
                "expected_vol": float(exp_vol),
                "expected_price_index": float(expected_index),
            }
        )

    return {
        "run_id": run_path.name,
        "as_of_date": str(dates[-1]),
        "horizon": int(horizon),
        "forecast": rows,
    }


def _resolve_run_path(run_id: Optional[str]) -> Path:
    try:
        return resolve_run_dir(run_id=run_id, runs_root=RUNS_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _list_run_ids() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        [p.name for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_")],
        reverse=True,
    )


def _latest_run_id_or_404() -> str:
    run_ids = _list_run_ids()
    if not run_ids:
        raise HTTPException(status_code=404, detail="No runs found under runs/")
    return run_ids[0]


def _resolve_effective_run_id(run_id: str | None, use_pinned: bool) -> str:
    if run_id is not None:
        return run_id
    if use_pinned:
        return _read_pinned_run_id_or_404()
    return _latest_run_id_or_404()


def _pinned_run_file() -> Path:
    return RUNS_ROOT / PINNED_RUN_FILENAME


def _validate_run_id_for_pin(run_id: str) -> None:
    if not run_id.startswith("run_"):
        raise HTTPException(
            status_code=404,
            detail=f"Only run ids starting with 'run_' can be pinned: {run_id}",
        )


def _read_pinned_run_id_or_404() -> str:
    pinned_path = _pinned_run_file()
    if not pinned_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Pinned run file not found: {PINNED_RUN_FILENAME}",
        )

    pinned_run_id = pinned_path.read_text(encoding="utf-8").strip()
    if not pinned_run_id:
        raise HTTPException(
            status_code=404,
            detail=f"Pinned run file is empty: {PINNED_RUN_FILENAME}",
        )

    _validate_run_id_for_pin(pinned_run_id)
    run_path = RUNS_ROOT / pinned_run_id
    if not run_path.exists() or not run_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Pinned run directory not found: {pinned_run_id}",
        )
    return pinned_run_id


def _latest_run_payload(run_id: str, run_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_path),
        "created_at_utc": None,
        "end_date": None,
        "last_label": None,
        "last_date": None,
    }

    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path, run_id=run_id)
        payload["created_at_utc"] = manifest.get("created_at_utc")
        payload["end_date"] = manifest.get("end_date")

    plot_meta_path = run_path / "plot_meta.json"
    if plot_meta_path.exists():
        plot_meta = _read_json(plot_meta_path, run_id=run_id)
        payload["created_at_utc"] = payload["created_at_utc"] or plot_meta.get("created_at_utc")
        payload["end_date"] = payload["end_date"] or plot_meta.get("end_date")
        payload["last_label"] = plot_meta.get("last_label")
        payload["last_date"] = plot_meta.get("last_date")

    if payload["last_label"] is None or payload["last_date"] is None:
        viterbi_path = run_path / "viterbi_states.json"
        if viterbi_path.exists():
            viterbi = _read_json(viterbi_path, run_id=run_id)
            dates = viterbi.get("dates", [])
            states = viterbi.get("states", [])
            labels = viterbi.get("labels", [])
            if dates and states:
                payload["last_date"] = payload["last_date"] or str(dates[-1])
                state = int(states[-1])
                if len(labels) == len(states):
                    payload["last_label"] = payload["last_label"] or str(labels[-1])
                else:
                    labels_map = _read_label_mapping(run_path)
                    payload["last_label"] = payload["last_label"] or labels_map.get(
                        str(state), f"regime_{state}"
                    )

    if payload["end_date"] is None:
        payload["end_date"] = payload["last_date"]
    return payload


def _read_label_mapping(run_path: Path) -> dict[str, str]:
    label_path = run_path / "regime_labels.json"
    if label_path.exists():
        payload = _read_json(label_path, run_id=run_path.name)
        mapping = payload.get("label_mapping", payload)
        if isinstance(mapping, dict):
            return {str(k): str(v) for k, v in mapping.items()}
    return {}


def _artifact_not_found(run_id: str, artifact_names: list[str]) -> HTTPException:
    names = ", ".join(artifact_names)
    return HTTPException(
        status_code=404,
        detail=f"Run {run_id} is missing artifact(s): {names}",
    )


def _with_default_labels(labels_map: dict[str, str], n_states: int) -> dict[str, str]:
    out = {str(k): str(v) for k, v in labels_map.items()}
    for idx in range(n_states):
        out.setdefault(str(idx), f"regime_{idx}")
    return out


def _load_forecast_model_params(run_path: Path) -> dict[str, np.ndarray]:
    model_json_path = run_path / "model_params.json"
    if model_json_path.exists():
        try:
            return load_model_params_json(model_json_path)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_path.name} has malformed artifact: model_params.json ({exc})",
            ) from exc

    npz_path = run_path / "model_params.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        return {
            "transition_matrix": np.asarray(npz["transition_matrix"], dtype=np.float64),
            "mu": np.asarray(npz["mu"], dtype=np.float64),
            "sigma": np.asarray(npz["sigma"], dtype=np.float64),
        }

    raise _artifact_not_found(run_path.name, ["model_params.json", "model_params.npz"])


def _load_model_params_or_none(run_id: str, run_path: Path) -> dict[str, np.ndarray] | None:
    model_json_path = run_path / "model_params.json"
    if not model_json_path.exists():
        return None
    try:
        return load_model_params_json(model_json_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed artifact: model_params.json ({exc})",
        ) from exc


def _load_last_prob_info(
    run_id: str,
    proba_path: Path,
    labels_map: dict[str, str],
) -> dict[str, Any] | None:
    if not proba_path.exists():
        return None
    probs = _read_json(proba_path, run_id=run_id)
    dates = probs.get("dates", [])
    regime_probabilities = np.asarray(probs.get("regime_probabilities", []), dtype=np.float64)
    if not dates or regime_probabilities.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has empty artifact: predict_proba.json",
        )
    if regime_probabilities.ndim != 2 or regime_probabilities.shape[0] != len(dates):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed artifact: predict_proba.json",
        )
    last_probs = regime_probabilities[-1]
    out: dict[str, float] = {}
    for idx, prob in enumerate(last_probs):
        label = labels_map.get(str(idx), f"regime_{idx}")
        out[label] = float(prob)
    state = int(np.argmax(last_probs))
    return {
        "as_of_date": str(dates[-1]),
        "state": state,
        "p_label": float(last_probs[state]),
        "raw_state_probs": [float(x) for x in last_probs],
        "label_probs": out,
    }


def _read_json(path: Path, run_id: str | None = None) -> dict[str, Any]:
    if not path.exists():
        if run_id is not None:
            raise _artifact_not_found(run_id, [path.name])
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
