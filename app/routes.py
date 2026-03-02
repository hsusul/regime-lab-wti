"""API routes for fitting and querying model run artifacts."""

from __future__ import annotations

import io
import html
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from models.infer import (
    forecast_predictive_distribution,
    load_model_params_json,
    probability_weighted_moments,
)
from models.provenance import sha256_file as _sha256_file
from models.provenance import write_manifest_with_provenance
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


class CompareRunsRequest(BaseModel):
    """Request payload for comparing multiple run artifacts."""

    run_ids: list[str] = Field(default_factory=list)


class RunTagsRequest(BaseModel):
    """Request payload for run tags metadata."""

    tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None)


class FreezeRunRequest(BaseModel):
    """Request payload for freezing a run."""

    reason: str | None = Field(default=None)


class TransitionAlertRequest(BaseModel):
    """Request payload for transition alert query."""

    run_id: str | None = Field(default=None)
    use_pinned: bool = Field(default=False)
    from_label: str | None = Field(default=None)
    to_label: str | None = Field(default=None)
    lookback_days: int = Field(default=30, ge=1, le=3650)


class EventSegmentResponse(BaseModel):
    """Response model for a contiguous Viterbi event segment."""

    model_config = ConfigDict(extra="allow")

    segment_index: int | None = None
    state: int
    label: str
    start_idx: int | None = None
    end_idx: int | None = None
    start_date: str
    end_date: str
    length: int | None = None
    duration_days: int | None = None
    cumulative_log_return: float | None = None
    mean_return: float | None = None
    realized_vol: float | None = None


class EventsResponseModel(BaseModel):
    """Response model for events endpoint."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    n_events: int
    events: list[EventSegmentResponse]


class ScorecardResponseModel(BaseModel):
    """Response model for scorecard endpoint."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    events_by_label: dict[str, int]
    last_regime: dict[str, Any]


class PredictCurrentResponseModel(BaseModel):
    """Response model for predict_current endpoint."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    as_of_date: str
    state: int
    label: str
    p_label: float | None
    source: str
    label_mapping: dict[str, str]


PREDICT_CURRENT_EXAMPLE = {
    "run_id": "run_20260301T120000Z_abcd1234",
    "as_of_date": "2026-03-01",
    "as_of": "2026-03-01",
    "state": 2,
    "label": "shock",
    "p_label": 0.72,
    "source": "viterbi",
    "label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"},
    "expected_return": -0.0085,
    "expected_vol": 0.032,
}

FORECAST_V2_EXAMPLE = {
    "run_id": "run_20260301T120000Z_abcd1234",
    "as_of_date": "2026-03-01",
    "horizon": 2,
    "forecast": [
        {
            "horizon": 1,
            "state_probs": {"low_vol": 0.62, "mid_vol": 0.28, "shock": 0.10},
            "expected_return": 0.0003,
            "expected_vol": 0.014,
            "expected_price_index": 1.0003,
        },
        {
            "horizon": 2,
            "state_probs": {"low_vol": 0.59, "mid_vol": 0.30, "shock": 0.11},
            "expected_return": 0.0002,
            "expected_vol": 0.0142,
            "expected_price_index": 1.0005,
        },
    ],
}


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
def list_runs(
    request: Request,
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    order: str = Query(default="desc"),
) -> dict[str, Any]:
    """List available run directories under runs/ with pagination and ordering."""
    run_ids = _list_run_ids(order=order)
    start = min(offset, len(run_ids))
    if "limit" in request.query_params:
        end = min(start + limit, len(run_ids))
    else:
        end = len(run_ids)
    return {"runs": run_ids[start:end]}


@router.get("/runs/active")
def active_run() -> dict[str, str]:
    """Resolve active run: pinned when available, otherwise latest."""
    run_id = _read_pinned_run_id_or_none() or _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return {"run_id": run_id, "run_dir": str(run_path)}


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


@router.post("/runs/unpin")
def unpin_run() -> dict[str, bool]:
    """Remove pinned run pointer if present."""
    pinned_path = _pinned_run_file()
    if not pinned_path.exists():
        return {"unpinned": False}
    pinned_path.unlink()
    return {"unpinned": True}


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


@router.get("/runs/latest/scorecard", response_model=ScorecardResponseModel)
def latest_run_scorecard() -> dict[str, Any]:
    """Return compact scorecard for latest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    return _build_scorecard_payload(run_id=run_id, run_path=run_path)


@router.get("/runs/latest/events", response_model=EventsResponseModel)
def latest_run_events(
    label: Optional[str] = Query(default=None),
    min_duration_days: Optional[int] = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Return regime events for latest run."""
    run_id = _latest_run_id_or_404()
    run_path = _resolve_run_path(run_id)
    payload = _load_or_build_events(run_id=run_id, run_path=run_path, allow_write=True)
    return _filter_events_payload(payload, label=label, min_duration_days=min_duration_days)


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


@router.get("/runs/{run_id}/scorecard", response_model=ScorecardResponseModel)
def run_scorecard(run_id: str) -> dict[str, Any]:
    """Return compact scorecard for a run."""
    run_path = _resolve_run_path(run_id)
    return _build_scorecard_payload(run_id=run_id, run_path=run_path)


@router.get("/runs/{run_id}/events", response_model=EventsResponseModel)
def run_events(
    run_id: str,
    label: Optional[str] = Query(default=None),
    min_duration_days: Optional[int] = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Return regime events artifact for a run."""
    run_path = _resolve_run_path(run_id)
    payload = _load_or_build_events(run_id=run_id, run_path=run_path, allow_write=True)
    return _filter_events_payload(payload, label=label, min_duration_days=min_duration_days)


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


@router.get("/runs/{run_id}/artifacts/{name}")
def download_run_artifact(run_id: str, name: str) -> Response:
    """Download a run artifact as raw bytes with content-type."""
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail=f"Invalid artifact name for run {run_id}: {name}")
    run_path = _resolve_run_path(run_id)
    artifact_path = run_path / name
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} missing artifact: {name}",
        )
    return Response(content=artifact_path.read_bytes(), media_type=_artifact_media_type(name))


@router.get("/runs/{run_id}/bundle.zip")
def download_run_bundle(
    run_id: str,
    artifacts: str | None = Query(default=None),
) -> StreamingResponse:
    """Download selected run artifacts as a zip bundle."""
    run_path = _resolve_run_path(run_id)
    if artifacts is None:
        manifest = _read_json(run_path / "manifest.json", run_id=run_id)
        names_raw = manifest.get("artifacts", [])
        if not isinstance(names_raw, list):
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_id} has malformed artifact: manifest.json",
            )
        artifact_names = [str(name) for name in names_raw]
    else:
        artifact_names = [name.strip() for name in artifacts.split(",") if name.strip()]

    if not artifact_names:
        raise HTTPException(status_code=400, detail=f"Run {run_id} has no artifacts to bundle")
    for name in artifact_names:
        if Path(name).name != name:
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_id} has invalid artifact name in request: {name}",
            )

    missing = [name for name in artifact_names if not (run_path / name).exists()]
    if missing:
        raise _artifact_not_found(run_id, missing)

    total_bytes = 0
    for name in artifact_names:
        path = run_path / name
        if path.is_file():
            total_bytes += path.stat().st_size
    max_bytes = 25 * 1024 * 1024
    if total_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Run {run_id} bundle exceeds 25MB limit ({total_bytes} bytes)",
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in artifact_names:
            path = run_path / name
            if path.is_file():
                zf.writestr(name, path.read_bytes())
    buf.seek(0)

    headers = {
        "Content-Disposition": f'attachment; filename="{run_id}_bundle.zip"',
    }
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@router.get("/runs/{run_id}/tags")
def get_run_tags(run_id: str) -> dict[str, Any]:
    """Return tags metadata for a run."""
    run_path = _resolve_run_path(run_id)
    return _read_json(run_path / "tags.json", run_id=run_id)


@router.put("/runs/{run_id}/tags")
def put_run_tags(run_id: str, request: RunTagsRequest) -> dict[str, Any]:
    """Create or update tags metadata for a run and stamp manifest provenance."""
    run_path = _resolve_run_path(run_id)
    _assert_run_not_frozen(run_id=run_id, run_path=run_path, action="update tags")
    tags = sorted({str(tag).strip() for tag in request.tags if str(tag).strip()})
    payload = {
        "tags": tags,
        "notes": request.notes,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_path / "tags.json", payload)
    _ensure_manifest_has_artifact(run_id=run_id, run_path=run_path, artifact_name="tags.json")
    return payload


@router.post("/runs/{run_id}/freeze")
def freeze_run(run_id: str, request: FreezeRunRequest | None = None) -> dict[str, Any]:
    """Freeze a run to prevent mutating artifacts inside run dir."""
    run_path = _resolve_run_path(run_id)
    reason = request.reason if request is not None else None
    payload = {
        "frozen": True,
        "reason": reason,
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_path / "frozen.json", payload)
    _ensure_manifest_has_artifact(run_id=run_id, run_path=run_path, artifact_name="frozen.json")
    return {"run_id": run_id, **payload}


@router.post("/runs/{run_id}/unfreeze")
def unfreeze_run(run_id: str) -> dict[str, Any]:
    """Unfreeze a run by deleting frozen.json when present."""
    run_path = _resolve_run_path(run_id)
    frozen_path = run_path / "frozen.json"
    if not frozen_path.exists():
        return {"run_id": run_id, "unfrozen": False}
    frozen_path.unlink()
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path, run_id=run_id)
        artifacts = manifest.get("artifacts", [])
        if isinstance(artifacts, list) and "frozen.json" in artifacts:
            manifest["artifacts"] = [name for name in artifacts if name != "frozen.json"]
            write_manifest_with_provenance(run_dir=run_path, manifest_payload=manifest)
    return {"run_id": run_id, "unfrozen": True}


@router.get("/runs/{run_id}/integrity")
def run_integrity(run_id: str) -> dict[str, Any]:
    """Verify manifest artifact hashes against files on disk."""
    run_path = _resolve_run_path(run_id)
    manifest = _read_json(run_path / "manifest.json", run_id=run_id)
    schema_version = manifest.get("schema_version")
    artifacts = manifest.get("artifacts", [])
    expected = manifest.get("artifacts_sha256", {})
    if schema_version is None:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed manifest.json: missing schema_version",
        )
    if not isinstance(artifacts, list) or not isinstance(expected, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed manifest.json",
        )

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for artifact_name in artifacts:
        name = str(artifact_name)
        path = run_path / name
        if name == "manifest.json":
            continue
        if not path.exists() or not path.is_file():
            missing.append(name)
            continue
        expected_hash = expected.get(name)
        actual_hash = _sha256_file(path)
        if expected_hash is None:
            missing.append(name)
            continue
        if str(expected_hash) != actual_hash:
            mismatched.append(
                {
                    "name": name,
                    "expected": str(expected_hash),
                    "actual": actual_hash,
                }
            )

    return {
        "run_id": run_id,
        "ok": len(missing) == 0 and len(mismatched) == 0,
        "missing": missing,
        "mismatched": mismatched,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


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


@router.post("/runs/compare")
def compare_runs(request: CompareRunsRequest) -> dict[str, Any]:
    """Compare run artifacts using persisted summary/evaluation data."""
    if not request.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must contain at least one run id")
    return _compare_runs_payload(request.run_ids)


@router.post("/runs/compare_pinned_latest")
def compare_pinned_latest() -> dict[str, Any]:
    """Convenience compare for [pinned, latest] run ids."""
    pinned = _read_pinned_run_id_or_404()
    latest = _latest_run_id_or_404()
    payload = _compare_runs_payload([pinned, latest])
    payload["pinned_run_id"] = pinned
    payload["latest_run_id"] = latest
    return payload


@router.post("/alerts/transition")
def transition_alerts(request: TransitionAlertRequest) -> dict[str, Any]:
    """Return recent regime transitions filtered by labels and lookback."""
    run_id = _resolve_effective_run_id(run_id=request.run_id, use_pinned=request.use_pinned)
    run_path = _resolve_run_path(run_id)
    events_payload = _load_or_build_events(run_id=run_id, run_path=run_path, allow_write=True)
    events = events_payload.get("events", [])
    if not isinstance(events, list):
        events = []

    as_of = _events_as_of_date(events)
    cutoff = as_of - timedelta(days=int(request.lookback_days))
    transitions: list[dict[str, Any]] = []
    for idx in range(1, len(events)):
        prev = events[idx - 1]
        current = events[idx]
        if not isinstance(prev, dict) or not isinstance(current, dict):
            continue
        from_label = str(prev.get("label", "unknown"))
        to_label = str(current.get("label", "unknown"))
        if request.from_label is not None and from_label != request.from_label:
            continue
        if request.to_label is not None and to_label != request.to_label:
            continue

        transition_date_raw = current.get("start_date")
        try:
            transition_date = datetime.strptime(str(transition_date_raw), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if transition_date < cutoff:
            continue
        transitions.append(
            {
                "from_label": from_label,
                "to_label": to_label,
                "from_state": int(prev.get("state", -1)),
                "to_state": int(current.get("state", -1)),
                "date": str(transition_date_raw),
                "from_segment": int(prev.get("segment_index", idx - 1)),
                "to_segment": int(current.get("segment_index", idx)),
            }
        )

    return {
        "run_id": run_id,
        "lookback_days": int(request.lookback_days),
        "count": int(len(transitions)),
        "last_transition": transitions[-1] if transitions else None,
        "transitions": transitions,
    }


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
  <div id="activeInfo"><strong>Active run:</strong> N/A</div>
  <div id="pinnedInfo"><strong>Pinned run:</strong> none</div>
  <div id="status">Loading runs...</div>
  <div class="row">
    <label for="runSelect"><strong>Run:</strong></label>
    <select id="runSelect"></select>
    <button id="refreshBtn" type="button">Refresh</button>
    <button id="pinBtn" type="button">Pin this run</button>
    <button id="unpinBtn" type="button" style="display:none;">Unpin</button>
  </div>
  <div class="box">
    <div id="currentRegime">Current regime: N/A</div>
    <div id="currentMeta"></div>
  </div>
  <div class="box">
    <strong>Run summary</strong>
    <pre id="summaryPre">N/A</pre>
  </div>
  <div class="box">
    <strong>Compare pinned vs latest</strong>
    <div id="compareStatus">N/A</div>
    <button id="compareBtn" type="button" style="display:none; margin-top:8px;">Compare pinned vs latest</button>
    <pre id="comparePre" style="margin-top:8px; white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;"></pre>
  </div>
  <iframe id="plotFrame" src="about:blank" title="Regime Plot"></iframe>

  <script>
    const runSelect = document.getElementById("runSelect");
    const refreshBtn = document.getElementById("refreshBtn");
    const pinBtn = document.getElementById("pinBtn");
    const unpinBtn = document.getElementById("unpinBtn");
    const statusEl = document.getElementById("status");
    const activeInfoEl = document.getElementById("activeInfo");
    const pinnedInfoEl = document.getElementById("pinnedInfo");
    const currentRegimeEl = document.getElementById("currentRegime");
    const currentMetaEl = document.getElementById("currentMeta");
    const summaryPre = document.getElementById("summaryPre");
    const plotFrame = document.getElementById("plotFrame");
    const compareBtn = document.getElementById("compareBtn");
    const compareStatus = document.getElementById("compareStatus");
    const comparePre = document.getElementById("comparePre");
    let pinnedRunId = null;
    let latestRunId = null;

    async function fetchJson(url, options) {{
      const res = await fetch(url, options || {{}});
      const data = await res.json().catch(() => ({{}}));
      if (!res.ok) {{
        const detail = data && data.detail ? data.detail : ("HTTP " + res.status);
        throw new Error(detail);
      }}
      return data;
    }}

    async function loadActive() {{
      try {{
        const active = await fetchJson("/runs/active");
        activeInfoEl.innerHTML = "<strong>Active run:</strong> " + active.run_id;
      }} catch (err) {{
        activeInfoEl.innerHTML = "<strong>Active run:</strong> N/A";
      }}
    }}

    async function loadPinned() {{
      try {{
        const pinned = await fetchJson("/runs/pinned");
        pinnedRunId = pinned.run_id;
        pinnedInfoEl.innerHTML = "<strong>Pinned run:</strong> " + pinned.run_id;
        unpinBtn.style.display = "inline-block";
      }} catch (err) {{
        pinnedRunId = null;
        pinnedInfoEl.innerHTML = "<strong>Pinned run:</strong> none";
        unpinBtn.style.display = "none";
      }}
      updateCompareControls();
    }}

    function updateCompareControls() {{
      const shouldShow = Boolean(pinnedRunId && latestRunId && pinnedRunId !== latestRunId);
      compareBtn.style.display = shouldShow ? "inline-block" : "none";
      compareStatus.textContent = shouldShow
        ? ("Ready: pinned=" + pinnedRunId + ", latest=" + latestRunId)
        : "N/A";
      if (!shouldShow) {{
        comparePre.textContent = "";
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
        latestRunId = runs.length > 0 ? runs[0] : null;
        await loadActive();
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
        await loadActive();
        await loadPinned();
        statusEl.textContent = "Pinned run " + runId;
      }} catch (err) {{
        statusEl.textContent = "Error pinning run: " + err.message;
      }}
    }});
    unpinBtn.addEventListener("click", async () => {{
      try {{
        const resp = await fetchJson("/runs/unpin", {{ method: "POST" }});
        await loadActive();
        await loadPinned();
        statusEl.textContent = resp.unpinned ? "Unpinned active run" : "No pinned run to remove";
      }} catch (err) {{
        statusEl.textContent = "Error unpinning run: " + err.message;
      }}
    }});
    compareBtn.addEventListener("click", async () => {{
      if (!pinnedRunId || !latestRunId || pinnedRunId === latestRunId) {{
        return;
      }}
      try {{
        const payload = await fetchJson("/runs/compare_pinned_latest", {{
          method: "POST"
        }});
        const diffs = payload.diffs || {{}};
        const runs = Array.isArray(payload.runs) ? payload.runs : [];
        const first = runs[0] || {{}};
        const second = runs[1] || {{}};
        comparePre.textContent =
          "pinned best_val_ll: " + (first.metrics ? first.metrics.best_val_ll : "N/A") + "\\n" +
          "latest best_val_ll: " + (second.metrics ? second.metrics.best_val_ll : "N/A") + "\\n" +
          "delta_best_val_ll: " + (diffs.delta_best_val_ll ?? "N/A") + "\\n" +
          "delta_transition_entropy: " + (diffs.delta_transition_entropy ?? "N/A") + "\\n" +
          "delta_shock_occupancy: " + (diffs.delta_shock_occupancy ?? "N/A");
      }} catch (err) {{
        comparePre.textContent = "Error comparing runs: " + err.message;
      }}
    }});
    refreshBtn.addEventListener("click", refreshRuns);
    refreshRuns();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_body)


@router.post(
    "/predict_current",
    response_model=PredictCurrentResponseModel,
    responses={
        200: {
            "content": {"application/json": {"example": PREDICT_CURRENT_EXAMPLE}},
        }
    },
)
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


@router.get(
    "/forecast_v2",
    responses={
        200: {
            "content": {"application/json": {"example": FORECAST_V2_EXAMPLE}},
        }
    },
)
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


def _list_run_ids(order: str = "desc") -> list[str]:
    order_lc = order.lower()
    if order_lc not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail=f"Invalid order '{order}'. Use 'asc' or 'desc'.")
    if not RUNS_ROOT.exists():
        return []
    run_ids = sorted(
        [p.name for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_")],
        reverse=(order_lc == "desc"),
    )
    return run_ids


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
    pinned_run_id = _read_pinned_run_id_or_none()
    if pinned_run_id is None:
        pinned_path = _pinned_run_file()
        if not pinned_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Pinned run file not found: {PINNED_RUN_FILENAME}",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Pinned run file is empty or invalid: {PINNED_RUN_FILENAME}",
        )
    return pinned_run_id


def _read_pinned_run_id_or_none() -> str | None:
    pinned_path = _pinned_run_file()
    if not pinned_path.exists():
        return None

    pinned_run_id = pinned_path.read_text(encoding="utf-8").strip()
    if not pinned_run_id:
        return None

    try:
        _validate_run_id_for_pin(pinned_run_id)
    except HTTPException:
        return None
    run_path = RUNS_ROOT / pinned_run_id
    if not run_path.exists() or not run_path.is_dir():
        return None
    return pinned_run_id


def _is_run_frozen(run_path: Path) -> bool:
    frozen_path = run_path / "frozen.json"
    if not frozen_path.exists():
        return False
    try:
        payload = _read_json(frozen_path, run_id=run_path.name)
    except HTTPException:
        return False
    return bool(payload.get("frozen", True))


def _assert_run_not_frozen(run_id: str, run_path: Path, action: str) -> None:
    if _is_run_frozen(run_path):
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is frozen; cannot {action}.",
        )


def _ensure_manifest_has_artifact(run_id: str, run_path: Path, artifact_name: str) -> None:
    manifest_path = run_path / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = _read_json(manifest_path, run_id=run_id)
    else:
        manifest = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": [],
        }
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    if artifact_name not in artifacts:
        artifacts.append(artifact_name)
    manifest["artifacts"] = artifacts
    write_manifest_with_provenance(run_dir=run_path, manifest_payload=manifest)


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


def _artifact_media_type(name: str) -> str:
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".html"):
        return "text/html; charset=utf-8"
    if name.endswith(".npz"):
        return "application/octet-stream"
    return "application/octet-stream"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_or_build_events(run_id: str, run_path: Path, allow_write: bool) -> dict[str, Any]:
    events_path = run_path / "events.json"
    needs_write = False
    if events_path.exists():
        existing = _read_json(events_path, run_id=run_id)
        if _events_payload_is_enriched(existing):
            return existing
        needs_write = True
    else:
        needs_write = True

    if needs_write and not allow_write:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} requires events enrichment but writes are disabled",
        )
    if needs_write:
        _assert_run_not_frozen(run_id=run_id, run_path=run_path, action="enrich events artifact")

    missing: list[str] = []
    viterbi_path = run_path / "viterbi_states.json"
    predict_path = run_path / "predict_proba.json"
    if not viterbi_path.exists():
        missing.append(viterbi_path.name)
    if not predict_path.exists():
        missing.append(predict_path.name)
    if missing:
        raise _artifact_not_found(run_id, missing)

    viterbi = _read_json(viterbi_path, run_id=run_id)
    predict = _read_json(predict_path, run_id=run_id)
    labels_map = _read_label_mapping(run_path)

    events_payload = _build_events_payload(
        run_id=run_id,
        dates=viterbi.get("dates", []),
        states=viterbi.get("states", []),
        returns=predict.get("returns", []),
        labels_map=labels_map,
    )
    with events_path.open("w", encoding="utf-8") as f:
        json.dump(events_payload, f, indent=2)

    _ensure_manifest_has_artifact(run_id=run_id, run_path=run_path, artifact_name="events.json")

    return events_payload


def _build_events_payload(
    run_id: str,
    dates: list[Any],
    states: list[Any],
    returns: list[Any],
    labels_map: dict[str, str],
) -> dict[str, Any]:
    if len(dates) == 0 or len(states) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has empty artifact content for events generation",
        )
    if len(dates) != len(states):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed artifact: viterbi_states.json",
        )
    if len(returns) != len(states):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has malformed artifact: predict_proba.json",
        )

    seq_states = [int(s) for s in states]
    seq_returns = np.asarray([float(r) for r in returns], dtype=np.float64)
    segments: list[dict[str, Any]] = []
    start_idx = 0
    current_state = seq_states[0]
    segment_index = 0

    for idx in range(1, len(seq_states)):
        if seq_states[idx] != current_state:
            end_idx = idx - 1
            segment_returns = seq_returns[start_idx : end_idx + 1]
            cumulative_log_return = float(np.sum(segment_returns))
            length = int(end_idx - start_idx + 1)
            segments.append(
                {
                    "segment_index": int(segment_index),
                    "state": int(current_state),
                    "label": labels_map.get(str(current_state), f"regime_{current_state}"),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_date": str(dates[start_idx]),
                    "end_date": str(dates[end_idx]),
                    "length": length,
                    "duration_days": length,
                    "cumulative_log_return": cumulative_log_return,
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
            current_state = seq_states[idx]

    end_idx = len(seq_states) - 1
    segment_returns = seq_returns[start_idx : end_idx + 1]
    cumulative_log_return = float(np.sum(segment_returns))
    length = int(end_idx - start_idx + 1)
    segments.append(
        {
            "segment_index": int(segment_index),
            "state": int(current_state),
            "label": labels_map.get(str(current_state), f"regime_{current_state}"),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_date": str(dates[start_idx]),
            "end_date": str(dates[end_idx]),
            "length": length,
            "duration_days": length,
            "cumulative_log_return": cumulative_log_return,
            "mean_return": float(np.mean(segment_returns)),
            "realized_vol": (
                float(np.std(segment_returns, ddof=1))
                if segment_returns.shape[0] > 1
                else 0.0
            ),
        }
    )

    return {
        "run_id": run_id,
        "n_events": int(len(segments)),
        "events": segments,
    }


def _events_payload_is_enriched(payload: dict[str, Any]) -> bool:
    events = payload.get("events", [])
    if not isinstance(events, list):
        return False
    if not events:
        return True
    required = {
        "segment_index",
        "start_idx",
        "end_idx",
        "duration_days",
        "mean_return",
        "realized_vol",
    }
    first = events[0]
    if not isinstance(first, dict):
        return False
    return required.issubset(set(first.keys()))


def _filter_events_payload(
    payload: dict[str, Any],
    label: str | None,
    min_duration_days: int | None,
) -> dict[str, Any]:
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []

    filtered = events
    if label is not None:
        filtered = [event for event in filtered if str(event.get("label")) == label]
    if min_duration_days is not None:
        filtered = [
            event
            for event in filtered
            if int(event.get("duration_days", event.get("length", 0))) >= min_duration_days
        ]

    out = dict(payload)
    out["events"] = filtered
    out["n_events"] = int(len(filtered))
    return out


def _build_scorecard_payload(run_id: str, run_path: Path) -> dict[str, Any]:
    evaluation = _read_json(run_path / "evaluation.json", run_id=run_id)
    events_payload = _load_or_build_events(run_id=run_id, run_path=run_path, allow_write=True)
    viterbi = _read_json(run_path / "viterbi_states.json", run_id=run_id)

    eval_metrics = evaluation.get("metrics", {})
    best_val_ll = _as_float_or_none(eval_metrics.get("best_val_log_likelihood"))
    epochs_ran = eval_metrics.get("epochs_ran")
    stopped_early = eval_metrics.get("stopped_early")

    if epochs_ran is None or stopped_early is None:
        metrics_path = run_path / "metrics.json"
        if metrics_path.exists():
            metrics_payload = _read_json(metrics_path, run_id=run_id)
            history = metrics_payload.get("history", {})
            if epochs_ran is None:
                epochs_ran = history.get("epochs_ran")
            if stopped_early is None:
                stopped_early = history.get("stopped_early")

    transition_entropy = _as_float_or_none(
        evaluation.get("regime_diagnostics", {}).get("transition_entropy")
    )
    shock_occupancy = _shock_occupancy(evaluation.get("regime_diagnostics", {}))
    if shock_occupancy is None:
        summary_path = run_path / "regime_summary.json"
        if summary_path.exists():
            shock_occupancy = _shock_occupancy(_read_json(summary_path, run_id=run_id))

    events = events_payload.get("events", [])
    events_by_label: dict[str, int] = {}
    if isinstance(events, list):
        for event in events:
            label = str(event.get("label", "unknown"))
            events_by_label[label] = events_by_label.get(label, 0) + 1

    dates = viterbi.get("dates", [])
    states = viterbi.get("states", [])
    labels = viterbi.get("labels", [])
    if not dates or not states:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has empty artifact: viterbi_states.json",
        )
    last_state = int(states[-1])
    if len(labels) == len(states):
        last_label = str(labels[-1])
    else:
        last_label = _read_label_mapping(run_path).get(str(last_state), f"regime_{last_state}")

    return {
        "run_id": run_id,
        "metrics": {
            "best_val_log_likelihood": best_val_ll,
            "epochs_ran": int(epochs_ran) if epochs_ran is not None else None,
            "stopped_early": bool(stopped_early) if stopped_early is not None else None,
        },
        "diagnostics": {
            "transition_entropy": transition_entropy,
            "shock_occupancy": shock_occupancy,
        },
        "events_by_label": events_by_label,
        "last_regime": {
            "state": last_state,
            "label": last_label,
            "date": str(dates[-1]),
        },
    }


def _compare_runs_payload(run_ids: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_path = _resolve_run_path(run_id)
        evaluation = _read_json(run_path / "evaluation.json", run_id=run_id)
        summary = _read_json(run_path / "regime_summary.json", run_id=run_id)
        transition = _read_json(run_path / "transition_matrix.json", run_id=run_id)
        metrics = evaluation.get("metrics", {})
        rows.append(
            {
                "run_id": run_id,
                "metrics": {
                    "best_val_ll": metrics.get("best_val_log_likelihood"),
                    "test_avg_ll": metrics.get("test_avg_log_likelihood"),
                },
                "evaluation": evaluation,
                "summary": summary,
                "transition": transition,
            }
        )

    response: dict[str, Any] = {"runs": rows}
    if len(rows) == 2:
        first = rows[0]
        second = rows[1]
        first_best = _as_float_or_none(first["metrics"].get("best_val_ll"))
        second_best = _as_float_or_none(second["metrics"].get("best_val_ll"))
        first_ent = _as_float_or_none(
            first["evaluation"].get("regime_diagnostics", {}).get("transition_entropy")
        )
        second_ent = _as_float_or_none(
            second["evaluation"].get("regime_diagnostics", {}).get("transition_entropy")
        )
        first_shock = _shock_occupancy(first["summary"])
        second_shock = _shock_occupancy(second["summary"])
        response["diffs"] = {
            "delta_best_val_ll": (
                None if first_best is None or second_best is None else float(second_best - first_best)
            ),
            "delta_transition_entropy": (
                None if first_ent is None or second_ent is None else float(second_ent - first_ent)
            ),
            "delta_shock_occupancy": (
                None if first_shock is None or second_shock is None else float(second_shock - first_shock)
            ),
        }
    return response


def _events_as_of_date(events: list[dict[str, Any]]) -> datetime:
    max_date: datetime | None = None
    for event in events:
        end_date_raw = event.get("end_date")
        try:
            end_dt = datetime.strptime(str(end_date_raw), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if max_date is None or end_dt > max_date:
            max_date = end_dt
    return max_date or datetime.now(timezone.utc)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shock_occupancy(summary: dict[str, Any]) -> float | None:
    regimes = summary.get("regimes", [])
    if not isinstance(regimes, list):
        return None
    for row in regimes:
        if str(row.get("label")) == "shock":
            return _as_float_or_none(row.get("avg_posterior_probability"))
    return None
