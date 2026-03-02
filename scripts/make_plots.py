"""Generate Plotly regime visualization artifacts for a trained run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.provenance import write_manifest_with_provenance

REGIME_COLORS = [
    "rgba(31, 119, 180, 0.18)",
    "rgba(255, 127, 14, 0.18)",
    "rgba(44, 160, 44, 0.18)",
]
LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly regime charts for a model run")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id under runs/. Uses runs/latest_run.txt when omitted.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML path. Defaults to runs/<run_id>/regimes.html.",
    )
    return parser.parse_args()


def resolve_run_id(runs_root: Path, run_id: str | None) -> str:
    if run_id:
        return run_id

    latest_path = runs_root / "latest_run.txt"
    if not latest_path.exists():
        raise FileNotFoundError("No run id provided and runs/latest_run.txt was not found.")

    latest = latest_path.read_text(encoding="utf-8").strip()
    if not latest:
        raise ValueError("runs/latest_run.txt is empty.")
    return latest


def ensure_run_artifacts(run_dir: Path, run_id: str, filenames: Sequence[str]) -> None:
    missing = [name for name in filenames if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Run {run_id} is missing required artifact(s): {', '.join(missing)}"
        )


def load_predict_payload(run_dir: Path) -> dict:
    payload_path = run_dir / "predict_proba.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Artifact not found: {payload_path}")

    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required = {"dates", "returns", "regime_probabilities"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"predict_proba.json missing fields: {sorted(missing)}")
    return payload


def load_labels_mapping(run_dir: Path, n_states: int) -> dict[str, str]:
    """Load stable regime labels from artifact with sensible fallback."""
    label_path = run_dir / "regime_labels.json"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            mapping_raw: Any
            if "label_mapping" in payload and isinstance(payload["label_mapping"], dict):
                mapping_raw = payload["label_mapping"]
            else:
                mapping_raw = payload

            mapping: dict[str, str] = {}
            for i in range(n_states):
                mapping[str(i)] = str(mapping_raw.get(str(i), f"regime_{i}"))
            return mapping

    return {str(i): f"regime_{i}" for i in range(n_states)}


def load_viterbi_states(run_dir: Path, n_obs: int) -> np.ndarray | None:
    """Load Viterbi states if available; return None when artifact is absent."""
    viterbi_path = run_dir / "viterbi_states.json"
    if not viterbi_path.exists():
        return None

    with viterbi_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    states = np.asarray(payload.get("states", []), dtype=np.int64)
    if states.ndim != 1:
        raise ValueError("viterbi_states.json field 'states' must be 1D.")
    if states.shape[0] != n_obs:
        raise ValueError("viterbi_states.json length mismatch with predict_proba observations.")
    return states


def load_regime_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "regime_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def contiguous_segments(states: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous state spans as (start_idx, end_idx, state)."""
    if states.size == 0:
        return []

    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(states[0])

    for idx in range(1, states.size):
        state = int(states[idx])
        if state != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = state

    segments.append((start, states.size - 1, current))
    return segments


def build_figure(
    run_id: str,
    date_range: str,
    as_of_date: str,
    dates: Sequence[str],
    returns: np.ndarray,
    regime_probabilities: np.ndarray,
    labels_mapping: dict[str, str],
    occupancies_by_label: dict[str, float],
    viterbi_states: np.ndarray | None = None,
) -> go.Figure:
    x = pd.to_datetime(dates)
    if x.empty:
        raise ValueError("No observations available for plotting.")

    if regime_probabilities.ndim != 2:
        raise ValueError("regime_probabilities must have shape [T, K].")
    if regime_probabilities.shape[0] != len(x):
        raise ValueError("dates and regime_probabilities length mismatch.")

    cumulative_index = 100.0 * np.exp(np.cumsum(returns))
    filter_states = np.argmax(regime_probabilities, axis=1)
    shading_states = viterbi_states if viterbi_states is not None else filter_states
    regime_source = "Viterbi" if viterbi_states is not None else "Filtering argmax"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.54, 0.33, 0.13],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{run_id} | {date_range} | as_of {as_of_date} | Cumulative Return Index with {regime_source} Regime Shading",
            "Filtering Regime Probabilities",
            "Regime Occupancy",
        ),
    )

    segments = contiguous_segments(shading_states)
    for start_idx, end_idx, state in segments:
        x0 = x[start_idx]
        if end_idx + 1 < len(x):
            x1 = x[end_idx + 1]
        else:
            x1 = x[end_idx] + pd.Timedelta(days=1)
        color = REGIME_COLORS[state % len(REGIME_COLORS)]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            row=1,
            col=1,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative_index,
            mode="lines",
            name="Cumulative Return Index",
            line={"color": "#111111", "width": 2},
        ),
        row=1,
        col=1,
    )

    n_states = regime_probabilities.shape[1]
    for k in range(n_states):
        label = labels_mapping.get(str(k), f"regime_{k}")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=regime_probabilities[:, k],
                mode="lines",
                name=f"{label} prob",
                line={"color": LINE_COLORS[k % len(LINE_COLORS)], "width": 2},
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Index (start ~100)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0.0, 1.0], row=2, col=1)
    if occupancies_by_label:
        labels = list(occupancies_by_label.keys())
        values = [float(occupancies_by_label[label]) for label in labels]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker={"color": [LINE_COLORS[i % len(LINE_COLORS)] for i in range(len(labels))]},
                name="Occupancy",
            ),
            row=3,
            col=1,
        )
    fig.update_yaxes(title_text="Share", range=[0.0, 1.0], row=3, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0.01},
        margin={"l": 60, "r": 40, "t": 90, "b": 50},
    )
    return fig


def build_plot_meta(
    run_id: str,
    summary: dict[str, Any],
    manifest: dict[str, Any] | None,
    labels_mapping: dict[str, str],
    occupancies_by_label: dict[str, float],
    last_state: int,
    last_date: str,
    n_obs: int,
) -> dict[str, Any]:
    labels = [labels_mapping.get(str(i), f"regime_{i}") for i in sorted(map(int, labels_mapping))]
    created_at = (
        str(manifest.get("created_at_utc"))
        if isinstance(manifest, dict) and manifest.get("created_at_utc")
        else datetime.now(timezone.utc).isoformat()
    )
    return {
        "run_id": run_id,
        "created_at_utc": created_at,
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "n_observations": int(summary.get("n_observations", n_obs)),
        "labels": labels,
        "occupancies_by_label": occupancies_by_label,
        "last_label": labels_mapping.get(str(last_state), f"regime_{last_state}"),
        "last_date": last_date,
    }


def _occupancies_by_label(summary: dict[str, Any], labels_mapping: dict[str, str]) -> dict[str, float]:
    occupancies: dict[str, float] = {}
    regimes = summary.get("regimes", [])
    if not isinstance(regimes, list):
        regimes = []
    for regime in regimes:
        if not isinstance(regime, dict):
            continue
        regime_idx = int(regime.get("regime", -1))
        if regime_idx < 0:
            continue
        label = labels_mapping.get(str(regime_idx), f"regime_{regime_idx}")
        occ = regime.get("avg_posterior_probability")
        if occ is None:
            continue
        occupancies[label] = float(occ)
    return occupancies


def main() -> None:
    args = parse_args()
    runs_root = Path("runs")
    run_id = resolve_run_id(runs_root=runs_root, run_id=args.run_id)
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ensure_run_artifacts(
        run_dir=run_dir,
        run_id=run_id,
        filenames=[
            "predict_proba.json",
            "viterbi_states.json",
            "regime_labels.json",
            "regime_summary.json",
        ],
    )

    payload = load_predict_payload(run_dir)
    summary = load_regime_summary(run_dir)
    manifest = load_manifest(run_dir)
    dates = payload["dates"]
    returns = np.asarray(payload["returns"], dtype=np.float64)
    regime_probabilities = np.asarray(payload["regime_probabilities"], dtype=np.float64)

    if returns.ndim != 1:
        raise ValueError("returns must be a 1D sequence.")
    if returns.shape[0] != len(dates):
        raise ValueError("dates and returns length mismatch.")

    labels_mapping = load_labels_mapping(run_dir=run_dir, n_states=regime_probabilities.shape[1])
    occupancies_by_label = _occupancies_by_label(summary=summary, labels_mapping=labels_mapping)
    viterbi_states = load_viterbi_states(run_dir=run_dir, n_obs=returns.shape[0])
    start_date = summary.get("start_date") if isinstance(summary, dict) else None
    end_date = summary.get("end_date") if isinstance(summary, dict) else None
    if start_date and end_date:
        date_range = f"{start_date} to {end_date}"
    else:
        date_range = f"{dates[0]} to {dates[-1]}"
    if viterbi_states is not None:
        last_state = int(viterbi_states[-1])
    else:
        last_state = int(np.argmax(regime_probabilities[-1]))
    as_of_date = str(dates[-1])

    fig = build_figure(
        run_id=run_id,
        date_range=date_range,
        as_of_date=as_of_date,
        dates=dates,
        returns=returns,
        regime_probabilities=regime_probabilities,
        labels_mapping=labels_mapping,
        occupancies_by_label=occupancies_by_label,
        viterbi_states=viterbi_states,
    )

    output_path = Path(args.out) if args.out else run_dir / "regimes.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    plot_meta = build_plot_meta(
        run_id=run_id,
        summary=summary,
        manifest=manifest,
        labels_mapping=labels_mapping,
        occupancies_by_label=occupancies_by_label,
        last_state=last_state,
        last_date=as_of_date,
        n_obs=int(returns.shape[0]),
    )
    with (run_dir / "plot_meta.json").open("w", encoding="utf-8") as f:
        json.dump(plot_meta, f, indent=2)

    if isinstance(manifest, dict):
        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, list):
            artifacts = []
            manifest["artifacts"] = artifacts
        for name in ("regimes.html", "plot_meta.json"):
            if name not in artifacts:
                artifacts.append(name)
        write_manifest_with_provenance(run_dir=run_dir, manifest_payload=manifest)

    print(output_path)


if __name__ == "__main__":
    main()
