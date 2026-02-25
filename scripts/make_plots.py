"""Generate Plotly regime visualization artifacts for a trained run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    dates: Sequence[str], returns: np.ndarray, regime_probabilities: np.ndarray
) -> go.Figure:
    x = pd.to_datetime(dates)
    if x.empty:
        raise ValueError("No observations available for plotting.")

    if regime_probabilities.ndim != 2:
        raise ValueError("regime_probabilities must have shape [T, K].")
    if regime_probabilities.shape[0] != len(x):
        raise ValueError("dates and regime_probabilities length mismatch.")

    cumulative_index = 100.0 * np.exp(np.cumsum(returns))
    states = np.argmax(regime_probabilities, axis=1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Return Index with Regime Shading", "Regime Probabilities"),
    )

    segments = contiguous_segments(states)
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
        fig.add_trace(
            go.Scatter(
                x=x,
                y=regime_probabilities[:, k],
                mode="lines",
                name=f"Regime {k} Prob",
                line={"color": LINE_COLORS[k % len(LINE_COLORS)], "width": 2},
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Index (start ~100)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0.0, 1.0], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0.01},
        margin={"l": 60, "r": 40, "t": 90, "b": 50},
    )
    return fig


def main() -> None:
    args = parse_args()
    runs_root = Path("runs")
    run_id = resolve_run_id(runs_root=runs_root, run_id=args.run_id)
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    payload = load_predict_payload(run_dir)
    dates = payload["dates"]
    returns = np.asarray(payload["returns"], dtype=np.float64)
    regime_probabilities = np.asarray(payload["regime_probabilities"], dtype=np.float64)

    if returns.ndim != 1:
        raise ValueError("returns must be a 1D sequence.")
    if returns.shape[0] != len(dates):
        raise ValueError("dates and returns length mismatch.")

    fig = build_figure(dates=dates, returns=returns, regime_probabilities=regime_probabilities)

    output_path = Path(args.out) if args.out else run_dir / "regimes.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    print(output_path)


if __name__ == "__main__":
    main()
