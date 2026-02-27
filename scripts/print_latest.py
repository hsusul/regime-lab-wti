"""Print useful pointers for the latest model run."""

from __future__ import annotations

import json
from pathlib import Path


def _latest_run_id(runs_root: Path) -> str:
    latest_file = runs_root / "latest_run.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id

    run_ids = sorted(
        [p.name for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        reverse=True,
    )
    if not run_ids:
        raise FileNotFoundError("No runs found under runs/")
    return run_ids[0]


def main() -> None:
    runs_root = Path("runs")
    run_id = _latest_run_id(runs_root)
    run_dir = runs_root / run_id

    viterbi_path = run_dir / "viterbi_states.json"
    if not viterbi_path.exists():
        raise FileNotFoundError(f"Missing artifact: {viterbi_path}")

    with viterbi_path.open("r", encoding="utf-8") as f:
        viterbi = json.load(f)

    dates = viterbi.get("dates", [])
    labels = viterbi.get("labels", [])
    states = viterbi.get("states", [])
    if not dates or not states:
        raise ValueError(f"viterbi_states.json has no states for run {run_id}")

    last_date = str(dates[-1])
    last_state = int(states[-1])
    if len(labels) == len(states):
        last_label = str(labels[-1])
    else:
        mapping = viterbi.get("label_mapping", {})
        last_label = str(mapping.get(str(last_state), f"regime_{last_state}"))

    plot_path = run_dir / "regimes.html"

    print(f"run_id: {run_id}")
    print(f"plot_path: {plot_path}")
    print(f"last_regime: {last_label} ({last_state})")
    print(f"last_date: {last_date}")


if __name__ == "__main__":
    main()
