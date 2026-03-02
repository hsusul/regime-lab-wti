"""Manual live-mode helper: fetch data, optionally retrain, regenerate plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from energy_data.eia_client import EIAClientConfig, EIAWTIClient
from models.train import load_config, train_model_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live mode: refresh run if data changed.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Pipeline config path",
    )
    return parser.parse_args()


def _read_latest_run_id(runs_root: Path) -> str | None:
    latest_path = runs_root / "latest_run.txt"
    if not latest_path.exists():
        return None
    latest = latest_path.read_text(encoding="utf-8").strip()
    return latest or None


def _read_pinned_run_id(runs_root: Path) -> str | None:
    pinned_path = runs_root / "pinned_run.txt"
    if not pinned_path.exists():
        return None
    pinned = pinned_path.read_text(encoding="utf-8").strip()
    if not pinned:
        return None
    run_dir = runs_root / pinned
    return pinned if run_dir.exists() else None


def _latest_data_date(prices: pd.DataFrame) -> str:
    dates = pd.to_datetime(prices["date"], errors="coerce")
    if dates.isna().all():
        raise ValueError("Fetched price data has no valid dates.")
    return str(dates.max().date())


def _run_end_date(runs_root: Path, run_id: str) -> str | None:
    summary_path = runs_root / run_id / "regime_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    end_date = summary.get("end_date")
    return str(end_date) if end_date else None


def _last_regime_label(runs_root: Path, run_id: str) -> str | None:
    viterbi_path = runs_root / run_id / "viterbi_states.json"
    if not viterbi_path.exists():
        return None
    with viterbi_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    states = payload.get("states", [])
    labels = payload.get("labels", [])
    mapping = payload.get("label_mapping", {})
    if not states:
        return None
    last_state = int(states[-1])
    if len(labels) == len(states):
        return str(labels[-1])
    return str(mapping.get(str(last_state), f"regime_{last_state}"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runs_root = Path(cfg.run_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    client = EIAWTIClient(
        EIAClientConfig(
            cache_dir=Path(cfg.data["raw_dir"]),
            cache_filename=str(cfg.data.get("cache_filename", "wti_cushing_daily.csv")),
        )
    )
    prices = client.fetch_daily_wti(
        force_refresh=False,
        start_date=cfg.data.get("start_date"),
        end_date=cfg.data.get("end_date"),
    )
    latest_data_date = _latest_data_date(prices)

    latest_run_id = _read_latest_run_id(runs_root)
    retrain = latest_run_id is None
    if latest_run_id is not None:
        run_end = _run_end_date(runs_root, latest_run_id)
        retrain = run_end is None or run_end < latest_data_date

    if retrain:
        result = train_model_run(config_path=args.config, force_refresh=False)
        run_id = str(result["run_id"])
        subprocess.run(
            [sys.executable, "-m", "scripts.make_plots", "--run-id", run_id],
            check=True,
        )
    else:
        run_id = str(latest_run_id)

    pinned_run_id = _read_pinned_run_id(runs_root)
    active_run_id = pinned_run_id or run_id
    last_regime = _last_regime_label(runs_root, active_run_id)

    print(f"run_id: {run_id}")
    print(f"active_run: {active_run_id}")
    print(f"pinned_run: {pinned_run_id}")
    print(f"last_regime: {last_regime}")


if __name__ == "__main__":
    main()

