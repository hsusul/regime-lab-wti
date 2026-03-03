"""Rolling retrain runner based on latest run staleness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling retrain runner for regime-lab-wti")
    parser.add_argument("--stale-days", type=int, default=3, help="Retrain if latest as_of is at least this many days old.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Pipeline config path.")
    parser.add_argument("--force-refresh", action="store_true", help="Force data refresh during training.")
    return parser.parse_args()


def is_stale(latest_as_of: date | None, *, today: date, stale_days: int) -> bool:
    """Return True when a retrain should run."""
    if latest_as_of is None:
        return True
    delta_days = (today - latest_as_of).days
    return delta_days >= int(stale_days)


def _read_pinned_run_id(runs_root: Path) -> str | None:
    pinned_path = runs_root / "pinned_run.txt"
    if not pinned_path.exists():
        return None
    run_id = pinned_path.read_text(encoding="utf-8").strip()
    return run_id or None


def _latest_run_id(runs_root: Path) -> str | None:
    latest_path = runs_root / "latest_run.txt"
    if latest_path.exists():
        run_id = latest_path.read_text(encoding="utf-8").strip()
        if run_id and (runs_root / run_id).is_dir():
            return run_id

    run_ids = sorted(
        [p.name for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        reverse=True,
    ) if runs_root.exists() else []
    return run_ids[0] if run_ids else None


def _load_run_as_of_date(run_dir: Path) -> date | None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        end_date = payload.get("end_date")
        if end_date:
            try:
                return datetime.strptime(str(end_date), "%Y-%m-%d").date()
            except ValueError:
                pass

    viterbi_path = run_dir / "viterbi_states.json"
    if viterbi_path.exists():
        try:
            payload = json.loads(viterbi_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        dates = payload.get("dates", [])
        if isinstance(dates, list) and dates:
            try:
                return datetime.strptime(str(dates[-1]), "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def _run_plots(run_id: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.make_plots", "--run-id", str(run_id)],
        check=True,
    )


def main() -> None:
    from models.train import load_config, train_model_run

    args = parse_args()
    cfg = load_config(args.config)
    runs_root = Path(cfg.run_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    latest_run_id = _latest_run_id(runs_root)
    latest_as_of: date | None = None
    if latest_run_id is not None:
        latest_as_of = _load_run_as_of_date(runs_root / latest_run_id)

    if not is_stale(
        latest_as_of,
        today=datetime.now(timezone.utc).date(),
        stale_days=int(args.stale_days),
    ):
        print("fresh")
        return

    pinned_before = _read_pinned_run_id(runs_root)
    result = train_model_run(
        config_path=args.config,
        force_refresh=bool(args.force_refresh),
        run_id=None,
    )
    run_id = str(result["run_id"])
    _run_plots(run_id=run_id)

    # Preserve the pinned pointer as the stable reference, even after a retrain.
    pinned_after = _read_pinned_run_id(runs_root)
    if pinned_before is not None and pinned_after != pinned_before:
        (runs_root / "pinned_run.txt").write_text(pinned_before, encoding="utf-8")

    print(f"trained {run_id}")


if __name__ == "__main__":
    main()
