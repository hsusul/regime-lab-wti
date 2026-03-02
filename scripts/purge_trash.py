"""Purge old run trash entries under runs/_trash."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _extract_run_id_from_trash_id(trash_id: str) -> str | None:
    if "_" not in trash_id:
        return None
    run_id = trash_id.rsplit("_", 1)[0]
    return run_id if run_id.startswith("run_") else None


def _deleted_at_from_trash_id(trash_id: str) -> datetime | None:
    run_id = _extract_run_id_from_trash_id(trash_id)
    if run_id is None:
        return None
    suffix = trash_id[len(run_id) + 1 :]
    try:
        return datetime.strptime(suffix, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _read_pinned_run_id(runs_root: Path) -> str | None:
    pinned_path = runs_root / "pinned_run.txt"
    if not pinned_path.exists():
        return None
    value = pinned_path.read_text(encoding="utf-8").strip()
    return value or None


def purge_trash(*, runs_root: Path, days: int) -> dict[str, int]:
    trash_root = runs_root / "_trash"
    if not trash_root.exists() or not trash_root.is_dir():
        return {"purged": 0, "skipped_pinned": 0, "skipped_invalid": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    pinned_run_id = _read_pinned_run_id(runs_root)
    purged = 0
    skipped_pinned = 0
    skipped_invalid = 0

    for path in sorted(trash_root.iterdir()):
        if not path.is_dir():
            continue
        trash_id = path.name
        run_id = _extract_run_id_from_trash_id(trash_id)
        deleted_at = _deleted_at_from_trash_id(trash_id)
        if run_id is None or deleted_at is None:
            skipped_invalid += 1
            continue
        if pinned_run_id is not None and run_id == pinned_run_id:
            skipped_pinned += 1
            continue
        if deleted_at <= cutoff:
            shutil.rmtree(path)
            purged += 1

    return {
        "purged": purged,
        "skipped_pinned": skipped_pinned,
        "skipped_invalid": skipped_invalid,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purge old run trash entries")
    parser.add_argument("--days", type=int, default=14, help="Delete trash older than this many days")
    parser.add_argument("--runs-root", default="runs", help="Runs root directory")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    result = purge_trash(runs_root=Path(args.runs_root), days=int(args.days))
    print(
        f"Purged={result['purged']} "
        f"skipped_pinned={result['skipped_pinned']} "
        f"skipped_invalid={result['skipped_invalid']}"
    )


if __name__ == "__main__":
    main()
