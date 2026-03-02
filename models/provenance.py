"""Helpers for manifest provenance and artifact hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_artifacts_sha256(run_dir: Path, artifact_filenames: Iterable[str]) -> dict[str, str]:
    """Compute hashes for files listed in manifest artifacts.

    Notes:
    - `manifest.json` is excluded to avoid self-referential hashing.
    - Missing artifact filenames are skipped.
    """
    out: dict[str, str] = {}
    for name in artifact_filenames:
        if name == "manifest.json":
            continue
        path = run_dir / name
        if path.exists() and path.is_file():
            out[name] = sha256_file(path)
    return out


def stamp_manifest_provenance(
    run_dir: Path,
    manifest_payload: dict[str, Any],
    schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Attach schema_version + artifacts_sha256 to manifest payload."""
    artifacts = manifest_payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
        manifest_payload["artifacts"] = artifacts

    manifest_payload["schema_version"] = int(schema_version)
    manifest_payload["artifacts_sha256"] = compute_artifacts_sha256(run_dir, artifacts)
    return manifest_payload


def write_manifest_with_provenance(
    run_dir: Path,
    manifest_payload: dict[str, Any],
    schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Stamp manifest provenance and write to run_dir/manifest.json."""
    stamped = stamp_manifest_provenance(
        run_dir=run_dir,
        manifest_payload=manifest_payload,
        schema_version=schema_version,
    )
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(stamped, f, indent=2, sort_keys=False)
    return stamped
