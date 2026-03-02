from __future__ import annotations

import hashlib
from pathlib import Path

from models.provenance import compute_artifacts_sha256, stamp_manifest_provenance


def test_manifest_sha256_helper(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload_path = run_dir / "small.json"
    payload_bytes = b'{"ok": true}\n'
    payload_path.write_bytes(payload_bytes)

    manifest = {
        "run_id": "run_test",
        "artifacts": ["small.json", "missing.json", "manifest.json"],
    }

    stamped = stamp_manifest_provenance(run_dir=run_dir, manifest_payload=manifest, schema_version=2)
    hashes = stamped["artifacts_sha256"]
    expected_hash = hashlib.sha256(payload_bytes).hexdigest()

    assert stamped["schema_version"] == 2
    assert "small.json" in hashes
    assert hashes["small.json"] == expected_hash
    assert "missing.json" not in hashes
    assert "manifest.json" not in hashes

    direct_hashes = compute_artifacts_sha256(run_dir=run_dir, artifact_filenames=manifest["artifacts"])
    assert direct_hashes["small.json"] == expected_hash
