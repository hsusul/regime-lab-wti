from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import app.routes as routes
from app.main import app


def test_runs_endpoint_lists_newest_first(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "run_test").mkdir()
    (runs_root / "run_20240101T000000Z_aaaa1111").mkdir()
    (runs_root / "run_20250101T000000Z_bbbb2222").mkdir()
    (runs_root / "misc_folder").mkdir()

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)

    client = TestClient(app)
    response = client.get("/runs")

    assert response.status_code == 200
    payload = response.json()
    assert "runs" in payload
    assert payload["runs"] == [
        "run_test",
        "run_20250101T000000Z_bbbb2222",
        "run_20240101T000000Z_aaaa1111",
    ]
