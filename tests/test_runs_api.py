from __future__ import annotations

import json
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


def test_latest_summary_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    latest_run = runs_root / "run_20250101T000000Z_bbbb2222"
    latest_run.mkdir()
    (runs_root / "run_20240101T000000Z_aaaa1111").mkdir()
    (latest_run / "regime_summary.json").write_text(
        json.dumps({"n_observations": 123}), encoding="utf-8"
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)
    response = client.get("/runs/latest/summary")

    assert response.status_code == 200
    assert response.json()["n_observations"] == 123


def test_latest_run_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    older = runs_root / "run_20240101T000000Z_aaaa1111"
    older.mkdir()

    latest = runs_root / "run_20250101T000000Z_bbbb2222"
    latest.mkdir()
    (latest / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": latest.name,
                "created_at_utc": "2026-02-28T00:00:00+00:00",
                "end_date": "2026-02-27",
            }
        ),
        encoding="utf-8",
    )
    (latest / "plot_meta.json").write_text(
        json.dumps(
            {
                "last_label": "mid_vol",
                "last_date": "2026-02-27",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get("/runs/latest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == latest.name
    assert payload["run_dir"].endswith(latest.name)
    assert payload["created_at_utc"] == "2026-02-28T00:00:00+00:00"
    assert payload["end_date"] == "2026-02-27"
    assert payload["last_label"] == "mid_vol"
    assert payload["last_date"] == "2026-02-27"


def test_run_model_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260110T000000Z_model"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "n_states": 3,
        "initial_probs": [0.4, 0.3, 0.3],
        "initial_logits": [0.0, 0.0, 0.0],
        "transition_matrix": [[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        "mu": [0.001, 0.0, -0.01],
        "sigma": [0.01, 0.02, 0.05],
    }
    (run_dir / "model_params.json").write_text(json.dumps(model_payload), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/model")
    assert response.status_code == 200
    assert response.json() == model_payload


def test_predict_current_prefers_viterbi_and_falls_back(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_viterbi = runs_root / "run_20260101T000000Z_viterbi111"
    run_viterbi.mkdir()
    (run_viterbi / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_viterbi / "viterbi_states.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-01", "2026-01-02"],
                "states": [0, 2],
                "labels": ["low_vol", "shock"],
                "label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"},
            }
        ),
        encoding="utf-8",
    )

    run_filter = runs_root / "run_20260102T000000Z_filter222"
    run_filter.mkdir()
    (run_filter / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_filter / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-03", "2026-01-06"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    viterbi_resp = client.post("/predict_current", json={"run_id": run_viterbi.name})
    assert viterbi_resp.status_code == 200
    assert viterbi_resp.json()["source"] == "viterbi"
    assert viterbi_resp.json()["label"] == "shock"
    assert viterbi_resp.json()["p_label"] is None
    assert viterbi_resp.json()["run_id"] == run_viterbi.name
    assert viterbi_resp.json()["as_of"] == "2026-01-02"
    assert viterbi_resp.json()["label_mapping"]["2"] == "shock"

    filter_resp = client.post("/predict_current", json={"run_id": run_filter.name})
    assert filter_resp.status_code == 200
    assert filter_resp.json()["source"] == "filtering_argmax"
    assert filter_resp.json()["state"] == 1
    assert filter_resp.json()["label"] == "mid_vol"
    assert isinstance(filter_resp.json()["p_label"], float)
    assert filter_resp.json()["run_id"] == run_filter.name
    assert filter_resp.json()["as_of"] == "2026-01-06"
    assert filter_resp.json()["label_mapping"]["1"] == "mid_vol"


def test_ui_page_loads(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)

    client = TestClient(app)
    response = client.get("/ui")

    assert response.status_code == 200
    assert "WTI Regime Monitor" in response.text
    assert "Current regime" in response.text


def test_run_artifacts_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260110T000000Z_artifacts"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "a.json").write_text("{}", encoding="utf-8")
    (run_dir / "b.txt").write_text("x", encoding="utf-8")
    (run_dir / "subdir").mkdir()

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/artifacts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["artifacts"] == ["a.json", "b.txt"]


def test_predict_current_include_probs(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20260110T000000Z_probs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-10", "2026-01-13"],
                "states": [2, 1],
                "labels": ["shock", "mid_vol"],
                "label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-10", "2026-01-13"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model_params.json").write_text(
        json.dumps(
            {
                "n_states": 3,
                "initial_probs": [0.3, 0.4, 0.3],
                "initial_logits": [0.0, 0.0, 0.0],
                "transition_matrix": [
                    [0.9, 0.08, 0.02],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.2, 0.6],
                ],
                "mu": [0.001, 0.0, -0.02],
                "sigma": [0.01, 0.015, 0.04],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.post("/predict_current?include_probs=true", json={"run_id": run_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "viterbi"
    assert payload["label"] == "mid_vol"
    assert isinstance(payload["probs"], dict)
    assert payload["probs"]["low_vol"] == 0.1
    assert payload["probs"]["mid_vol"] == 0.7
    assert payload["probs"]["shock"] == 0.2
    assert payload["raw_state_probs"] == [0.1, 0.7, 0.2]
    assert payload["run_id"] == run_id
    assert payload["as_of"] == "2026-01-13"
    assert payload["label_mapping"]["0"] == "low_vol"
    assert isinstance(payload["expected_return"], float)
    assert isinstance(payload["expected_vol"], float)
