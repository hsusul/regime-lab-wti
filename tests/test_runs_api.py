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


def test_pinned_run_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20250101T000000Z_pinned111"
    (runs_root / run_id).mkdir()

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    missing_resp = client.get("/runs/pinned")
    assert missing_resp.status_code == 404
    assert "pinned_run.txt" in missing_resp.json()["detail"]

    pin_resp = client.post(f"/runs/{run_id}/pin")
    assert pin_resp.status_code == 200
    assert pin_resp.json()["pinned_run_id"] == run_id

    pinned_resp = client.get("/runs/pinned")
    assert pinned_resp.status_code == 200
    assert pinned_resp.json()["run_id"] == run_id
    assert pinned_resp.json()["run_dir"].endswith(run_id)

    invalid_resp = client.post("/runs/not_a_run_id/pin")
    assert invalid_resp.status_code == 404


def test_active_and_unpin_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    latest_run = runs_root / "run_20260101T000000Z_latest"
    latest_run.mkdir()
    pinned_run = runs_root / "run_20260102T000000Z_pinned"
    pinned_run.mkdir()

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    active_latest = client.get("/runs/active")
    assert active_latest.status_code == 200
    assert active_latest.json()["run_id"] == pinned_run.name

    pin_resp = client.post(f"/runs/{latest_run.name}/pin")
    assert pin_resp.status_code == 200
    active_pinned = client.get("/runs/active")
    assert active_pinned.status_code == 200
    assert active_pinned.json()["run_id"] == latest_run.name

    unpin_true = client.post("/runs/unpin")
    assert unpin_true.status_code == 200
    assert unpin_true.json()["unpinned"] is True

    unpin_false = client.post("/runs/unpin")
    assert unpin_false.status_code == 200
    assert unpin_false.json()["unpinned"] is False

    active_after_unpin = client.get("/runs/active")
    assert active_after_unpin.status_code == 200
    assert active_after_unpin.json()["run_id"] == pinned_run.name


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


def test_evaluation_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20260110T000000Z_eval"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at_utc": "2026-03-01T00:00:00+00:00",
        "n_obs": 100,
        "n_states": 3,
        "metrics": {
            "best_val_log_likelihood": 1.23,
            "test_avg_log_likelihood": 0.45,
        },
        "regime_diagnostics": {
            "implied_avg_duration_days": [10.0, 5.0, 2.0],
            "transition_entropy": 0.9,
            "regimes": [],
            "viterbi_segment_stats": {
                "segment_count": 5,
                "mean_duration": 20.0,
                "max_duration": 31,
            },
        },
    }
    (run_dir / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/evaluation")
    assert by_id.status_code == 200
    assert by_id.json()["run_id"] == run_id
    assert "regime_diagnostics" in by_id.json()

    latest = client.get("/runs/latest/evaluation")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id


def test_runs_compare_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_a = runs_root / "run_20260101T000000Z_a"
    run_b = runs_root / "run_20260102T000000Z_b"
    run_a.mkdir()
    run_b.mkdir()

    eval_a = {
        "metrics": {"best_val_log_likelihood": 10.0, "test_avg_log_likelihood": 1.0},
        "regime_diagnostics": {"transition_entropy": 0.5},
    }
    eval_b = {
        "metrics": {"best_val_log_likelihood": 12.5, "test_avg_log_likelihood": 1.2},
        "regime_diagnostics": {"transition_entropy": 0.9},
    }
    summary_a = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.7},
            {"label": "mid_vol", "avg_posterior_probability": 0.2},
            {"label": "shock", "avg_posterior_probability": 0.1},
        ]
    }
    summary_b = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.5},
            {"label": "mid_vol", "avg_posterior_probability": 0.25},
            {"label": "shock", "avg_posterior_probability": 0.25},
        ]
    }
    transition_stub = {"transition_matrix": [[0.9, 0.1], [0.2, 0.8]]}

    (run_a / "evaluation.json").write_text(json.dumps(eval_a), encoding="utf-8")
    (run_b / "evaluation.json").write_text(json.dumps(eval_b), encoding="utf-8")
    (run_a / "regime_summary.json").write_text(json.dumps(summary_a), encoding="utf-8")
    (run_b / "regime_summary.json").write_text(json.dumps(summary_b), encoding="utf-8")
    (run_a / "transition_matrix.json").write_text(json.dumps(transition_stub), encoding="utf-8")
    (run_b / "transition_matrix.json").write_text(json.dumps(transition_stub), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.post(
        "/runs/compare", json={"run_ids": [run_a.name, run_b.name]}
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["metrics"]["best_val_ll"] == 10.0
    assert payload["runs"][1]["metrics"]["test_avg_ll"] == 1.2
    assert abs(payload["diffs"]["delta_best_val_ll"] - 2.5) < 1e-12
    assert abs(payload["diffs"]["delta_transition_entropy"] - 0.4) < 1e-12
    assert abs(payload["diffs"]["delta_shock_occupancy"] - 0.15) < 1e-12


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
    assert "Pin this run" in response.text
    assert "Active run" in response.text
    assert "Unpin" in response.text


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


def test_forecast_v2_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20260115T000000Z_fcastv2"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "model_params.json").write_text(
        json.dumps(
            {
                "n_states": 3,
                "initial_probs": [0.3, 0.4, 0.3],
                "transition_matrix": [
                    [0.9, 0.08, 0.02],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.2, 0.6],
                ],
                "mu": [0.001, 0.0, -0.01],
                "sigma": [0.01, 0.015, 0.04],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-14", "2026-01-15"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.get(f"/forecast_v2?run_id={run_id}&horizon=4")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_id
    assert payload["horizon"] == 4
    assert len(payload["forecast"]) == 4
    first = payload["forecast"][0]
    assert "state_probs" in first
    assert "expected_return" in first
    assert "expected_vol" in first
    assert "expected_price_index" in first

    pin_resp = client.post(f"/runs/{run_id}/pin")
    assert pin_resp.status_code == 200
    pinned_forecast = client.get("/forecast_v2?use_pinned=true&horizon=2")
    assert pinned_forecast.status_code == 200
    assert pinned_forecast.json()["run_id"] == run_id

    missing_run_id = "run_20260116T000000Z_missing"
    missing_dir = runs_root / missing_run_id
    missing_dir.mkdir(parents=True, exist_ok=True)
    (missing_dir / "model_params.json").write_text(
        json.dumps(
            {
                "n_states": 3,
                "initial_probs": [0.3, 0.4, 0.3],
                "transition_matrix": [
                    [0.9, 0.08, 0.02],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.2, 0.6],
                ],
                "mu": [0.001, 0.0, -0.01],
                "sigma": [0.01, 0.015, 0.04],
            }
        ),
        encoding="utf-8",
    )
    missing_resp = client.get(f"/forecast_v2?run_id={missing_run_id}&horizon=2")
    assert missing_resp.status_code == 404
    assert "predict_proba.json" in missing_resp.json()["detail"]


def test_events_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20260120T000000Z_events"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23"],
                "states": [0, 0, 2, 2],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23"],
                "returns": [0.01, -0.02, 0.03, 0.01],
                "regime_probabilities": [
                    [0.9, 0.05, 0.05],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/events")
    assert by_id.status_code == 200
    payload = by_id.json()
    assert payload["run_id"] == run_id
    assert payload["n_events"] == 2
    assert payload["events"][0]["label"] == "low_vol"
    assert payload["events"][1]["label"] == "shock"
    assert payload["events"][0]["length"] == 2
    assert (run_dir / "events.json").exists()

    latest = client.get("/runs/latest/events")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id
