from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
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


def test_health_and_ready_with_empty_runs(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code == 200
    payload = ready.json()
    assert payload["status"] == "ok"
    assert payload["runs_root_exists"] is True
    assert payload["latest"]["available"] is False
    assert payload["active"]["source"] == "none"


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
        "event_classifier_summary": {
            "event_counts_by_label": {"low_vol": 2, "mid_vol": 1, "shock": 1},
            "avg_event_duration_days_by_label": {"low_vol": 5.0, "mid_vol": 2.0, "shock": 1.0},
            "top_5_longest_events": [],
        },
        "regime_stability": {
            "window_days": 60,
            "label_flip_rate_last_n_days": 0.15,
            "avg_segment_length_by_label": {"low_vol": 5.0, "mid_vol": 3.0, "shock": 2.0},
        },
        "baseline_comparison": {
            "best_baseline_name": "single_regime_gaussian",
            "delta_vs_hmm_test_avg_log_likelihood": -0.02,
        },
    }
    (run_dir / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/evaluation")
    assert by_id.status_code == 200
    assert by_id.json()["run_id"] == run_id
    assert "regime_diagnostics" in by_id.json()
    assert "event_classifier_summary" in by_id.json()
    assert "event_counts_by_label" in by_id.json()["event_classifier_summary"]
    assert "regime_stability" in by_id.json()
    assert "baseline_comparison" in by_id.json()

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
    assert "Compare pinned vs latest" in response.text
    assert "Download bundle.zip" in response.text
    assert "View report.md" in response.text
    assert "Tags" in response.text
    assert "Run Notes" in response.text
    assert "Save notes.md" in response.text
    assert "Trash" in response.text
    assert "Purge trash entry" in response.text
    assert "Frozen:" in response.text


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


def test_download_artifact_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260110T000000Z_download"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "payload.json").write_text('{"x":1}', encoding="utf-8")
    (run_dir / "page.html").write_text("<html>ok</html>", encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    json_resp = client.get(f"/runs/{run_id}/artifacts/payload.json")
    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"].startswith("application/json")
    assert json_resp.content == b'{"x":1}'

    html_resp = client.get(f"/runs/{run_id}/artifacts/page.html")
    assert html_resp.status_code == 200
    assert html_resp.headers["content-type"].startswith("text/html")
    assert html_resp.content == b"<html>ok</html>"

    missing_resp = client.get(f"/runs/{run_id}/artifacts/missing.json")
    assert missing_resp.status_code == 404
    assert run_id in missing_resp.json()["detail"]
    assert "missing.json" in missing_resp.json()["detail"]


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
                "robust_sigma": [0.012, 0.017, 0.05],
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
    assert payload["events"][0]["segment_index"] == 0
    assert payload["events"][0]["start_idx"] == 0
    assert payload["events"][0]["end_idx"] == 1
    assert payload["events"][0]["duration_days"] == 2
    assert "mean_return" in payload["events"][0]
    assert "realized_vol" in payload["events"][0]
    assert payload["events"][0]["length"] == 2
    assert (run_dir / "events.json").exists()

    latest = client.get("/runs/latest/events")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id

    filtered_label = client.get(f"/runs/{run_id}/events?label=shock")
    assert filtered_label.status_code == 200
    assert filtered_label.json()["n_events"] == 1
    assert filtered_label.json()["events"][0]["label"] == "shock"

    filtered_duration = client.get(f"/runs/{run_id}/events?min_duration_days=3")
    assert filtered_duration.status_code == 200
    assert filtered_duration.json()["n_events"] == 0


def test_events_enrichment_for_legacy_payload(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260121T000000Z_events_legacy"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "n_events": 1,
                "events": [
                    {
                        "state": 0,
                        "label": "low_vol",
                        "start_date": "2026-01-20",
                        "end_date": "2026-01-21",
                        "length": 2,
                        "cumulative_log_return": -0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-01-20", "2026-01-21"], "states": [0, 0]}),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-20", "2026-01-21"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/events")
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_events"] == 1
    assert "segment_index" in payload["events"][0]
    assert "duration_days" in payload["events"][0]
    assert "mean_return" in payload["events"][0]


def test_scorecard_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260122T000000Z_score"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "best_val_log_likelihood": 12.3,
                    "epochs_ran": 42,
                    "stopped_early": True,
                },
                "regime_diagnostics": {
                    "transition_entropy": 0.77,
                    "regimes": [
                        {"label": "low_vol", "avg_posterior_probability": 0.7},
                        {"label": "mid_vol", "avg_posterior_probability": 0.2},
                        {"label": "shock", "avg_posterior_probability": 0.1},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "n_events": 3,
                "events": [
                    {
                        "segment_index": 0,
                        "label": "low_vol",
                        "start_idx": 0,
                        "end_idx": 0,
                        "duration_days": 1,
                        "mean_return": 0.0,
                        "realized_vol": 0.0,
                    },
                    {
                        "segment_index": 1,
                        "label": "shock",
                        "start_idx": 1,
                        "end_idx": 1,
                        "duration_days": 1,
                        "mean_return": 0.0,
                        "realized_vol": 0.0,
                    },
                    {
                        "segment_index": 2,
                        "label": "shock",
                        "start_idx": 2,
                        "end_idx": 2,
                        "duration_days": 1,
                        "mean_return": 0.0,
                        "realized_vol": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps(
            {
                "dates": ["2026-01-20", "2026-01-21"],
                "states": [0, 2],
                "labels": ["low_vol", "shock"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/scorecard")
    assert by_id.status_code == 200
    payload = by_id.json()
    assert payload["run_id"] == run_id
    assert isinstance(payload["metrics"]["best_val_log_likelihood"], float)
    assert isinstance(payload["metrics"]["epochs_ran"], int)
    assert isinstance(payload["metrics"]["stopped_early"], bool)
    assert isinstance(payload["diagnostics"]["transition_entropy"], float)
    assert isinstance(payload["diagnostics"]["shock_occupancy"], float)
    assert payload["events_by_label"]["shock"] == 2
    assert payload["last_regime"]["label"] == "shock"

    latest = client.get("/runs/latest/scorecard")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id


def test_runs_endpoint_pagination_and_order(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_ids = [
        "run_20260101T000000Z_a",
        "run_20260102T000000Z_b",
        "run_20260103T000000Z_c",
    ]
    for run_id in run_ids:
        (runs_root / run_id).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    default_resp = client.get("/runs")
    assert default_resp.status_code == 200
    assert default_resp.json()["runs"] == sorted(run_ids, reverse=True)

    paged_resp = client.get("/runs?order=asc&limit=1&offset=1")
    assert paged_resp.status_code == 200
    assert paged_resp.json()["runs"] == [sorted(run_ids)[1]]

    invalid_order_resp = client.get("/runs?order=invalid")
    assert invalid_order_resp.status_code == 400


def test_tags_endpoints_and_manifest_hash(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_tags"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    missing_get = client.get(f"/runs/{run_id}/tags")
    assert missing_get.status_code == 404

    put_resp = client.put(
        f"/runs/{run_id}/tags",
        json={"tags": ["prod", "daily", "prod"], "notes": "primary deployment run"},
    )
    assert put_resp.status_code == 200
    put_payload = put_resp.json()
    assert put_payload["tags"] == ["daily", "prod"]
    assert "updated_at_utc" in put_payload

    get_resp = client.get(f"/runs/{run_id}/tags")
    assert get_resp.status_code == 200
    assert get_resp.json()["notes"] == "primary deployment run"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "tags.json" in manifest["artifacts"]
    assert "schema_version" in manifest
    assert "artifacts_sha256" in manifest
    assert "tags.json" in manifest["artifacts_sha256"]


def test_freeze_blocks_mutating_writes(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_frozen"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-02-01", "2026-02-02"], "states": [0, 0]}),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-02-01", "2026-02-02"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    freeze_resp = client.post(f"/runs/{run_id}/freeze", json={"reason": "lock for release"})
    assert freeze_resp.status_code == 200
    assert freeze_resp.json()["frozen"] is True

    tags_resp = client.put(f"/runs/{run_id}/tags", json={"tags": ["x"]})
    assert tags_resp.status_code == 409
    assert run_id in tags_resp.json()["detail"]
    run_mutations = json.loads((run_dir / "mutations.json").read_text(encoding="utf-8"))
    assert len(run_mutations.get("mutations", [])) == 1
    assert run_mutations["mutations"][0]["action"] == "freeze.post"
    global_audit = json.loads((runs_root / "_mutations_audit.json").read_text(encoding="utf-8"))
    blocked = [m for m in global_audit["mutations"] if m["status"] == "blocked_attempt"]
    assert len(blocked) >= 1
    assert blocked[-1]["endpoint"] == "/runs/{run_id}/tags"
    assert blocked[-1]["run_id"] == run_id

    events_resp = client.get(f"/runs/{run_id}/events")
    assert events_resp.status_code == 409
    assert "frozen" in events_resp.json()["detail"]

    unfreeze_resp = client.post(f"/runs/{run_id}/unfreeze")
    assert unfreeze_resp.status_code == 200
    assert unfreeze_resp.json()["unfrozen"] is True

    tags_after = client.put(f"/runs/{run_id}/tags", json={"tags": ["x"]})
    assert tags_after.status_code == 200


def test_integrity_endpoint_detects_mismatch(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_integrity"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = run_dir / "a.json"
    payload_path.write_text('{"ok":1}', encoding="utf-8")
    expected_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "schema_version": 1,
                "artifacts": ["manifest.json", "a.json"],
                "artifacts_sha256": {"a.json": expected_hash},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    ok_resp = client.get(f"/runs/{run_id}/integrity")
    assert ok_resp.status_code == 200
    assert ok_resp.json()["ok"] is True

    payload_path.write_text('{"ok":2}', encoding="utf-8")
    bad_resp = client.get(f"/runs/{run_id}/integrity")
    assert bad_resp.status_code == 200
    body = bad_resp.json()
    assert body["ok"] is False
    assert len(body["mismatched"]) == 1
    assert body["mismatched"][0]["name"] == "a.json"


def test_integrity_endpoint_warns_when_data_meta_missing(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_integrity_missing_data_meta"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload_path = run_dir / "a.json"
    payload_path.write_text('{"ok":1}', encoding="utf-8")
    expected_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "schema_version": 1,
                "data_hash": "abc123",
                "artifacts": ["manifest.json", "a.json", "data_meta.json"],
                "artifacts_sha256": {"a.json": expected_hash},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.get(f"/runs/{run_id}/integrity")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["mismatched"] == []
    assert payload["missing"] == []
    assert payload["warnings"]
    assert "data_meta.json" in payload["warnings"][0]


def test_integrity_endpoint_detects_tampered_data_meta(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_integrity_data_meta"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_meta_path = run_dir / "data_meta.json"
    data_meta_path.write_text(
        json.dumps({"dataset_hash": "abc123", "row_counts": {"feature_rows": 100}}),
        encoding="utf-8",
    )
    hash_initial = hashlib.sha256(data_meta_path.read_bytes()).hexdigest()
    manifest = {
        "run_id": run_id,
        "schema_version": 1,
        "data_hash": "abc123",
        "artifacts": ["manifest.json", "data_meta.json"],
        "artifacts_sha256": {"data_meta.json": hash_initial},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    ok_resp = client.get(f"/runs/{run_id}/integrity")
    assert ok_resp.status_code == 200
    assert ok_resp.json()["ok"] is True

    data_meta_path.write_text(
        json.dumps({"dataset_hash": "tampered999", "row_counts": {"feature_rows": 100}}),
        encoding="utf-8",
    )
    manifest["artifacts_sha256"]["data_meta.json"] = hashlib.sha256(data_meta_path.read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bad_resp = client.get(f"/runs/{run_id}/integrity")
    assert bad_resp.status_code == 200
    payload = bad_resp.json()
    assert payload["ok"] is False
    assert any(row["name"] == "data_hash" for row in payload["mismatched"])


def test_bundle_zip_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_bundle"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "a.json").write_text('{"a":1}', encoding="utf-8")
    (run_dir / "b.html").write_text("<html>b</html>", encoding="utf-8")
    hash_a = hashlib.sha256((run_dir / "a.json").read_bytes()).hexdigest()
    hash_b = hashlib.sha256((run_dir / "b.html").read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "schema_version": 1,
                "artifacts": ["a.json", "b.html", "manifest.json"],
                "artifacts_sha256": {"a.json": hash_a, "b.html": hash_b},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/bundle.zip?artifacts=a.json,b.html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert "attachment" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content), mode="r") as zf:
        names = sorted(zf.namelist())
        assert names == ["a.json", "b.html", "integrity.json"]
        assert zf.namelist() == sorted(zf.namelist())


def test_bundle_zip_extras_opt_in(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260201T000000Z_bundle_extras"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "a.json").write_text('{"a":1}', encoding="utf-8")
    hash_a = hashlib.sha256((run_dir / "a.json").read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "schema_version": 1,
                "artifacts": ["a.json", "manifest.json"],
                "artifacts_sha256": {"a.json": hash_a},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    no_extras = client.get(f"/runs/{run_id}/bundle.zip")
    assert no_extras.status_code == 200
    with zipfile.ZipFile(io.BytesIO(no_extras.content), mode="r") as zf:
        names = sorted(zf.namelist())
        assert names == ["a.json", "integrity.json", "manifest.json"]

    with_extras = client.get(f"/runs/{run_id}/bundle.zip?extras=openapi,run_info")
    assert with_extras.status_code == 200
    with zipfile.ZipFile(io.BytesIO(with_extras.content), mode="r") as zf:
        names = sorted(zf.namelist())
        assert "openapi.json" in names
        assert "RUN_INFO.json" in names
        assert "integrity.json" in names
        assert zf.namelist() == sorted(zf.namelist())


def test_compare_pinned_latest_and_transition_alerts(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_a = runs_root / "run_20260201T000000Z_a"
    run_b = runs_root / "run_20260202T000000Z_b"
    run_a.mkdir()
    run_b.mkdir()

    eval_payload = {
        "metrics": {"best_val_log_likelihood": 10.0, "test_avg_log_likelihood": 1.0},
        "regime_diagnostics": {"transition_entropy": 0.5},
    }
    summary_payload = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.6},
            {"label": "mid_vol", "avg_posterior_probability": 0.3},
            {"label": "shock", "avg_posterior_probability": 0.1},
        ]
    }
    transition_payload = {"transition_matrix": [[0.9, 0.1], [0.2, 0.8]]}
    for run_dir in (run_a, run_b):
        (run_dir / "evaluation.json").write_text(json.dumps(eval_payload), encoding="utf-8")
        (run_dir / "regime_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
        (run_dir / "transition_matrix.json").write_text(
            json.dumps(transition_payload), encoding="utf-8"
        )

    (run_b / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_b.name,
                "n_events": 3,
                "events": [
                    {
                        "segment_index": 0,
                        "state": 0,
                        "label": "low_vol",
                        "start_idx": 0,
                        "end_idx": 1,
                        "start_date": "2026-01-01",
                        "end_date": "2026-01-02",
                        "length": 2,
                        "duration_days": 2,
                        "cumulative_log_return": 0.01,
                        "mean_return": 0.005,
                        "realized_vol": 0.01,
                    },
                    {
                        "segment_index": 1,
                        "state": 2,
                        "label": "shock",
                        "start_idx": 2,
                        "end_idx": 2,
                        "start_date": "2026-01-03",
                        "end_date": "2026-01-03",
                        "length": 1,
                        "duration_days": 1,
                        "cumulative_log_return": -0.03,
                        "mean_return": -0.03,
                        "realized_vol": 0.0,
                    },
                    {
                        "segment_index": 2,
                        "state": 1,
                        "label": "mid_vol",
                        "start_idx": 3,
                        "end_idx": 4,
                        "start_date": "2026-01-04",
                        "end_date": "2026-01-05",
                        "length": 2,
                        "duration_days": 2,
                        "cumulative_log_return": 0.02,
                        "mean_return": 0.01,
                        "realized_vol": 0.005,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    pin_resp = client.post(f"/runs/{run_a.name}/pin")
    assert pin_resp.status_code == 200

    compare_resp = client.post("/runs/compare_pinned_latest")
    assert compare_resp.status_code == 200
    compare_payload = compare_resp.json()
    assert compare_payload["pinned_run_id"] == run_a.name
    assert compare_payload["latest_run_id"] == run_b.name
    assert len(compare_payload["runs"]) == 2

    alert_resp = client.post(
        "/alerts/transition",
        json={"run_id": run_b.name, "from_label": "low_vol", "lookback_days": 365},
    )
    assert alert_resp.status_code == 200
    alert_payload = alert_resp.json()
    assert alert_payload["run_id"] == run_b.name
    assert alert_payload["count"] == 1
    assert alert_payload["last_transition"]["from_label"] == "low_vol"


def test_version_endpoint_and_additive_fields(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_version"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": {"best_val_log_likelihood": 1.0, "epochs_ran": 2, "stopped_early": False},
                "regime_diagnostics": {
                    "transition_entropy": 0.5,
                    "regimes": [
                        {"label": "low_vol", "avg_posterior_probability": 0.6},
                        {"label": "mid_vol", "avg_posterior_probability": 0.3},
                        {"label": "shock", "avg_posterior_probability": 0.1},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "n_events": 1,
                "events": [
                    {
                        "segment_index": 0,
                        "state": 0,
                        "label": "low_vol",
                        "start_idx": 0,
                        "end_idx": 0,
                        "start_date": "2026-03-03",
                        "end_date": "2026-03-03",
                        "length": 1,
                        "duration_days": 1,
                        "cumulative_log_return": 0.0,
                        "mean_return": 0.0,
                        "realized_vol": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-03"], "states": [0], "labels": ["low_vol"]}),
        encoding="utf-8",
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    version = client.get("/version")
    assert version.status_code == 200
    version_payload = version.json()
    assert "api_version" in version_payload
    assert "schema_version" in version_payload
    assert "built_at_utc" in version_payload

    latest = client.get("/runs/latest")
    assert latest.status_code == 200
    assert "api_version" in latest.json()
    assert "schema_version" in latest.json()

    scorecard = client.get(f"/runs/{run_id}/scorecard")
    assert scorecard.status_code == 200
    assert "api_version" in scorecard.json()
    assert "schema_version" in scorecard.json()

    current = client.post("/predict_current", json={"run_id": run_id})
    assert current.status_code == 200
    assert "api_version" in current.json()
    assert "schema_version" in current.json()


def test_delete_and_restore_run_lifecycle(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "run_20260301T000000Z_a"
    run_b = runs_root / "run_20260302T000000Z_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (runs_root / "latest_run.txt").write_text(run_b.name, encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    delete_resp = client.delete(f"/runs/{run_b.name}")
    assert delete_resp.status_code == 200
    trash_id = delete_resp.json()["trash_id"]
    assert not run_b.exists()
    assert (runs_root / "_trash" / trash_id).exists()
    assert (runs_root / "latest_run.txt").read_text(encoding="utf-8").strip() == run_a.name

    restore_resp = client.post(f"/runs/trash/{trash_id}/restore")
    assert restore_resp.status_code == 200
    assert (runs_root / run_b.name).exists()
    global_audit = json.loads((runs_root / "_mutations_audit.json").read_text(encoding="utf-8"))
    assert len(global_audit["mutations"]) >= 2
    delete_entry = global_audit["mutations"][-2]
    restore_entry = global_audit["mutations"][-1]
    assert delete_entry["endpoint"] == "/runs/{run_id}"
    assert delete_entry["run_id"] == run_b.name
    assert delete_entry["trash_id"] == trash_id
    assert delete_entry["status"] == "ok"
    assert "ts_utc" in delete_entry
    assert restore_entry["endpoint"] == "/runs/trash/{trash_id}/restore"
    assert restore_entry["run_id"] == run_b.name
    assert restore_entry["trash_id"] == trash_id
    assert restore_entry["status"] == "ok"


def test_delete_rejects_pinned_or_frozen(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    pinned = runs_root / "run_20260301T000000Z_pinned"
    frozen = runs_root / "run_20260302T000000Z_frozen"
    pinned.mkdir(parents=True, exist_ok=True)
    frozen.mkdir(parents=True, exist_ok=True)
    (runs_root / "pinned_run.txt").write_text(pinned.name, encoding="utf-8")
    (frozen / "frozen.json").write_text(json.dumps({"frozen": True}), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    pinned_resp = client.delete(f"/runs/{pinned.name}")
    assert pinned_resp.status_code == 409
    frozen_resp = client.delete(f"/runs/{frozen.name}")
    assert frozen_resp.status_code == 409


def test_json_read_cache_invalidation(tmp_path: Path) -> None:
    tmp_root = tmp_path / "cache"
    tmp_root.mkdir(parents=True, exist_ok=True)
    path = tmp_root / "manifest.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    first = routes._read_json(path, run_id="run_cache")
    first["value"] = 999
    again = routes._read_json(path, run_id="run_cache")
    assert again["value"] == 1
    time.sleep(0.01)
    path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    os.utime(path, None)
    updated = routes._read_json(path, run_id="run_cache")
    assert updated["value"] == 2


def test_mutation_api_key_guard(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_auth"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    monkeypatch.setenv("REGIME_API_KEY", "secret")
    client = TestClient(app)

    missing_pin = client.post(f"/runs/{run_id}/pin")
    assert missing_pin.status_code == 401
    ok_pin = client.post(f"/runs/{run_id}/pin", headers={"X-API-Key": "secret"})
    assert ok_pin.status_code == 200

    missing_tags = client.put(f"/runs/{run_id}/tags", json={"tags": ["x"]})
    assert missing_tags.status_code == 401
    ok_tags = client.put(
        f"/runs/{run_id}/tags",
        json={"tags": ["x"]},
        headers={"X-API-Key": "secret"},
    )
    assert ok_tags.status_code == 200
    global_audit = json.loads((runs_root / "_mutations_audit.json").read_text(encoding="utf-8"))
    statuses = [entry["status"] for entry in global_audit["mutations"]]
    assert "denied_auth" in statuses
    assert "ok" in statuses


def test_forecast_eval_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_forecast_eval"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "interval": 0.95,
        "window": 120,
        "include_quantiles": True,
        "horizons": [
            {
                "horizon": 1,
                "n_samples": 100,
                "mae": 0.01,
                "coverage": 0.93,
                "quantile_coverage_p05_p95": 0.92,
            },
            {
                "horizon": 2,
                "n_samples": 99,
                "mae": 0.012,
                "coverage": 0.92,
                "quantile_coverage_p05_p95": 0.9,
            },
        ],
    }
    (run_dir / "forecast_eval.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/forecast_eval")
    assert by_id.status_code == 200
    assert by_id.json()["run_id"] == run_id
    assert by_id.json()["include_quantiles"] is True
    assert "quantile_coverage_p05_p95" in by_id.json()["horizons"][0]

    latest = client.get("/runs/latest/forecast_eval")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id


def test_backtest_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_backtest"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "strategy_name": "avoid_shock_regime",
        "annualized_vol_proxy": 0.12,
        "max_drawdown_proxy": -0.08,
    }
    (run_dir / "backtest.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    by_id = client.get(f"/runs/{run_id}/backtest")
    assert by_id.status_code == 200
    assert by_id.json()["run_id"] == run_id
    assert by_id.json()["strategy_name"] == "avoid_shock_regime"

    latest = client.get("/runs/latest/backtest")
    assert latest.status_code == 200
    assert latest.json()["run_id"] == run_id


def test_drift_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "run_20260303T000000Z_da"
    run_b = runs_root / "run_20260304T000000Z_db"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    eval_a = {
        "metrics": {"best_val_log_likelihood": 1.0},
        "regime_diagnostics": {"transition_entropy": 0.5},
        "event_classifier_summary": {
            "event_counts_by_label": {"low_vol": 2, "shock": 1},
            "avg_event_duration_days_by_label": {"low_vol": 10.0, "shock": 2.0},
        },
    }
    eval_b = {
        "metrics": {"best_val_log_likelihood": 2.0},
        "regime_diagnostics": {"transition_entropy": 0.8},
        "event_classifier_summary": {
            "event_counts_by_label": {"low_vol": 1, "shock": 3},
            "avg_event_duration_days_by_label": {"low_vol": 5.0, "shock": 4.0},
        },
    }
    summary_a = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.9},
            {"label": "shock", "avg_posterior_probability": 0.1},
        ]
    }
    summary_b = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.7},
            {"label": "shock", "avg_posterior_probability": 0.3},
        ]
    }
    plot_meta_a = {"occupancies_by_label": {"low_vol": 0.8, "shock": 0.2}}
    plot_meta_b = {"occupancies_by_label": {"low_vol": 0.6, "shock": 0.4}}

    (run_a / "evaluation.json").write_text(json.dumps(eval_a), encoding="utf-8")
    (run_b / "evaluation.json").write_text(json.dumps(eval_b), encoding="utf-8")
    (run_a / "regime_summary.json").write_text(json.dumps(summary_a), encoding="utf-8")
    (run_b / "regime_summary.json").write_text(json.dumps(summary_b), encoding="utf-8")
    (run_a / "plot_meta.json").write_text(json.dumps(plot_meta_a), encoding="utf-8")
    (run_b / "plot_meta.json").write_text(json.dumps(plot_meta_b), encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.post("/runs/drift", json={"run_a": run_a.name, "run_b": run_b.name})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_a"] == run_a.name
    assert payload["run_b"] == run_b.name
    assert payload["deltas"]["delta_best_val_ll"] == 1.0
    assert payload["deltas"]["delta_transition_entropy"] == 0.30000000000000004
    assert payload["deltas"]["delta_shock_occupancy"] == 0.19999999999999998
    assert isinstance(payload["deltas"]["occupancy_kl_divergence"], float)
    assert payload["event_deltas"]["event_counts_by_label"]["shock"] == 2
    assert payload["event_deltas"]["avg_event_duration_days_by_label"]["low_vol"] == -5.0


def test_report_markdown_generation_and_alias(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_report"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": {"best_val_log_likelihood": 3.0, "epochs_ran": 10, "stopped_early": False},
                "regime_diagnostics": {
                    "transition_entropy": 0.4,
                    "regimes": [{"label": "shock", "avg_posterior_probability": 0.2}],
                },
                "event_classifier_summary": {
                    "event_counts_by_label": {"shock": 1},
                    "avg_event_duration_days_by_label": {"shock": 2.0},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "n_events": 1,
                "events": [
                    {
                        "segment_index": 0,
                        "state": 2,
                        "label": "shock",
                        "start_idx": 0,
                        "end_idx": 1,
                        "start_date": "2026-03-01",
                        "end_date": "2026-03-02",
                        "length": 2,
                        "duration_days": 2,
                        "cumulative_log_return": -0.05,
                        "mean_return": -0.025,
                        "realized_vol": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-02"], "states": [2], "labels": ["shock"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/report.md")
    assert response.status_code == 200
    assert "# Run Report:" in response.text
    assert "## Scorecard" in response.text
    assert "## Top Events" in response.text
    assert (run_dir / "report.md").exists()

    latest = client.get("/runs/latest/report.md")
    assert latest.status_code == 200
    assert "# Run Report:" in latest.text

    html_resp = client.get(f"/runs/{run_id}/report.html")
    assert html_resp.status_code == 200
    assert "<!doctype html>" in html_resp.text.lower()
    assert "Occupancy Chart" in html_resp.text
    assert "Transition Matrix" in html_resp.text
    assert (run_dir / "report.html").exists()

    latest_html = client.get("/runs/latest/report.html")
    assert latest_html.status_code == 200
    assert "<html" in latest_html.text.lower()


def test_report_generation_for_frozen_run_does_not_write(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260303T000000Z_frozen_report"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frozen.json").write_text(json.dumps({"frozen": True, "reason": "hold"}), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "metrics": {"best_val_log_likelihood": 1.0},
                "regime_diagnostics": {"transition_entropy": 0.2, "regimes": []},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.json").write_text(
        json.dumps({"run_id": run_id, "n_events": 0, "events": []}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-02"], "states": [0], "labels": ["low_vol"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    response = client.get(f"/runs/{run_id}/report.md")
    assert response.status_code == 200
    assert "# Run Report:" in response.text
    assert not (run_dir / "report.md").exists()

    html_response = client.get(f"/runs/{run_id}/report.html")
    assert html_response.status_code == 200
    assert "<html" in html_response.text.lower()
    assert not (run_dir / "report.html").exists()


def test_trash_management_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    trash_root = runs_root / "_trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    trash_a = "run_20260301T000000Z_alpha_20260302T120000Z"
    trash_b = "run_20260301T000000Z_beta_20260303T120000Z"
    (trash_root / trash_a).mkdir()
    (trash_root / trash_b).mkdir()

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    list_resp = client.get("/runs/trash")
    assert list_resp.status_code == 200
    items = list_resp.json()["trash"]
    assert [item["trash_id"] for item in items] == [trash_b, trash_a]
    assert items[0]["original_run_id"] == "run_20260301T000000Z_beta"
    assert items[0]["deleted_at_utc"].startswith("2026-03-03T12:00:00")
    assert items[0]["path"].endswith(trash_b)

    paged_resp = client.get("/runs/trash?order=asc&limit=1&offset=1")
    assert paged_resp.status_code == 200
    assert [item["trash_id"] for item in paged_resp.json()["trash"]] == [trash_b]

    get_resp = client.get(f"/runs/trash/{trash_a}")
    assert get_resp.status_code == 200
    assert get_resp.json()["trash_id"] == trash_a

    purge_resp = client.delete(f"/runs/trash/{trash_a}")
    assert purge_resp.status_code == 200
    assert purge_resp.json()["purged"] is True
    assert not (trash_root / trash_a).exists()

    invalid_order = client.get("/runs/trash?order=bad")
    assert invalid_order.status_code == 400


def test_notes_and_mutations_endpoints(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_notes"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    missing_notes = client.get(f"/runs/{run_id}/notes")
    assert missing_notes.status_code == 404

    put_resp = client.put(
        f"/runs/{run_id}/notes",
        json={"content": "# Deployment Notes\nall good"},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["run_id"] == run_id
    assert (run_dir / "notes.md").exists()

    notes_resp = client.get(f"/runs/{run_id}/notes")
    assert notes_resp.status_code == 200
    assert "Deployment Notes" in notes_resp.json()["content"]

    mutations_resp = client.get(f"/runs/{run_id}/mutations")
    assert mutations_resp.status_code == 200
    mutations = mutations_resp.json()["mutations"]
    assert len(mutations) == 1
    assert mutations[0]["action"] == "notes.put"
    assert mutations[0]["endpoint"] == "/runs/{run_id}/notes"
    assert mutations[0]["status"] == "ok"
    assert mutations[0]["actor"] == "anonymous"
    assert "ts_utc" in mutations[0]
    assert mutations[0]["run_id"] == run_id

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "notes.md" in manifest["artifacts"]
    assert "mutations.json" in manifest["artifacts"]
    assert "notes.md" in manifest["artifacts_sha256"]
    assert "mutations.json" in manifest["artifacts_sha256"]

    global_audit_path = runs_root / "_mutations_audit.json"
    assert global_audit_path.exists()
    global_payload = json.loads(global_audit_path.read_text(encoding="utf-8"))
    assert global_payload["mutations"][-1]["endpoint"] == "/runs/{run_id}/notes"
    assert global_payload["mutations"][-1]["run_id"] == run_id


def test_notes_put_respects_freeze_and_auth(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_notes_auth"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "artifacts": ["manifest.json"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    monkeypatch.setenv("REGIME_API_KEY", "secret")
    client = TestClient(app)

    unauthorized = client.put(f"/runs/{run_id}/notes", json={"content": "x"})
    assert unauthorized.status_code == 401

    authorized = client.put(
        f"/runs/{run_id}/notes",
        json={"content": "x"},
        headers={"X-API-Key": "secret"},
    )
    assert authorized.status_code == 200

    freeze_resp = client.post(
        f"/runs/{run_id}/freeze",
        json={"reason": "release lock"},
        headers={"X-API-Key": "secret"},
    )
    assert freeze_resp.status_code == 200

    frozen_put = client.put(
        f"/runs/{run_id}/notes",
        json={"content": "y"},
        headers={"X-API-Key": "secret"},
    )
    assert frozen_put.status_code == 409
    assert run_id in frozen_put.json()["detail"]


def test_bundle_zip_rejects_integrity_mismatch(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_bundle_bad"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "a.json").write_text('{"a":1}', encoding="utf-8")
    bad_hash = "0" * 64
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "schema_version": 1,
                "artifacts": ["manifest.json", "a.json"],
                "artifacts_sha256": {"a.json": bad_hash},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.get(f"/runs/{run_id}/bundle.zip?artifacts=a.json")
    assert resp.status_code == 409
    assert "integrity" in resp.json()["detail"]


def test_forecast_v3_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = "run_20260304T000000Z_fcastv3"
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
                "dates": ["2026-03-03", "2026-03-04"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
            }
        ),
        encoding="utf-8",
    )

    missing_id = "run_20260305T000000Z_fcastv3_missing"
    missing_dir = runs_root / missing_id
    missing_dir.mkdir(parents=True, exist_ok=True)
    (missing_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03"],
                "returns": [0.01],
                "regime_probabilities": [[0.3, 0.4, 0.3]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.get(
        f"/forecast_v3?run_id={run_id}&horizon=3&include_stationary=true&include_quantiles=true"
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_id
    assert payload["horizon"] == 3
    assert payload["as_of"] == "2026-03-04"
    assert payload["params_source"] == "model_params.json"
    assert payload["label_mapping"]["2"] == "shock"
    assert len(payload["forecast"]) == 3
    first = payload["forecast"][0]
    assert "probs_by_label" in first
    assert "raw_state_probs" in first
    assert "expected_return" in first
    assert "expected_vol" in first
    assert "interval_low" in first
    assert "interval_high" in first
    assert "p05" in first
    assert "p50" in first
    assert "p95" in first
    assert payload["quantiles_enabled"] is True
    assert payload["stationary"] is not None
    assert len(payload["stationary"]["state_probs"]) == 3
    assert set(payload["stationary"]["implied_durations_days"].keys()) == {"low_vol", "mid_vol", "shock"}

    missing_resp = client.get(f"/forecast_v3?run_id={missing_id}&horizon=2")
    assert missing_resp.status_code == 404
    assert "model_params.json" in missing_resp.json()["detail"]


def test_forecast_v3_npz_fallback(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_fcastv3_npz"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "model_params.npz",
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        mu=np.array([0.001, -0.01], dtype=float),
        sigma=np.array([0.01, 0.04], dtype=float),
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03", "2026-03-04"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.8, 0.2], [0.7, 0.3]],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)
    resp = client.get(f"/forecast_v3?run_id={run_id}&horizon=2")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["params_source"] == "model_params.npz"
    assert payload["quantiles_enabled"] is False
    assert len(payload["forecast"]) == 2


def test_forecast_v3_quantiles_prefer_robust_sigma(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_fcastv3_robust_sigma"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_params.json").write_text(
        json.dumps(
            {
                "n_states": 2,
                "initial_probs": [0.5, 0.5],
                "transition_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "mu": [0.0, 0.0],
                "sigma": [0.01, 0.01],
                "robust_sigma": [0.10, 0.10],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "shock"}}),
        encoding="utf-8",
    )
    (run_dir / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03", "2026-03-04"],
                "returns": [0.0, 0.0],
                "regime_probabilities": [[0.2, 0.8], [0.25, 0.75]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)
    resp = client.get(
        f"/forecast_v3?run_id={run_id}&horizon=1&include_quantiles=true"
    )
    assert resp.status_code == 200
    payload = resp.json()
    row = payload["forecast"][0]
    z95 = NormalDist().inv_cdf(0.95)
    assert row["p50"] == pytest.approx(row["expected_return"], rel=0, abs=1e-12)
    assert row["p95"] == pytest.approx(row["expected_return"] + z95 * 0.10, rel=0, abs=1e-9)
    assert row["p95"] > row["expected_return"] + z95 * 0.01


def test_alerts_evaluate_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_target = runs_root / "run_20260304T000000Z_alert_target"
    run_base = runs_root / "run_20260303T000000Z_alert_base"
    run_target.mkdir(parents=True, exist_ok=True)
    run_base.mkdir(parents=True, exist_ok=True)

    (run_target / "regime_summary.json").write_text(
        json.dumps(
            {
                "regimes": [
                    {"label": "low_vol", "avg_posterior_probability": 0.4},
                    {"label": "mid_vol", "avg_posterior_probability": 0.2},
                    {"label": "shock", "avg_posterior_probability": 0.4},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_target / "evaluation.json").write_text(
        json.dumps({"regime_diagnostics": {"transition_entropy": 1.0}}),
        encoding="utf-8",
    )
    (run_target / "forecast_eval.json").write_text(
        json.dumps({"horizons": [{"horizon": 1, "coverage": 0.7}]}),
        encoding="utf-8",
    )
    (run_target / "transition_matrix.json").write_text(
        json.dumps({"transition_matrix": [[0.6, 0.4, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]}),
        encoding="utf-8",
    )
    (run_target / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "mid_vol", "2": "shock"}}),
        encoding="utf-8",
    )
    (run_target / "viterbi_states.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03", "2026-03-04"],
                "states": [0, 0],
                "labels": ["low_vol", "low_vol"],
            }
        ),
        encoding="utf-8",
    )
    (run_target / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03", "2026-03-04"],
                "returns": [0.01, -0.02],
                "regime_probabilities": [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]],
            }
        ),
        encoding="utf-8",
    )
    (run_base / "evaluation.json").write_text(
        json.dumps({"regime_diagnostics": {"transition_entropy": 0.2}}),
        encoding="utf-8",
    )
    (runs_root / "pinned_run.txt").write_text(run_base.name, encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.post("/alerts/evaluate", json={"run_id": run_target.name})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_target.name
    assert payload["triggered"] is True
    assert payload["severity"] == "high"
    assert payload["selection"] == "run_id"
    assert payload["current_regime_label"] == "low_vol"
    assert payload["shock_prob"] == 0.7
    assert payload["transition_alert"]["triggered"] is True
    assert "recommended_action" in payload
    ids = {row["id"] for row in payload["alerts"]}
    assert "shock_occupancy_high" in ids
    assert "forecast_coverage_low" in ids
    assert "transition_entropy_jump" in ids
    assert "transition_probability_high" in ids
    for row in payload["alerts"]:
        assert row["severity"] in {"low", "medium", "high"}
        assert "recommended_action" in row


def test_alerts_evaluate_calibration_window(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_ids = [
        "run_20260301T000000Z_cal_a",
        "run_20260302T000000Z_cal_b",
        "run_20260303T000000Z_cal_target",
    ]
    for run_id in run_ids:
        (runs_root / run_id).mkdir(parents=True, exist_ok=True)

    fixtures = {
        run_ids[0]: {
            "end_date": "2026-03-01",
            "shock_occ": 0.1,
            "entropy": 0.2,
            "coverage": 0.95,
            "transition_matrix": [[0.7, 0.3], [0.1, 0.9]],
        },
        run_ids[1]: {
            "end_date": "2026-03-02",
            "shock_occ": 0.2,
            "entropy": 0.4,
            "coverage": 0.9,
            "transition_matrix": [[0.65, 0.35], [0.2, 0.8]],
        },
        run_ids[2]: {
            "end_date": "2026-03-03",
            "shock_occ": 0.3,
            "entropy": 0.8,
            "coverage": 0.85,
            "transition_matrix": [[0.55, 0.45], [0.25, 0.75]],
        },
    }

    for run_id, row in fixtures.items():
        run_dir = runs_root / run_id
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "end_date": row["end_date"]}),
            encoding="utf-8",
        )
        (run_dir / "regime_summary.json").write_text(
            json.dumps(
                {
                    "regimes": [
                        {"label": "low_vol", "avg_posterior_probability": 1.0 - row["shock_occ"]},
                        {"label": "shock", "avg_posterior_probability": row["shock_occ"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "evaluation.json").write_text(
            json.dumps({"regime_diagnostics": {"transition_entropy": row["entropy"]}}),
            encoding="utf-8",
        )
        (run_dir / "forecast_eval.json").write_text(
            json.dumps({"horizons": [{"horizon": 1, "coverage": row["coverage"]}]}),
            encoding="utf-8",
        )
        (run_dir / "transition_matrix.json").write_text(
            json.dumps({"transition_matrix": row["transition_matrix"]}),
            encoding="utf-8",
        )

    run_target = runs_root / run_ids[2]
    (run_target / "regime_labels.json").write_text(
        json.dumps({"label_mapping": {"0": "low_vol", "1": "shock"}}),
        encoding="utf-8",
    )
    (run_target / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-03"], "states": [0], "labels": ["low_vol"]}),
        encoding="utf-8",
    )
    (run_target / "predict_proba.json").write_text(
        json.dumps(
            {
                "dates": ["2026-03-03"],
                "returns": [0.01],
                "regime_probabilities": [[0.7, 0.3]],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)
    resp = client.post(
        "/alerts/evaluate",
        json={"run_id": run_ids[2], "calibration_window_days": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    rec = payload["threshold_recommendations"]
    assert rec["n_runs"] == 3
    assert rec["calibration_window_days"] == 5
    thresholds = rec["recommended_thresholds"]
    assert thresholds["shock_occupancy_threshold"] == pytest.approx(0.28, rel=0, abs=1e-9)
    assert thresholds["coverage_threshold"] == pytest.approx(0.86, rel=0, abs=1e-9)
    assert thresholds["transition_prob_threshold"] == pytest.approx(0.43, rel=0, abs=1e-9)
    assert thresholds["transition_entropy_jump_threshold"] == pytest.approx(0.38, rel=0, abs=1e-9)


def test_alerts_evaluate_missing_proba(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_id = "run_20260304T000000Z_alert_missing_proba"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "regime_summary.json").write_text(
        json.dumps({"regimes": [{"label": "shock", "avg_posterior_probability": 0.1}]}),
        encoding="utf-8",
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps({"regime_diagnostics": {"transition_entropy": 0.3}}),
        encoding="utf-8",
    )
    (run_dir / "transition_matrix.json").write_text(
        json.dumps({"transition_matrix": [[0.95, 0.05], [0.1, 0.9]]}),
        encoding="utf-8",
    )
    (run_dir / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-04"], "states": [0], "labels": ["low_vol"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)
    resp = client.post("/alerts/evaluate", json={"use_latest": True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == run_id
    assert payload["shock_prob"] is None
    assert payload["selection"] == "latest"
    assert payload["severity"] == "low"


def test_compare_view_endpoint(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "run_20260304T000000Z_cmp_a"
    run_b = runs_root / "run_20260305T000000Z_cmp_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    eval_a = {
        "metrics": {"best_val_log_likelihood": 1.0, "test_avg_log_likelihood": 0.5},
        "regime_diagnostics": {"transition_entropy": 0.3},
        "event_classifier_summary": {
            "event_counts_by_label": {"low_vol": 2, "shock": 1},
            "avg_event_duration_days_by_label": {"low_vol": 5.0, "shock": 2.0},
        },
    }
    eval_b = {
        "metrics": {"best_val_log_likelihood": 1.5, "test_avg_log_likelihood": 0.6},
        "regime_diagnostics": {"transition_entropy": 0.6},
        "event_classifier_summary": {
            "event_counts_by_label": {"low_vol": 1, "shock": 2},
            "avg_event_duration_days_by_label": {"low_vol": 4.0, "shock": 3.0},
        },
    }
    summary_a = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.8},
            {"label": "shock", "avg_posterior_probability": 0.2},
        ]
    }
    summary_b = {
        "regimes": [
            {"label": "low_vol", "avg_posterior_probability": 0.6},
            {"label": "shock", "avg_posterior_probability": 0.4},
        ]
    }
    transition_stub = {"transition_matrix": [[0.9, 0.1], [0.2, 0.8]]}

    (run_a / "evaluation.json").write_text(json.dumps(eval_a), encoding="utf-8")
    (run_b / "evaluation.json").write_text(json.dumps(eval_b), encoding="utf-8")
    (run_a / "regime_summary.json").write_text(json.dumps(summary_a), encoding="utf-8")
    (run_b / "regime_summary.json").write_text(json.dumps(summary_b), encoding="utf-8")
    (run_a / "transition_matrix.json").write_text(json.dumps(transition_stub), encoding="utf-8")
    (run_b / "transition_matrix.json").write_text(json.dumps(transition_stub), encoding="utf-8")
    (run_a / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-04"], "states": [0], "labels": ["low_vol"]}),
        encoding="utf-8",
    )
    (run_b / "viterbi_states.json").write_text(
        json.dumps({"dates": ["2026-03-05"], "states": [1], "labels": ["shock"]}),
        encoding="utf-8",
    )
    (run_a / "tags.json").write_text(
        json.dumps({"tags": ["prod", "daily"]}),
        encoding="utf-8",
    )
    (run_b / "tags.json").write_text(
        json.dumps({"tags": ["prod", "candidate"]}),
        encoding="utf-8",
    )
    (run_a / "notes.md").write_text("same content", encoding="utf-8")
    (run_b / "notes.md").write_text("changed content", encoding="utf-8")

    monkeypatch.setattr(routes, "RUNS_ROOT", runs_root)
    client = TestClient(app)

    resp = client.get(f"/runs/{run_a.name}/compare/{run_b.name}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_a"] == run_a.name
    assert payload["run_b"] == run_b.name
    assert "drift" in payload
    assert "metrics_diff" in payload
    assert "events_diff" in payload
    assert "delta_best_val_ll" in payload
    assert "delta_transition_entropy" in payload
    assert "delta_shock_occupancy" in payload
    assert payload["delta_last_label"] == "low_vol -> shock"
    assert payload["delta_last_date"] == "2026-03-04 -> 2026-03-05"
    assert payload["notes_diff_hint"] == "changed"
    assert payload["tags_diff"]["shared"] == ["prod"]
    assert payload["tags_diff"]["only_in_a"] == ["daily"]
    assert payload["tags_diff"]["only_in_b"] == ["candidate"]
