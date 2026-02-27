from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from models.train import train_model_run


def test_train_model_run_short_series_error_message(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch_daily_wti(self, force_refresh=False, start_date=None, end_date=None):  # noqa: ANN001, ANN201
        return pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ],
                "price": [100.0, 101.0, 0.0, None, -2.0, 99.0],
            }
        )

    monkeypatch.setattr(
        "models.train.EIAWTIClient.fetch_daily_wti",
        fake_fetch_daily_wti,
    )

    cfg = {
        "run_dir": str(tmp_path / "runs"),
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "cache_filename": "wti_test.csv",
            "start_date": None,
            "end_date": None,
        },
        "split": {"train_frac": 0.7, "val_frac": 0.15},
        "model": {
            "n_states": 3,
            "learning_rate": 0.05,
            "max_epochs": 10,
            "patience": 2,
            "min_delta": 0.001,
            "seed": 42,
        },
        "forecast": {"default_horizon": 5, "interval": 0.95},
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        train_model_run(config_path=cfg_path, force_refresh=False)

    message = str(exc_info.value)
    assert "Insufficient positive price history" in message
    assert "original_count=" in message
    assert "filtered_count=" in message
    assert "dropped_non_positive_or_nan=" in message
