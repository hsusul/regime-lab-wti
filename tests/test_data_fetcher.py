from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from energy_data.eia_client import EIAClientConfig, EIAWTIClient


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_fetcher_caches_and_reads_offline(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    def fake_get(self, url, params):  # noqa: ANN001, ANN201
        calls["count"] += 1
        offset = int(params["offset"])
        if offset == 0:
            return DummyResponse(
                {
                    "response": {
                        "total": "3",
                        "data": [
                            {"period": "2024-01-02", "value": "70.0"},
                            {"period": "2024-01-03", "value": "71.0"},
                        ],
                    }
                }
            )
        return DummyResponse(
            {
                "response": {
                    "total": "3",
                    "data": [
                        {"period": "2024-01-04", "value": "72.0"},
                    ],
                }
            }
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    client = EIAWTIClient(
        EIAClientConfig(cache_dir=tmp_path, cache_filename="wti.csv", page_size=2)
    )
    df = client.fetch_daily_wti(force_refresh=True)

    assert calls["count"] >= 2
    assert list(df.columns) == ["date", "price"]
    assert len(df) == 3
    assert (tmp_path / "wti.csv").exists()

    def offline_get(self, url, params):  # noqa: ANN001, ANN201
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.Client, "get", offline_get)
    cached = client.fetch_daily_wti(force_refresh=False)

    assert len(cached) == 3
    assert pd.api.types.is_datetime64_any_dtype(cached["date"])
