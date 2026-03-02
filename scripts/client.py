"""Tiny HTTP client for the local WTI Regime Monitor API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _request_json(method: str, path: str, *, base_url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.request(method, path, json=payload)
        response.raise_for_status()
        return response.json()


def list_runs(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", "/runs", base_url=base_url)


def get_latest(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", "/runs/latest", base_url=base_url)


def get_scorecard(run_id: str, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", f"/runs/{run_id}/scorecard", base_url=base_url)


def predict_current(
    *,
    include_probs: bool = False,
    run_id: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    query = "?include_probs=true" if include_probs else ""
    payload = {"run_id": run_id} if run_id is not None else {}
    return _request_json("POST", f"/predict_current{query}", base_url=base_url, payload=payload)


def fetch_artifact(run_id: str, name: str, *, base_url: str = DEFAULT_BASE_URL) -> bytes:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.get(f"/runs/{run_id}/artifacts/{name}")
        response.raise_for_status()
        return response.content


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small client for local WTI Regime Monitor API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list_runs", help="GET /runs")
    sub.add_parser("latest", help="GET /runs/latest")

    scorecard = sub.add_parser("scorecard", help="GET /runs/{run_id}/scorecard")
    scorecard.add_argument("--run-id", required=True)

    predict = sub.add_parser("predict_current", help="POST /predict_current")
    predict.add_argument("--include-probs", action="store_true")
    predict.add_argument("--run-id", default=None)

    artifact = sub.add_parser("fetch_artifact", help="GET /runs/{run_id}/artifacts/{name}")
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--name", required=True)
    artifact.add_argument("--out", default=None, help="Output path for artifact bytes")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    base_url = str(args.base_url)

    if args.command == "list_runs":
        print(json.dumps(list_runs(base_url=base_url), indent=2))
        return
    if args.command == "latest":
        print(json.dumps(get_latest(base_url=base_url), indent=2))
        return
    if args.command == "scorecard":
        print(json.dumps(get_scorecard(run_id=str(args.run_id), base_url=base_url), indent=2))
        return
    if args.command == "predict_current":
        print(
            json.dumps(
                predict_current(
                    include_probs=bool(args.include_probs),
                    run_id=args.run_id,
                    base_url=base_url,
                ),
                indent=2,
            )
        )
        return
    if args.command == "fetch_artifact":
        data = fetch_artifact(run_id=str(args.run_id), name=str(args.name), base_url=base_url)
        out_path = Path(args.out) if args.out else Path(str(args.name)).name
        path = Path(out_path)
        path.write_bytes(data)
        print(path)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

