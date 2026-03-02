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


def list_trash(
    *,
    limit: int = 25,
    offset: int = 0,
    order: str = "desc",
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    return _request_json(
        "GET",
        f"/runs/trash?limit={int(limit)}&offset={int(offset)}&order={order}",
        base_url=base_url,
    )


def get_trash(trash_id: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", f"/runs/trash/{trash_id}", base_url=base_url)


def purge_trash(trash_id: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("DELETE", f"/runs/trash/{trash_id}", base_url=base_url)


def get_notes(run_id: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", f"/runs/{run_id}/notes", base_url=base_url)


def put_notes(run_id: str, content: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json(
        "PUT",
        f"/runs/{run_id}/notes",
        base_url=base_url,
        payload={"content": content},
    )


def compare_runs(run_a: str, run_b: str, *, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request_json("GET", f"/runs/{run_a}/compare/{run_b}", base_url=base_url)


def evaluate_alerts(
    *,
    run_id: str | None = None,
    use_pinned: bool = False,
    rules: dict[str, Any] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"use_pinned": bool(use_pinned)}
    if run_id is not None:
        payload["run_id"] = run_id
    if rules is not None:
        payload["rules"] = rules
    return _request_json("POST", "/alerts/evaluate", base_url=base_url, payload=payload)


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

    trash_list = sub.add_parser("trash_list", help="GET /runs/trash")
    trash_list.add_argument("--limit", type=int, default=25)
    trash_list.add_argument("--offset", type=int, default=0)
    trash_list.add_argument("--order", default="desc")

    trash_get = sub.add_parser("trash_get", help="GET /runs/trash/{trash_id}")
    trash_get.add_argument("--trash-id", required=True)

    trash_purge = sub.add_parser("trash_purge", help="DELETE /runs/trash/{trash_id}")
    trash_purge.add_argument("--trash-id", required=True)

    notes_get = sub.add_parser("notes_get", help="GET /runs/{run_id}/notes")
    notes_get.add_argument("--run-id", required=True)

    notes_put = sub.add_parser("notes_put", help="PUT /runs/{run_id}/notes")
    notes_put.add_argument("--run-id", required=True)
    notes_put.add_argument("--content", required=True)

    compare = sub.add_parser("compare", help="GET /runs/{run_a}/compare/{run_b}")
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)

    alerts = sub.add_parser("alerts_evaluate", help="POST /alerts/evaluate")
    alerts.add_argument("--run-id", default=None)
    alerts.add_argument("--use-pinned", action="store_true")
    alerts.add_argument(
        "--rules-json",
        default=None,
        help="Optional JSON object string for rules",
    )

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
    if args.command == "trash_list":
        print(
            json.dumps(
                list_trash(
                    limit=int(args.limit),
                    offset=int(args.offset),
                    order=str(args.order),
                    base_url=base_url,
                ),
                indent=2,
            )
        )
        return
    if args.command == "trash_get":
        print(json.dumps(get_trash(trash_id=str(args.trash_id), base_url=base_url), indent=2))
        return
    if args.command == "trash_purge":
        print(json.dumps(purge_trash(trash_id=str(args.trash_id), base_url=base_url), indent=2))
        return
    if args.command == "notes_get":
        print(json.dumps(get_notes(run_id=str(args.run_id), base_url=base_url), indent=2))
        return
    if args.command == "notes_put":
        print(
            json.dumps(
                put_notes(run_id=str(args.run_id), content=str(args.content), base_url=base_url),
                indent=2,
            )
        )
        return
    if args.command == "compare":
        print(
            json.dumps(
                compare_runs(run_a=str(args.run_a), run_b=str(args.run_b), base_url=base_url),
                indent=2,
            )
        )
        return
    if args.command == "alerts_evaluate":
        rules = None
        if args.rules_json:
            rules = json.loads(str(args.rules_json))
        print(
            json.dumps(
                evaluate_alerts(
                    run_id=args.run_id,
                    use_pinned=bool(args.use_pinned),
                    rules=rules,
                    base_url=base_url,
                ),
                indent=2,
            )
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
