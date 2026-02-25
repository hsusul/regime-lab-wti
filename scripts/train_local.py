"""CLI script to train a local WTI regime model run."""

from __future__ import annotations

import argparse
import json

from models.train import train_model_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local WTI HMM regime model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refetch of EIA data and overwrite cache.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom run id. Auto-generated if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model_run(
        config_path=args.config,
        force_refresh=args.force_refresh,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
