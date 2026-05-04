from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_uplift.bundle import build_inference_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the production inference bundle")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_full.yaml",
        help="Path to the training configuration used for the production report",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/production",
        help="Directory where the production bundle will be written",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = build_inference_bundle(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
