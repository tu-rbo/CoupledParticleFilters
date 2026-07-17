"""Command-line interface for offline experiments."""

from __future__ import annotations

import argparse
import logging

from .config import load_experiment_config
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a coupled particle-filter experiment")
    parser.add_argument("--config", required=True, help="Complete experiment YAML")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and resolve the experiment config without running data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.output.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.validate_only:
        print(f"valid experiment: {config.name}")
        return 0
    run_experiment(config)
    return 0
