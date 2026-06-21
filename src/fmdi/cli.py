"""Command-line interface for reproducible FMDI experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fmdi.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an FMDI experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML configuration.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The experiment configuration must be a YAML mapping.")
    run_experiment(config)


if __name__ == "__main__":
    main()
