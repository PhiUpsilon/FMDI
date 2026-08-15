"""Command-line interface for reproducible FMDI experiments."""

import argparse
import json
from pathlib import Path

import yaml

from .experiment import run_experiment


def build_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate FMDI.")
    parser.add_argument("--dataset", choices=("physionet", "air_quality"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-ratio", type=float, default=0.1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--valid-index", type=int, default=0)
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--lambda-freq", type=float)
    parser.add_argument("--ode-steps", type=int)
    parser.add_argument("--uniform-variance", action="store_true")
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--estimate-frequency", action="store_true")
    parser.add_argument("--frequency-batches", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the configuration.")
    return parser


def main():
    args = build_parser().parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not {"train", "diffusion", "model"}.issubset(config):
        raise ValueError("Configuration must contain train, diffusion, and model mappings")
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "config": config}, indent=2))
        return
    output_dir = run_experiment(config, args)
    print("Results written to {}".format(output_dir))


if __name__ == "__main__":
    main()
