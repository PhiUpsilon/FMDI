from pathlib import Path

import yaml

from fmdi.cli import build_parser


def test_paper_configurations_are_complete():
    for name in ("physionet.yaml", "air_quality.yaml"):
        config = yaml.safe_load((Path("configs") / name).read_text())
        assert {"train", "diffusion", "model"}.issubset(config)
        assert config["diffusion"]["lambda_freq"] == 0.05
        assert config["diffusion"]["ode_steps"] == 50


def test_cli_accepts_documented_dry_run():
    args = build_parser().parse_args(
        ["--dataset", "physionet", "--config", "configs/physionet.yaml", "--dry-run"]
    )
    assert args.dataset == "physionet"
    assert args.dry_run
