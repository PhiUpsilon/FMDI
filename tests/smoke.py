"""Dependency-light public artifact smoke test (no pytest required)."""

from pathlib import Path

import yaml


def main():
    for name in ("physionet.yaml", "air_quality.yaml"):
        config = yaml.safe_load((Path("configs") / name).read_text())
        assert {"train", "diffusion", "model"}.issubset(config)

        assert config["train"]["epochs"] > 0
        print("validated", name)


if __name__ == "__main__":
    main()
