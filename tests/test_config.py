from pathlib import Path

import yaml


def test_main_configuration_is_a_mapping() -> None:
    path = Path("configs/main.yaml")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict)
    assert config["model"]["name"] == "fmdi"
