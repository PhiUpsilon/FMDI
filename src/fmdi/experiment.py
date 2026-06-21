"""Core experiment interface.

Keep all paper-specific training, inference, and result serialization logic in
this module so launch scripts remain stable across experiments.
"""

from __future__ import annotations

from typing import Any


def run_experiment(config: dict[str, Any]) -> None:
    """Run one configured FMDI experiment."""
    name = config.get("experiment", {}).get("name", "unnamed")
    raise NotImplementedError(
        f"FMDI experiment '{name}' is not implemented yet. "
        "Add the paper-specific pipeline in src/fmdi/experiment.py."
    )
