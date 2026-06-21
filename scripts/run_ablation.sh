#!/usr/bin/env bash
set -euo pipefail

fmdi --config configs/ablation.yaml "$@"
