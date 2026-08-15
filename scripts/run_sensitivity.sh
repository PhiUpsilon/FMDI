#!/usr/bin/env bash
set -euo pipefail

lambda_freq="${1:-0.05}"
shift || true
dataset="${1:-physionet}"
shift || true
case "${dataset}" in
  physionet) config="configs/physionet.yaml" ;;
  air_quality) config="configs/air_quality.yaml" ;;
  *) echo "dataset must be physionet or air_quality" >&2; exit 2 ;;
esac
fmdi --dataset "${dataset}" --config "${config}" --lambda-freq "${lambda_freq}" "$@"
