#!/usr/bin/env bash
set -euo pipefail

ablation="${1:-uniform}"
shift || true
dataset="${1:-physionet}"
shift || true
case "${dataset}" in
  physionet) config="configs/physionet.yaml" ;;
  air_quality) config="configs/air_quality.yaml" ;;
  *) echo "dataset must be physionet or air_quality" >&2; exit 2 ;;
esac
case "${ablation}" in
  uniform) extra=(--uniform-variance) ;;
  no_frequency_loss) extra=(--lambda-freq 0) ;;
  *) echo "ablation must be uniform or no_frequency_loss" >&2; exit 2 ;;
esac
fmdi --dataset "${dataset}" --config "${config}" "${extra[@]}" "$@"
