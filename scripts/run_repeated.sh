#!/usr/bin/env bash
set -u

# Run the public example schedule with the repository's neutral seed.

SEEDS=(0)
RATIOS=(0.1 0.5 0.9)
NSAMPLE="${NSAMPLE:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/repeated_runs}"
failed=0

for seed in "${SEEDS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    fmdi --dataset physionet --config configs/physionet.yaml \
      --missing-ratio "${ratio}" --fold 0 --seed "${seed}" \
      --nsample "${NSAMPLE}" --output-dir "${OUTPUT_DIR}" || failed=1
  done
  fmdi --dataset air_quality --config configs/air_quality.yaml \
    --valid-index 0 --seed "${seed}" --nsample "${NSAMPLE}" \
    --output-dir "${OUTPUT_DIR}" || failed=1
done

exit "${failed}"
