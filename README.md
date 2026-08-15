# FMDI

Official implementation of **FMDI**.

FMDI constructs a Parseval-calibrated frequency-shaped base distribution,
learns a conditional velocity field along a linear RFFT-domain path, and
repairs missing entries with an observation-anchored ODE sampler. This release
contains the model used in the paper, data adapters, paper configurations,
and training/evaluation commands. Trained weights, logs, and numerical result
artifacts are intentionally not distributed in this repository.

## What is included

```text
src/fmdi/       FMDI model, backbone, spectral utilities, data adapters, CLI
configs/        Main and ablation configurations used in the paper
scripts/        Shell entry points for training, evaluation, and aggregation
results/        Destination for locally generated evaluation summaries
data/           Acquisition and expected-layout instructions (raw data excluded)
tests/          Configuration and public-interface smoke tests
```

The supplementary document is available as
[`FMDI_SupplementaryMaterials_ICDE.pdf`](FMDI_SupplementaryMaterials_ICDE.pdf).

## Installation

The experiments were developed with Python 3.8 and PyTorch. Create a clean
environment and install the repository in editable mode:

```bash
conda create -n fmdi python=3.8 -y
conda activate fmdi
pip install -e '.[dev]'
```

Validate the installed entry point without downloading data or using a GPU:

```bash
fmdi --dataset physionet --config configs/physionet.yaml --dry-run
PYTHONPATH=src python tests/smoke.py
```

Both commands exercise the released package and configuration formats.

After installing the optional development dependencies, the lightweight unit
tests can additionally be run with `pytest -q`.

## Data

Raw benchmark data are not redistributed. Follow [data/README.md](data/README.md)
to obtain PhysioNet 2012 and the Air Quality data and place them in the layout
expected by the public data adapters.

## Training and evaluation

PhysioNet, fold 0, 50% missingness, seed 0:

```bash
fmdi \
  --dataset physionet \
  --config configs/physionet.yaml \
  --missing-ratio 0.5 \
  --fold 0 \
  --seed 0 \
  --nsample 100 \
  --output-dir outputs/main
```

Air Quality, validation index 0:

```bash
fmdi \
  --dataset air_quality \
  --config configs/air_quality.yaml \
  --valid-index 0 \
  --seed 0 \
  --nsample 100 \
  --output-dir outputs/main
```

Evaluate a locally trained checkpoint without retraining:

```bash
fmdi \
  --dataset physionet \
  --config configs/physionet.yaml \
  --missing-ratio 0.5 \
  --fold 0 \
  --seed 0 \
  --nsample 100 \
  --checkpoint outputs/main/<run> \
  --output-dir outputs/pretrained
```

The command writes `config.json`, `train.log`, `model.pth` when training, the
100 repair samples, and `result_nsample100.pk`. Inspect a metric file with:

```bash
python -m fmdi.evaluate outputs/pretrained/<run>/result_nsample100.pk
```

## Ablations and sensitivity

Uniform per-frequency-bin base variance while retaining flow matching and the
frequency-domain loss:

```bash
scripts/run_ablation.sh uniform physionet --missing-ratio 0.5 --seed 0 --nsample 100
```

Frequency-shaped flow matching without the frequency-domain loss:

```bash
scripts/run_ablation.sh no_frequency_loss physionet --missing-ratio 0.5 --seed 0 --nsample 100
```

Run one value from the frequency-loss sensitivity sweep:

```bash
scripts/run_sensitivity.sh 0.02 physionet --missing-ratio 0.5 --seed 0 --nsample 100
```

Aggregate result files generated under an output directory:

```bash
scripts/analyze.sh outputs --output results/generated/aggregated_results.csv
```

An example schedule using seed `0` is available as `scripts/run_repeated.sh`.

## Generated artifacts

The repository does not include trained weights, logs, predictions, or
numerical experiment results. Running the commands above creates these files
locally under `outputs/`; see [results/README.md](results/README.md) for the
aggregation interface.

## Reproducibility notes

- Main models use `lambda_freq=0.05` and 50 ODE steps.
- Main evaluation generates 100 repair samples and uses their median for point
  metrics.
- PhysioNet results use fold 0; Air Quality results use validation index 0.
- Public examples use seed `0`; users can override it with `--seed`.
- GPU selection is controlled externally through `CUDA_VISIBLE_DEVICES`.

## Citation

The paper is currently under review. Please use the following temporary entry
until final bibliographic metadata are available:

```bibtex
@inproceedings{fmdi,
  title = {FMDI},
  author = {Anonymous},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering},
  year = {2026}
}
```

## License

The source code is released under the [MIT License](LICENSE). Dataset access
and use remain subject to the respective dataset licenses and terms.
