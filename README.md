# FMDI

Official implementation repository for **FMDI**.

> This repository is an initial public release scaffold. The package interfaces,
> experiment configuration, and reproducibility entry points are in place; add
> the paper-specific model and dataset adapters under `src/fmdi/` as they are
> finalized.

## Supplementary material

We submit supplemental material through the following public repository:
https://github.com/PhiUpsilon/FMDI

The repository contains the implementation of FMDI, configuration files,
experiment launch scripts, dataset preparation instructions, evaluation
utilities, and analysis scripts needed to reproduce the main comparison,
ablation, sensitivity, and runtime profiling results reported in the paper. It
also includes the processed result summaries used to generate the manuscript
tables and figures.

In addition, the supplementary document is provided as
[`FMDI_SupplementaryMaterials_ICDE.pdf`](FMDI_SupplementaryMaterials_ICDE.pdf)
at: https://github.com/PhiUpsilon/FMDI/blob/main/FMDI_SupplementaryMaterials_ICDE.pdf

The repository is hosted on GitHub, an openly accessible file sharing and
version-control platform with standard privacy policies.

## Repository layout

```text
configs/       Experiment configurations
data/          Dataset preparation notes; raw and processed data stay untracked
scripts/       Training, evaluation, and analysis launchers
src/fmdi/      FMDI implementation and command-line entry point
results/       Generated metrics, tables, and figures (untracked except examples)
tests/         Lightweight regression tests
```

## Quick start

```bash
conda create -n fmdi python=3.10
conda activate fmdi
pip install -e '.[dev]'
fmdi --config configs/main.yaml
```

The starter command validates and loads the experiment configuration. Replace
the `NotImplementedError` in `src/fmdi/experiment.py` with the paper's training
and evaluation pipeline before running the full benchmarks.

## Reproducibility workflow

1. Follow [data/README.md](data/README.md) to acquire and prepare each dataset.
2. Run `scripts/run_main.sh`, `scripts/run_ablation.sh`, and `scripts/run_sensitivity.sh`.
3. Evaluate generated predictions with `scripts/evaluate.sh`.
4. Aggregate outputs and regenerate paper artifacts with `scripts/analyze.sh`.

## Citation

Please replace this placeholder with the final ICDE bibliographic entry when the paper metadata is available.

```bibtex
@inproceedings{fmdi,
  title = {FMDI},
  author = {Anonymous},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering},
  year = {2026}
}
```

## License

The source code in this repository is released under the
[MIT License](LICENSE). Dataset access and use remain subject to each dataset's
own license and terms.
