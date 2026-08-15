"""Training and evaluation pipeline used by the public FMDI CLI."""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data_physio import get_dataloader as get_physio_dataloader
from .data_pm25 import get_dataloader as get_pm25_dataloader
from .model import FMDI_PM25, FMDI_Physio
from .spectral import prepare_frequency_config
from .training import evaluate, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: dict[str, Any], args: Any) -> Path:
    """Train or evaluate FMDI and return the output directory."""
    config = copy.deepcopy(config)
    set_seed(args.seed)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint = checkpoint / "model.pth" if checkpoint.is_dir() else checkpoint
        saved_config = checkpoint.parent / "config.json"
        if saved_config.is_file():
            config = json.loads(saved_config.read_text())
    else:
        checkpoint = None

    config["model"]["is_unconditional"] = bool(args.unconditional)
    if args.lambda_freq is not None:
        config["diffusion"]["lambda_freq"] = args.lambda_freq
    if args.ode_steps is not None:
        config["diffusion"]["ode_steps"] = args.ode_steps
    if args.uniform_variance:
        config["diffusion"]["freq_noise"] = False
        config["diffusion"]["freq_shaping_gamma"] = 0.0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output_dir / "train.log", level=logging.INFO, force=True)

    if args.dataset == "physionet":
        config["model"]["test_missing_ratio"] = args.missing_ratio
        train_loader, valid_loader, test_loader = get_physio_dataloader(
            seed=args.seed,
            nfold=args.fold,
            batch_size=config["train"]["batch_size"],
            missing_ratio=args.missing_ratio,
        )
        model_cls = FMDI_Physio
        scaler, mean_scaler = 1, 0
    else:
        train_loader, valid_loader, test_loader, scaler, mean_scaler = get_pm25_dataloader(
            batch_size=config["train"]["batch_size"],
            device=args.device,
            validindex=args.valid_index,
        )
        model_cls = FMDI_PM25

    prepare_frequency_config(
        config,
        train_loader,
        device=args.device,
        force=args.estimate_frequency,
        max_batches=args.frequency_batches,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = model_cls(config, args.device).to(args.device)
    if checkpoint is None:
        train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=str(output_dir))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=str(output_dir),
    )
    return output_dir
