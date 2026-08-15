"""Frequency-domain utilities used by FMDI."""

import logging
import math

import numpy as np
import torch


def rfft_half(x):
    return torch.fft.rfft(x, dim=-1, norm="ortho")


def irfft_half(spectrum, length):
    return torch.fft.irfft(spectrum, n=length, dim=-1, norm="ortho")


def white_complex_half(shape, length, device):
    values = torch.randn(*shape, 2, device=device) / math.sqrt(2.0)
    values = torch.view_as_complex(values)
    values[..., 0] = torch.randn(*shape[:-1], device=device)
    if length % 2 == 0:
        values[..., -1] = torch.randn(*shape[:-1], device=device)
    return values


@torch.no_grad()
def inverse_s2(weights):
    inverse = 1.0 / (weights + 1e-12)
    return inverse / inverse.sum() * weights.numel()


@torch.no_grad()
def estimate_frequency_importance(data_loader, seq_len, device="cpu", max_batches=None, eps=1e-6):
    psd_sum = None
    valid_count = 0
    for batch_no, batch in enumerate(data_loader):
        if max_batches is not None and batch_no >= max_batches:
            break
        data = batch["observed_data"].to(device).float()
        mask = batch["observed_mask"].to(device).float()
        if data.shape[1] == seq_len:
            data, mask = data.permute(0, 2, 1), mask.permute(0, 2, 1)
        elif data.shape[2] != seq_len:
            raise ValueError("seq_len does not match observed_data")
        counts = mask.sum(dim=-1, keepdim=True)
        valid = counts.squeeze(-1) > 0
        if not valid.any():
            continue
        spectrum = rfft_half(data * mask)
        periodogram = spectrum.abs().square() * (float(seq_len) / counts.clamp(min=1.0))
        batch_psd = periodogram[valid].sum(dim=0)
        psd_sum = batch_psd if psd_sum is None else psd_sum + batch_psd
        valid_count += int(valid.sum())
    if psd_sum is None or valid_count == 0:
        raise ValueError("Cannot estimate frequency importance from empty data")
    psd = psd_sum / valid_count
    if psd.numel() > 2:
        padded = torch.cat([psd[:1], psd, psd[-1:]])
        psd = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    importance = psd.clamp(min=eps)
    importance = importance / importance.mean().clamp(min=eps)
    return importance.cpu().numpy().astype(np.float32).tolist()


def prepare_frequency_config(config, data_loader, device="cpu", force=False, max_batches=None):
    diffusion = config["diffusion"]
    if not diffusion.get("freq_noise", False):
        return None
    expected = int(diffusion["seq_len"]) // 2 + 1
    current = diffusion.get("s_vec")
    valid = current is not None and np.asarray(current, dtype=np.float32).size == expected
    if force or not valid:
        current = estimate_frequency_importance(
            data_loader, int(diffusion["seq_len"]), device=device, max_batches=max_batches
        )
        diffusion["s_vec"] = current
        logging.info("estimated frequency importance vector with %d bins", expected)
        return current
    return None
