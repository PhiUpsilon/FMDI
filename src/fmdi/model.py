import math
import logging

import numpy as np
import torch
import torch.nn as nn

from .backbone import diff_FSDI
from .spectral import inverse_s2, irfft_half, rfft_half, white_complex_half


class FMDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(self.target_dim, self.emb_feature_dim)

        cfg = dict(config["diffusion"])
        cfg["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_FSDI(cfg, input_dim)

        self.num_steps = int(cfg["num_steps"])
        self.ode_steps = int(cfg.get("ode_steps", self.num_steps))
        self.lambda_freq = float(cfg.get("lambda_freq", 0.05))
        self.freq_shaping_gamma = float(cfg.get("freq_shaping_gamma", 1.0))
        self.eps_floor = float(cfg.get("eps_floor", 1e-6))

        self.L = int(cfg["seq_len"])
        self.Lf = self.L // 2 + 1

        w0 = torch.ones(self.Lf, device=device)
        if self.Lf > 1:
            w0[1:self.Lf - 1] = 2.0
        if self.L % 2 == 0:
            w0[-1] = 1.0
        self.register_buffer("w0_parseval", w0)

        def normalize_time_energy(s2):
            scale = self.L / torch.clamp((s2 * self.w0_parseval).sum(), min=1e-12)
            return s2 * scale

        s2_uniform = torch.ones(self.Lf, device=device)
        s2_uniform = normalize_time_energy(s2_uniform)

        s_vec = cfg.get("s_vec", None)
        if bool(cfg.get("freq_noise", False)) and s_vec is not None:
            importance = torch.tensor(np.asarray(s_vec, np.float32).reshape(-1), device=device)
            if importance.numel() != self.Lf:
                raise ValueError(
                    "s_vec length {} does not match seq_len {} half-spectrum length {}".format(
                        importance.numel(), self.L, self.Lf
                    )
                )
            importance = importance / torch.clamp(importance.mean(), min=1e-8)
            s2_wf = normalize_time_energy(inverse_s2(importance))
        else:
            if bool(cfg.get("freq_noise", False)):
                logging.warning("freq_noise is enabled but s_vec is missing; using uniform FM noise")
            importance = torch.ones(self.Lf, device=device)
            s2_wf = s2_uniform.clone()

        gamma = self.freq_shaping_gamma
        s2_fm = (1.0 - gamma) * s2_uniform + gamma * s2_wf
        s2_fm = normalize_time_energy(torch.clamp(s2_fm, min=self.eps_floor))

        self.register_buffer("w_importance", importance)
        self.register_buffer("s2_uniform", s2_uniform)
        self.register_buffer("s2_fm", s2_fm)
        self.register_buffer("shape_std", torch.sqrt(torch.clamp(s2_fm, min=self.eps_floor)))

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand = torch.rand_like(observed_mask) * observed_mask
        rand = rand.reshape(len(rand), -1)
        for i in range(len(observed_mask)):
            ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * ratio)
            rand[i][rand[i].topk(num_masked).indices] = -1
        return (rand > 0).reshape(observed_mask.shape).float()

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy == "mix" and np.random.rand() > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1).permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def flow_time(self, step):
        return (step.float() + 0.5) / float(self.num_steps)

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if L != self.L:
            raise ValueError("Input length {} does not match configured seq_len {}".format(L, self.L))

        if is_train != 1:
            step = torch.ones(B, dtype=torch.long, device=self.device) * set_t
        else:
            step = torch.randint(0, self.num_steps, [B], device=self.device)

        tau = self.flow_time(step).view(B, 1, 1)
        X1 = rfft_half(observed_data)
        Z = white_complex_half(X1.shape, L, self.device)
        shaped_noise = self.shape_std.view(1, 1, self.Lf) * Z

        Xt = tau * X1 + (1.0 - tau) * shaped_noise
        x_t = irfft_half(Xt, L)
        velocity_f = X1 - shaped_noise
        velocity_time = irfft_half(velocity_f, L)

        total_input = self.set_input_to_diffmodel(x_t, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, step)

        target_mask = observed_mask - cond_mask
        denom = target_mask.sum()
        loss_time = (((predicted - velocity_time) * target_mask) ** 2).sum() / (denom if denom > 0 else 1.0)

        pred_f = rfft_half(predicted)
        freq_residual = pred_f - velocity_f
        freq_power = freq_residual.real.square() + freq_residual.imag.square()
        loss_freq = (freq_power * self.w0_parseval.view(1, 1, self.Lf)).sum(dim=-1).mean() / self.L

        return loss_time + self.lambda_freq * loss_freq

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)

    @torch.no_grad()
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        if L != self.L:
            raise ValueError("Input length {} does not match configured seq_len {}".format(L, self.L))

        imputed = torch.zeros(B, n_samples, K, L, device=self.device)
        dt = 1.0 / float(self.ode_steps)

        for i in range(n_samples):
            current_f = self.shape_std.view(1, 1, self.Lf) * white_complex_half((B, K, self.Lf), L, self.device)
            for ode_step in range(self.ode_steps):
                model_step = min(int(ode_step * self.num_steps / self.ode_steps), self.num_steps - 1)
                current = irfft_half(current_f, L)
                diff_input = self.set_input_to_diffmodel(current, observed_data, cond_mask)
                pred_time = self.diffmodel(diff_input, side_info, torch.tensor([model_step], device=self.device))
                current_f = current_f + dt * rfft_half(pred_time)

            current = irfft_half(current_f, L)
            current = cond_mask * observed_data + (1 - cond_mask) * current
            imputed[:, i] = current
        return imputed

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, _ = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        observed_data, observed_mask, observed_tp, gt_mask, _, cut_length = self.process_data(batch)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class FMDI_PM25(FMDI_base):
    def __init__(self, config, device, target_dim=36):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        return (
            observed_data.permute(0, 2, 1),
            observed_mask.permute(0, 2, 1),
            observed_tp,
            gt_mask.permute(0, 2, 1),
            for_pattern_mask.permute(0, 2, 1),
            cut_length,
        )


class FMDI_Physio(FMDI_base):
    def __init__(self, config, device, target_dim=35):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        return (
            observed_data.permute(0, 2, 1),
            observed_mask.permute(0, 2, 1),
            observed_tp,
            gt_mask.permute(0, 2, 1),
            observed_mask.permute(0, 2, 1),
            torch.zeros(len(observed_data), dtype=torch.long, device=self.device),
        )
