from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureTrajectoryCritic(nn.Module):
    """Predict a response-conditioned K-step ST trajectory."""

    def __init__(
        self,
        llm_hidden_size: int,
        latent_dim: int = 64,
        critic_hidden_size: int = 256,
        k_steps: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.k_steps = k_steps
        self.dropout = dropout
        input_size = 2 * llm_hidden_size + latent_dim + 3
        self.prompt_norm = nn.LayerNorm(llm_hidden_size)
        self.response_norm = nn.LayerNorm(llm_hidden_size)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.fusion_in = nn.Linear(input_size, critic_hidden_size)
        self.fusion_out = nn.Linear(critic_hidden_size, critic_hidden_size)
        self.fusion_norm = nn.LayerNorm(critic_hidden_size)
        self.trajectory_cell = nn.GRUCell(3, critic_hidden_size)
        self.delta_in = nn.Linear(critic_hidden_size, critic_hidden_size)
        self.delta_out = nn.Linear(critic_hidden_size, 3)
        nn.init.normal_(self.delta_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_out.bias)

    def _dropout(self, value: torch.Tensor, enabled: bool) -> torch.Tensor:
        return F.dropout(value, p=self.dropout, training=enabled)

    def _single_trajectory(
        self,
        fused_input: torch.Tensor,
        current_state: torch.Tensor,
        dropout_enabled: bool,
    ) -> torch.Tensor:
        hidden = F.gelu(self.fusion_in(fused_input))
        hidden = self._dropout(hidden, dropout_enabled)
        hidden = self.fusion_norm(self.fusion_out(hidden))
        previous = current_state.clamp_min(1e-6)
        predictions = []
        for _ in range(self.k_steps):
            previous = previous.clamp_min(1e-6)
            previous = previous / previous.sum(dim=-1, keepdim=True)
            hidden = self.trajectory_cell(previous, hidden)
            delta_hidden = F.gelu(self.delta_in(hidden))
            delta_hidden = self._dropout(delta_hidden, dropout_enabled)
            delta_logits = 4.0 * torch.tanh(self.delta_out(delta_hidden) / 4.0)
            previous = F.softmax(previous.log() + delta_logits, dim=-1)
            predictions.append(previous)
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        prompt_feature: torch.Tensor,
        response_feature: torch.Tensor,
        current_state: torch.Tensor,
        st_latent: torch.Tensor,
        mc_samples: int = 4,
        mc_dropout: bool | None = None,
    ) -> torch.Tensor:
        if mc_samples < 1:
            raise ValueError("mc_samples must be at least 1")
        if st_latent.shape[-1] + 3 + 2 * prompt_feature.shape[-1] != self.fusion_in.in_features:
            raise ValueError("Trajectory critic input dimensions do not match its configuration.")
        dropout_enabled = self.training if mc_dropout is None else mc_dropout
        fused_input = torch.cat(
            [
                self.prompt_norm(prompt_feature.float()),
                self.response_norm(response_feature.float()),
                current_state.float(),
                self.latent_norm(st_latent.float()),
            ],
            dim=-1,
        )
        samples = [
            self._single_trajectory(fused_input, current_state.float(), dropout_enabled)
            for _ in range(mc_samples)
        ]
        return torch.stack(samples, dim=0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(0)
    expanded_mask = expanded_mask.expand_as(values)
    return (values * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def trajectory_kl_per_row(
    trajectory_samples: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
    gamma: float = 0.95,
) -> torch.Tensor:
    """Return the horizon-weighted trajectory KL for each batch row."""

    samples = trajectory_samples.float().clamp_min(1e-7)
    target = target.float().clamp_min(1e-7)
    mask = future_mask.float()
    k_steps = samples.shape[2]
    horizon_weight = torch.tensor(
        [gamma**index for index in range(k_steps)],
        device=samples.device,
        dtype=samples.dtype,
    )
    weighted_mask = mask * horizon_weight.unsqueeze(0)
    kl = (
        target.unsqueeze(0)
        * (target.unsqueeze(0).log() - samples.log())
    ).sum(dim=-1).mean(dim=0)
    return (kl * weighted_mask).sum(dim=-1) / weighted_mask.sum(
        dim=-1
    ).clamp_min(1.0)


def compute_trajectory_losses(
    trajectory_samples: torch.Tensor,
    target: torch.Tensor,
    future_mask: torch.Tensor,
    current_state: torch.Tensor,
    gamma: float = 0.95,
    direction_tolerance: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute masked losses for samples shaped [S, B, K, 3]."""

    samples = trajectory_samples.float().clamp_min(1e-7)
    target = target.float().clamp_min(1e-7)
    mask = future_mask.float()
    k_steps = samples.shape[2]
    horizon_weight = torch.tensor(
        [gamma**index for index in range(k_steps)],
        device=samples.device,
        dtype=samples.dtype,
    )
    weighted_mask = mask * horizon_weight.unsqueeze(0)

    kl = (
        target.unsqueeze(0)
        * (target.unsqueeze(0).log() - samples.log())
    ).sum(dim=-1)
    trajectory_kl = _masked_mean(kl, weighted_mask)

    sample_previous = torch.cat(
        [
            current_state.float().unsqueeze(0).unsqueeze(2).expand(samples.shape[0], -1, -1, -1),
            samples[:, :, :-1],
        ],
        dim=2,
    )
    target_previous = torch.cat(
        [current_state.float().unsqueeze(1), target[:, :-1]],
        dim=1,
    )
    sample_delta = samples - sample_previous
    target_delta = target - target_previous
    delta_l1 = _masked_mean(
        (sample_delta - target_delta.unsqueeze(0)).abs().mean(dim=-1),
        mask,
    )

    direction_mask = mask.unsqueeze(-1) * (target_delta.abs() >= direction_tolerance)
    direction_margin = sample_delta * target_delta.sign().unsqueeze(0)
    direction_loss = _masked_mean(F.softplus(-direction_margin * 10.0), direction_mask)

    if k_steps >= 3:
        sample_curvature = samples[:, :, 2:] - 2 * samples[:, :, 1:-1] + samples[:, :, :-2]
        target_curvature = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        curvature_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        curvature_l1 = _masked_mean(
            (sample_curvature - target_curvature.unsqueeze(0)).abs().mean(dim=-1),
            curvature_mask,
        )
    else:
        curvature_l1 = samples.sum() * 0.0

    sample_mean = samples.mean(dim=0)
    mc_consistency = _masked_mean(
        (samples - sample_mean.unsqueeze(0)).pow(2).mean(dim=-1),
        mask,
    )
    future_loss = (
        trajectory_kl
        + 0.35 * delta_l1
        + 0.15 * direction_loss
        + 0.05 * curvature_l1
        + 0.01 * mc_consistency
    )
    return {
        "future_loss": future_loss,
        "trajectory_kl": trajectory_kl,
        "delta_l1": delta_l1,
        "direction_loss": direction_loss,
        "curvature_l1": curvature_l1,
        "mc_consistency": mc_consistency,
        "mc_variance": samples.var(dim=0, unbiased=False).mean(),
    }
