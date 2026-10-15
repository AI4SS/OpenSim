from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FutureAwareLatentSTNet(nn.Module):
    """Agent-aware state transition model with latent and K-step future heads."""

    def __init__(
        self,
        day_dim: int,
        agent_dim: int,
        d_model: int = 192,
        latent_dim: int = 64,
        nhead: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_dropout: float = 0.0,
        max_len: int = 64,
        k_steps: int = 3,
    ) -> None:
        super().__init__()
        self.day_dim = day_dim
        self.agent_dim = agent_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.k_steps = k_steps
        self.input_dropout = nn.Dropout(feature_dropout)

        self.day_proj = nn.Sequential(
            nn.Linear(day_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.agent_proj = nn.Sequential(
            nn.Linear(agent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.day_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.state_head = nn.Linear(latent_dim, 3)

        future_in_dim = d_model + latent_dim + 3
        self.future_daily_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(future_in_dim + k * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 3),
                )
                for k in range(k_steps)
            ]
        )
        self.future_cumulative_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(future_in_dim + k * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 3),
                )
                for k in range(k_steps)
            ]
        )
        self.future_delta_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(future_in_dim + k * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 3),
                )
                for k in range(k_steps)
            ]
        )
        self.future_impact_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(future_in_dim + k * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for k in range(k_steps)
            ]
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        day_features: torch.Tensor,
        agent_features: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        day_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = day_features.shape
        A = agent_features.shape[2]
        device = day_features.device
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}")

        day_h = self.day_proj(self.input_dropout(day_features))
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        day_h = day_h + self.pos_emb(pos)

        key_padding_mask = None
        if day_mask is not None:
            key_padding_mask = day_mask == 0
        day_ctx = self.day_encoder(
            day_h,
            mask=self._causal_mask(T, device),
            src_key_padding_mask=key_padding_mask,
        )

        agent_ctx = self.agent_proj(self.input_dropout(agent_features))
        day_for_agents = day_ctx.unsqueeze(2).expand(B, T, A, self.d_model)
        fused = torch.cat([day_for_agents, agent_ctx], dim=-1)
        state_latent = self.fuse(fused)
        state_dist = F.softmax(self.state_head(state_latent), dim=-1)

        if agent_mask is None:
            agent_mask = torch.ones(B, T, A, device=device, dtype=state_dist.dtype)
        mask = agent_mask.unsqueeze(-1)
        denom = mask.sum(dim=2).clamp_min(1.0)
        daily_pred = (state_dist * mask).sum(dim=2) / denom
        latent_mean = (state_latent * mask).sum(dim=2) / denom

        future_base = torch.cat([day_ctx, latent_mean, daily_pred], dim=-1)
        daily_preds = []
        cumulative_preds = []
        delta_preds = []
        impact_preds = []
        previous_daily = []
        previous_cumulative = []
        for k in range(self.k_steps):
            daily_input = torch.cat([future_base, *previous_daily], dim=-1) if previous_daily else future_base
            cumulative_input = (
                torch.cat([future_base, *previous_cumulative], dim=-1)
                if previous_cumulative
                else future_base
            )
            daily_k = F.softmax(self.future_daily_heads[k](daily_input), dim=-1)
            cumulative_k = F.softmax(self.future_cumulative_heads[k](cumulative_input), dim=-1)
            delta_k = torch.tanh(self.future_delta_heads[k](daily_input))
            impact_k = F.softplus(self.future_impact_heads[k](daily_input)).squeeze(-1)
            daily_preds.append(daily_k)
            cumulative_preds.append(cumulative_k)
            delta_preds.append(delta_k)
            impact_preds.append(impact_k)
            previous_daily.append(daily_k)
            previous_cumulative.append(cumulative_k)

        return {
            "state_dist": state_dist,
            "state_latent": state_latent,
            "daily_pred": daily_pred,
            "future_daily_pred": torch.stack(daily_preds, dim=2),
            "future_cumulative_pred": torch.stack(cumulative_preds, dim=2),
            "future_delta_pred": torch.stack(delta_preds, dim=2),
            "future_impact_pred": torch.stack(impact_preds, dim=2),
        }


def cumulative_from_daily(daily_pred: torch.Tensor, update_weight: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
    """Causal cumulative aggregation from predicted daily distributions."""
    B, T, _ = daily_pred.shape
    running = daily_pred.new_zeros(B, 3)
    total = daily_pred.new_zeros(B, 1)
    out = []
    for t in range(T):
        valid = day_mask[:, t].unsqueeze(-1)
        weight = update_weight[:, t].clamp_min(1.0).unsqueeze(-1) * valid
        running = running * total + daily_pred[:, t] * weight
        total = total + weight
        current = running / total.clamp_min(1.0)
        out.append(current)
        running = current
    return torch.stack(out, dim=1)
