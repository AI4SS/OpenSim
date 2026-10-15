from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from opensim.policy.critic import (
    FutureTrajectoryCritic,
    compute_trajectory_losses,
    trajectory_kl_per_row,
)
from opensim.policy.objectives import response_objective


def extract_policy_features(
    last_hidden_state: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the final prompt state and mean assistant response state."""

    response_mask = labels.ne(-100)
    if not bool(response_mask.any(dim=1).all()):
        raise ValueError("Every policy sample must contain supervised response tokens.")
    hidden_float = last_hidden_state.float()
    first_response = response_mask.float().argmax(dim=1)
    prompt_index = (first_response - 1).clamp_min(0)
    batch_index = torch.arange(labels.shape[0], device=labels.device)
    prompt_feature = hidden_float[batch_index, prompt_index]
    response_weight = response_mask.unsqueeze(-1).float()
    response_feature = (
        hidden_float * response_weight
    ).sum(dim=1) / response_weight.sum(dim=1).clamp_min(1.0)
    return prompt_feature, response_feature


class FutureConstrainedPolicyModel(nn.Module):
    """Policy LoRA with a response-conditioned future trajectory critic."""

    def __init__(
        self,
        llm: nn.Module,
        llm_hidden_size: int,
        latent_dim: int = 64,
        critic_hidden_size: int = 256,
        k_steps: int = 10,
        critic_dropout: float = 0.2,
        label_smoothing: float = 0.02,
        action_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.label_smoothing = label_smoothing
        self.action_loss_weight = action_loss_weight
        self.register_buffer(
            "action_class_weights",
            torch.ones(2, dtype=torch.float32),
            persistent=False,
        )
        self.use_action_class_weights = False
        self.critic = FutureTrajectoryCritic(
            llm_hidden_size=llm_hidden_size,
            latent_dim=latent_dim,
            critic_hidden_size=critic_hidden_size,
            k_steps=k_steps,
            dropout=critic_dropout,
        )

    def set_lora_trainable(self, enabled: bool) -> None:
        for name, parameter in self.llm.named_parameters():
            parameter.requires_grad = enabled and "lora_" in name

    def set_action_class_weights(
        self,
        weights: torch.Tensor | None,
    ) -> None:
        if weights is None:
            self.action_class_weights.fill_(1.0)
            self.use_action_class_weights = False
            return
        if tuple(weights.shape) != (2,):
            raise ValueError("Action class weights must have shape [2].")
        self.action_class_weights.copy_(
            weights.to(
                device=self.action_class_weights.device,
                dtype=self.action_class_weights.dtype,
            )
        )
        self.use_action_class_weights = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        action_token_mask: torch.Tensor | None,
        action_class: torch.Tensor | None,
        current_state: torch.Tensor,
        st_latent: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        mc_samples: int = 4,
        mc_dropout: bool | None = None,
        detach_llm_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        llm_context = torch.no_grad() if detach_llm_features else nullcontext()
        with llm_context:
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        text_losses = response_objective(
            outputs.logits,
            labels,
            action_token_mask,
            action_class=action_class,
            action_weights=(
                self.action_class_weights
                if self.use_action_class_weights
                else None
            ),
            label_smoothing=self.label_smoothing,
            action_loss_weight=self.action_loss_weight,
        )
        prompt_feature, response_feature = extract_policy_features(
            outputs.hidden_states[-1],
            labels,
        )
        trajectory_samples = self.critic(
            prompt_feature=prompt_feature,
            response_feature=response_feature,
            current_state=current_state,
            st_latent=st_latent,
            mc_samples=mc_samples,
            mc_dropout=mc_dropout,
        )
        future_losses = compute_trajectory_losses(
            trajectory_samples=trajectory_samples,
            target=future_states,
            future_mask=future_mask,
            current_state=current_state,
        )
        return {
            "text_loss": text_losses["total_loss"],
            "response_loss": text_losses["response_loss"],
            "action_loss": text_losses["action_loss"],
            "logits": outputs.logits,
            "prompt_feature": prompt_feature,
            "response_feature": response_feature,
            "trajectory_samples": trajectory_samples,
            **future_losses,
        }

    def response_contrastive_loss(
        self,
        *,
        prompt_feature: torch.Tensor,
        response_feature: torch.Tensor,
        current_state: torch.Tensor,
        st_latent: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        matched_trajectory_kl: torch.Tensor,
        matched_trajectory_samples: torch.Tensor | None = None,
        margin: float = 0.05,
        mc_samples: int = 1,
        mc_dropout: bool = True,
        shuffle_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Make the critic prefer its matched response over masked/shuffled text."""

        masked_samples, shuffled_samples = self._response_condition_samples(
            prompt_feature=prompt_feature,
            response_feature=response_feature,
            current_state=current_state,
            st_latent=st_latent,
            mc_samples=mc_samples,
            mc_dropout=mc_dropout,
            shuffle_index=shuffle_index,
        )
        masked_losses = compute_trajectory_losses(
            trajectory_samples=masked_samples,
            target=future_states,
            future_mask=future_mask,
            current_state=current_state,
        )
        shuffled_losses = compute_trajectory_losses(
            trajectory_samples=shuffled_samples,
            target=future_states,
            future_mask=future_mask,
            current_state=current_state,
        )
        if matched_trajectory_samples is None:
            masked_margin = F.relu(
                float(margin)
                + matched_trajectory_kl
                - masked_losses["trajectory_kl"]
            )
            shuffled_margin = F.relu(
                float(margin)
                + matched_trajectory_kl
                - shuffled_losses["trajectory_kl"]
            )
        else:
            matched_per_row = trajectory_kl_per_row(
                matched_trajectory_samples,
                future_states,
                future_mask,
            )
            masked_per_row = trajectory_kl_per_row(
                masked_samples,
                future_states,
                future_mask,
            )
            shuffled_per_row = trajectory_kl_per_row(
                shuffled_samples,
                future_states,
                future_mask,
            )
            valid_rows = future_mask.sum(dim=-1).gt(0)
            if bool(valid_rows.any()):
                masked_margin = F.relu(
                    float(margin)
                    + matched_per_row[valid_rows]
                    - masked_per_row[valid_rows]
                ).mean()
                shuffled_margin = F.relu(
                    float(margin)
                    + matched_per_row[valid_rows]
                    - shuffled_per_row[valid_rows]
                ).mean()
            else:
                zero = matched_trajectory_kl * 0.0
                masked_margin = zero
                shuffled_margin = zero
        return {
            "contrastive_loss": 0.5 * (masked_margin + shuffled_margin),
            "masked_contrastive_kl": masked_losses["trajectory_kl"],
            "shuffled_contrastive_kl": shuffled_losses["trajectory_kl"],
            "masked_trajectory_kl": masked_losses["trajectory_kl"],
            "shuffled_trajectory_kl": shuffled_losses["trajectory_kl"],
        }

    def _response_condition_samples(
        self,
        *,
        prompt_feature: torch.Tensor,
        response_feature: torch.Tensor,
        current_state: torch.Tensor,
        st_latent: torch.Tensor,
        mc_samples: int,
        mc_dropout: bool,
        shuffle_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if shuffle_index is None:
            shuffled_response = response_feature.roll(1, dims=0)
        else:
            shuffled_response = response_feature.index_select(
                0,
                shuffle_index.to(response_feature.device),
            )
        masked_samples = self.critic(
            prompt_feature=prompt_feature,
            response_feature=torch.zeros_like(response_feature),
            current_state=current_state,
            st_latent=st_latent,
            mc_samples=mc_samples,
            mc_dropout=mc_dropout,
        )
        shuffled_samples = self.critic(
            prompt_feature=prompt_feature,
            response_feature=shuffled_response,
            current_state=current_state,
            st_latent=st_latent,
            mc_samples=mc_samples,
            mc_dropout=mc_dropout,
        )
        return masked_samples, shuffled_samples

    def response_condition_losses(
        self,
        *,
        prompt_feature: torch.Tensor,
        response_feature: torch.Tensor,
        current_state: torch.Tensor,
        st_latent: torch.Tensor,
        future_states: torch.Tensor,
        future_mask: torch.Tensor,
        mc_samples: int = 1,
        mc_dropout: bool = True,
        shuffle_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score feature-level masked and shuffled response conditions."""

        masked_samples, shuffled_samples = self._response_condition_samples(
            prompt_feature=prompt_feature,
            response_feature=response_feature,
            current_state=current_state,
            st_latent=st_latent,
            mc_samples=mc_samples,
            mc_dropout=mc_dropout,
            shuffle_index=shuffle_index,
        )
        masked_losses = compute_trajectory_losses(
            trajectory_samples=masked_samples,
            target=future_states,
            future_mask=future_mask,
            current_state=current_state,
        )
        shuffled_losses = compute_trajectory_losses(
            trajectory_samples=shuffled_samples,
            target=future_states,
            future_mask=future_mask,
            current_state=current_state,
        )
        return {
            "masked_contrastive_kl": masked_losses["trajectory_kl"],
            "shuffled_contrastive_kl": shuffled_losses["trajectory_kl"],
            "masked_trajectory_kl": masked_losses["trajectory_kl"],
            "shuffled_trajectory_kl": shuffled_losses["trajectory_kl"],
        }
