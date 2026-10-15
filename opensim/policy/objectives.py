from __future__ import annotations

import torch
import torch.nn.functional as F


def action_class_weights(
    targets: list[int],
    mode: str = "inverse_sqrt",
) -> torch.Tensor:
    """Return mean-one binary class weights derived from training labels only."""

    if mode not in {"none", "inverse_sqrt", "inverse"}:
        raise ValueError(f"Unsupported action class balance mode: {mode}")
    counts = torch.tensor(
        [
            max(sum(int(target) == label for target in targets), 1)
            for label in (0, 1)
        ],
        dtype=torch.float32,
    )
    if mode == "none":
        weights = torch.ones_like(counts)
    elif mode == "inverse":
        weights = counts.reciprocal()
    else:
        weights = counts.rsqrt()
    sample_weight_mean = (
        weights[0] * counts[0] + weights[1] * counts[1]
    ) / counts.sum().clamp_min(1.0)
    return weights / sample_weight_mean.clamp_min(1e-12)


def response_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_mask: torch.Tensor | None = None,
    action_class: torch.Tensor | None = None,
    action_weights: torch.Tensor | None = None,
    *,
    label_smoothing: float = 0.02,
    action_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return row-balanced response loss plus an explicit action-token loss."""

    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    row_loss = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    response_loss = row_loss.mean()

    if action_token_mask is None:
        action_mask = torch.zeros_like(valid)
    else:
        action_mask = action_token_mask[:, 1:].bool() & valid
    if bool(action_mask.any()):
        action_row_loss = (token_losses * action_mask).sum(dim=1) / action_mask.sum(
            dim=1
        ).clamp_min(1)
        action_rows = action_mask.any(dim=1)
        selected_loss = action_row_loss[action_rows]
        if action_class is not None and action_weights is not None:
            selected_class = action_class.to(
                device=selected_loss.device,
                dtype=torch.long,
            )[action_rows]
            selected_weights = action_weights.to(
                device=selected_loss.device,
                dtype=selected_loss.dtype,
            )[selected_class]
            action_loss = (
                selected_loss * selected_weights
            ).sum() / selected_weights.sum().clamp_min(1e-12)
        else:
            action_loss = selected_loss.mean()
    else:
        action_loss = response_loss * 0.0

    total_loss = response_loss + float(action_loss_weight) * action_loss
    return {
        "total_loss": total_loss,
        "response_loss": response_loss,
        "action_loss": action_loss,
        "row_loss": row_loss,
    }


def action_predictions(
    logits: torch.Tensor,
    decision_indices: torch.Tensor,
    comment_token_id: int,
    share_token_id: int,
) -> torch.Tensor:
    """Classify the first autonomous action token from prompt-only logits."""

    batch_index = torch.arange(logits.shape[0], device=logits.device)
    prediction_positions = (decision_indices.to(logits.device) - 1).clamp_min(0)
    decision_logits = logits[batch_index, prediction_positions]
    candidates = decision_logits[:, [comment_token_id, share_token_id]]
    return candidates.argmax(dim=-1)


def binary_macro_f1(targets: list[int], predictions: list[int]) -> float:
    scores = []
    for label in (0, 1):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
    return sum(scores) / len(scores)
