from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from opensim.policy.dataset import (  # noqa: E402
    PolicyMFDataset,
    load_and_process_csv_data,
)
from opensim.policy.model import FutureConstrainedPolicyModel  # noqa: E402
from opensim.policy.objectives import (  # noqa: E402
    action_class_weights,
    action_predictions,
    binary_macro_f1,
)
from opensim.policy.prompting import (  # noqa: E402
    POLICY_FORMAT_ACTION_CONTENT,
    POLICY_FORMATS,
    normalize_action,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


RESUME_CONFIG_KEYS = (
    "pretrain",
    "init_adapter",
    "fresh_lora",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "dataset",
    "eval_dataset",
    "output_dir",
    "max_len",
    "k_steps",
    "latent_dim",
    "critic_hidden_size",
    "critic_dropout",
    "mc_samples",
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "warmup_epochs",
    "joint_epochs",
    "head_warmup_lr",
    "head_joint_lr",
    "lora_lr",
    "lambda_future",
    "warmup_ratio",
    "label_smoothing",
    "policy_format",
    "action_loss_weight",
    "action_class_balance",
    "min_action_macro_f1",
    "anti_shortcut_weight",
    "anti_shortcut_margin",
    "grad_clip",
    "num_workers",
    "anti_shortcut_samples",
    "max_train_batches",
    "max_eval_batches",
    "comment_nll_limit",
    "seed",
)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(
    state: dict[str, object] | None,
    fallback_seed: int,
) -> bool:
    if not isinstance(state, dict):
        seed_everything(fallback_seed)
        return False
    required = {"python", "numpy", "torch", "cuda"}
    if not required.issubset(state):
        seed_everything(fallback_seed)
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    return True


def _history_identity(history: list[dict]) -> list[tuple[str, int]]:
    return [
        (str(row.get("stage", "")), int(row.get("epoch", -1)))
        for row in history
    ]


def _expected_history_identity(
    stage: str,
    epoch: int,
    warmup_epochs: int,
) -> list[tuple[str, int]]:
    warmup_end = epoch if stage == "warmup" else warmup_epochs
    expected = [("warmup", index) for index in range(1, warmup_end + 1)]
    if stage == "joint":
        expected.extend(("joint", index) for index in range(1, epoch + 1))
    return expected


def load_resume_training_state(
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    required_files = (
        checkpoint_path / "adapter_config.json",
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "trajectory_critic.pt",
        checkpoint_path / "trainer_state.pt",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Future resume checkpoint is incomplete; missing: {missing}"
        )
    state = torch.load(
        checkpoint_path / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    required_state_keys = {
        "stage",
        "epoch",
        "global_step",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
        "history",
    }
    missing_state = sorted(required_state_keys.difference(state))
    if missing_state:
        raise ValueError(
            f"Future resume trainer_state is missing keys: {missing_state}"
        )

    saved_config = state["config"]
    mismatches = {}
    for key in RESUME_CONFIG_KEYS:
        saved = saved_config.get(key)
        current = getattr(args, key)
        if saved != current:
            mismatches[key] = {"checkpoint": saved, "current": current}
    if mismatches:
        raise ValueError(
            "Future resume configuration mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    stage = str(state["stage"])
    epoch = int(state["epoch"])
    stage_epochs = {
        "warmup": int(args.warmup_epochs),
        "joint": int(args.joint_epochs),
    }
    if stage not in stage_epochs:
        raise ValueError(f"Invalid future resume stage: {stage!r}.")
    if epoch < 1 or epoch > stage_epochs[stage]:
        raise ValueError(
            f"Invalid completed {stage} epoch {epoch}/{stage_epochs[stage]}."
        )
    expected_checkpoint_name = f"{stage}-epoch-{epoch}"
    if checkpoint_path.name != expected_checkpoint_name:
        raise ValueError(
            "Future resume checkpoint directory disagrees with trainer state: "
            f"{checkpoint_path.name!r} != {expected_checkpoint_name!r}."
        )

    state_history = state["history"]
    if not isinstance(state_history, list) or not state_history:
        raise ValueError("Future resume trainer_state has no epoch history.")
    expected_history = _expected_history_identity(
        stage,
        epoch,
        int(args.warmup_epochs),
    )
    state_identity = _history_identity(state_history)
    if state_identity != expected_history:
        raise ValueError(
            "Future resume history must contain each completed stage/epoch "
            f"exactly once; got {state_identity}, expected {expected_history}."
        )

    summary_path = output_dir / "train_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Future resume requires the existing training summary: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") == "completed":
        raise ValueError("Refusing to resume a completed future run.")
    summary_history = summary.get("history")
    if not isinstance(summary_history, list):
        raise ValueError("Future resume summary history is not a list.")
    summary_identity = _history_identity(summary_history)
    if summary_identity not in (state_identity, state_identity[:-1]):
        raise ValueError(
            "Future resume summary must match the checkpoint history or lag "
            f"only its final atomic write; got {summary_identity}, "
            f"expected {state_identity} or {state_identity[:-1]}."
        )
    if not state["optimizer_state_dict"].get("param_groups"):
        raise ValueError("Future resume checkpoint has no optimizer groups.")
    if "last_epoch" not in state["scheduler_state_dict"]:
        raise ValueError("Future resume checkpoint has no scheduler last_epoch.")

    warmup_start_epoch = int(args.warmup_epochs) + 1
    joint_start_epoch = 1
    resume_stage = None
    if stage == "warmup":
        warmup_start_epoch = epoch + 1
        if warmup_start_epoch <= int(args.warmup_epochs):
            resume_stage = "warmup"
    else:
        joint_start_epoch = epoch + 1
        if joint_start_epoch <= int(args.joint_epochs):
            resume_stage = "joint"
    fallback_seed = int(args.seed) + len(state_history)
    audit = {
        "checkpoint": str(checkpoint_path),
        "from_stage": stage,
        "from_epoch": epoch,
        "from_global_step": int(state["global_step"]),
        "warmup_start_epoch": warmup_start_epoch,
        "joint_start_epoch": joint_start_epoch,
        "optimizer_state_restored": resume_stage is not None,
        "scheduler_state_restored": resume_stage is not None,
        "rng_state_available": isinstance(state.get("rng_state"), dict),
        "train_generator_state_available": isinstance(
            state.get("train_generator_state"),
            torch.Tensor,
        ),
        "valid_generator_state_available": isinstance(
            state.get("valid_generator_state"),
            torch.Tensor,
        ),
        "deterministic_fallback_seed": fallback_seed,
        "summary_lag_repaired": summary_identity == state_identity[:-1],
    }
    return {
        "state": state,
        "history": state_history,
        "warmup_start_epoch": warmup_start_epoch,
        "joint_start_epoch": joint_start_epoch,
        "resume_stage": resume_stage,
        "fallback_seed": fallback_seed,
        "audit": audit,
    }


def collate_policy(batch: list[dict]) -> dict:
    max_length = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)
    pad_token_id = int(batch[0]["pad_token_id"])
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    action_token_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[index, :length] = item["input_ids"]
        attention_mask[index, :length] = item["attention_mask"]
        labels[index, :length] = item["labels"]
        action_token_mask[index, :length] = item["action_token_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "action_token_mask": action_token_mask,
        "action_decision_index": torch.tensor(
            [item["action_decision_index"] for item in batch],
            dtype=torch.long,
        ),
        "action_class": torch.tensor(
            [item["action_class"] for item in batch],
            dtype=torch.long,
        ),
        "current_state": torch.stack([item["current_state"] for item in batch]),
        "future_states": torch.stack([item["future_states"] for item in batch]),
        "future_mask": torch.stack([item["future_mask"] for item in batch]),
        "st_latent": torch.stack([item["st_latent"] for item in batch]),
        "action_type": [item["action_type"] for item in batch],
        "shuffle_group": [
            f"{item['event_id']}::{item['day_index']}"
            for item in batch
        ],
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def future_weight(global_step: int, ramp_steps: int, maximum: float) -> float:
    if ramp_steps <= 0:
        return maximum
    return maximum * min(1.0, global_step / ramp_steps)


def cross_group_shuffle_index(
    groups: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Pair each row with the nearest response from another event/day."""

    size = len(groups)
    if size < 2:
        return torch.arange(size, device=device)
    indices = []
    for target_index, target_group in enumerate(groups):
        source_index = None
        for offset in range(1, size):
            candidate = (target_index + offset) % size
            if groups[candidate] != target_group:
                source_index = candidate
                break
        if source_index is None:
            source_index = (target_index + 1) % size
        indices.append(source_index)
    return torch.tensor(indices, device=device, dtype=torch.long)


def raw_token_stats(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        return 0.0, 0
    losses = F.cross_entropy(
        shift_logits[valid],
        shift_labels[valid],
        reduction="sum",
    )
    return float(losses.detach().cpu()), int(valid.sum().detach().cpu())


@torch.no_grad()
def evaluate(
    model: FutureConstrainedPolicyModel,
    dataloader: DataLoader,
    device: torch.device,
    mc_samples: int,
    anti_shortcut_samples: int,
    max_eval_batches: int,
    comment_token_id: int,
    share_token_id: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "matched_trajectory_kl": 0.0,
        "matched_delta_l1": 0.0,
        "matched_direction_loss": 0.0,
        "matched_mc_variance": 0.0,
        "masked_trajectory_kl": 0.0,
        "shuffled_trajectory_kl": 0.0,
        "anti_matched_trajectory_kl": 0.0,
    }
    matched_batches = 0
    anti_rows = 0
    anti_batches = 0
    token_loss_sum = 0.0
    token_count = 0
    comment_loss_sum = 0.0
    comment_token_count = 0
    action_loss_sum = 0.0
    action_token_count = 0
    action_targets: list[int] = []
    action_predictions_all: list[int] = []

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Validate", leave=False)):
        if max_eval_batches > 0 and batch_index >= max_eval_batches:
            break
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            action_token_mask=batch["action_token_mask"],
            action_class=batch["action_class"],
            current_state=batch["current_state"],
            st_latent=batch["st_latent"],
            future_states=batch["future_states"],
            future_mask=batch["future_mask"],
            mc_samples=mc_samples,
            mc_dropout=True,
        )
        for key in (
            "trajectory_kl",
            "delta_l1",
            "direction_loss",
            "mc_variance",
        ):
            totals[f"matched_{key}"] += float(outputs[key].detach().cpu())
        matched_batches += 1

        loss_sum, count = raw_token_stats(outputs["logits"], batch["labels"])
        token_loss_sum += loss_sum
        token_count += count
        for index, action in enumerate(batch["action_type"]):
            if action != "评论":
                continue
            row_loss, row_count = raw_token_stats(
                outputs["logits"][index : index + 1],
                batch["labels"][index : index + 1],
            )
            comment_loss_sum += row_loss
            comment_token_count += row_count
        shifted_action_mask = batch["action_token_mask"][:, 1:].bool()
        shifted_labels = batch["labels"][:, 1:]
        shifted_logits = outputs["logits"][:, :-1].contiguous().float()
        action_losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        action_loss_sum += float(
            (action_losses * shifted_action_mask).sum().detach().cpu()
        )
        action_token_count += int(shifted_action_mask.sum().detach().cpu())
        valid_action_rows = batch["action_decision_index"].ge(0)
        if bool(valid_action_rows.any()):
            predicted = action_predictions(
                outputs["logits"],
                batch["action_decision_index"],
                comment_token_id,
                share_token_id,
            )
            action_targets.extend(
                batch["action_class"][valid_action_rows].detach().cpu().tolist()
            )
            action_predictions_all.extend(
                predicted[valid_action_rows].detach().cpu().tolist()
            )

        if anti_rows >= anti_shortcut_samples:
            continue
        totals["anti_matched_trajectory_kl"] += float(
            outputs["trajectory_kl"].detach().cpu()
        )
        condition_losses = model.response_condition_losses(
            prompt_feature=outputs["prompt_feature"],
            response_feature=outputs["response_feature"],
            current_state=batch["current_state"],
            st_latent=batch["st_latent"],
            future_states=batch["future_states"],
            future_mask=batch["future_mask"],
            mc_samples=mc_samples,
            mc_dropout=True,
            shuffle_index=cross_group_shuffle_index(
                batch["shuffle_group"],
                device,
            ),
        )
        totals["masked_trajectory_kl"] += float(
            condition_losses["masked_trajectory_kl"].detach().cpu()
        )
        totals["shuffled_trajectory_kl"] += float(
            condition_losses["shuffled_trajectory_kl"].detach().cpu()
        )
        anti_rows += batch["input_ids"].shape[0]
        anti_batches += 1

    metrics = {
        key: value / max(
            anti_batches
            if key.startswith(("masked_", "shuffled_", "anti_matched_"))
            else matched_batches,
            1,
        )
        for key, value in totals.items()
    }
    metrics["overall_nll"] = token_loss_sum / max(token_count, 1)
    metrics["comment_nll"] = comment_loss_sum / max(comment_token_count, 1)
    metrics["comment_ppl"] = math.exp(min(metrics["comment_nll"], 20.0))
    metrics["action_nll"] = action_loss_sum / max(action_token_count, 1)
    metrics["action_macro_f1"] = (
        binary_macro_f1(action_targets, action_predictions_all)
        if action_targets
        else 0.0
    )
    metrics["predicted_share_ratio"] = (
        sum(action_predictions_all) / len(action_predictions_all)
        if action_predictions_all
        else 0.0
    )
    metrics["target_share_ratio"] = (
        sum(action_targets) / len(action_targets)
        if action_targets
        else 0.0
    )
    metrics["masked_gap"] = (
        metrics["masked_trajectory_kl"] - metrics["anti_matched_trajectory_kl"]
    ) / max(metrics["anti_matched_trajectory_kl"], 1e-8)
    metrics["shuffled_gap"] = (
        metrics["shuffled_trajectory_kl"] - metrics["anti_matched_trajectory_kl"]
    ) / max(metrics["anti_matched_trajectory_kl"], 1e-8)
    metrics["text_sensitive"] = float(
        metrics["masked_gap"] >= 0.05 and metrics["shuffled_gap"] >= 0.05
    )
    model.train()
    return metrics


def save_checkpoint(
    model: FutureConstrainedPolicyModel,
    tokenizer,
    output_dir: Path,
    name: str,
    state: dict,
    sidecars: list[Path] | None = None,
) -> Path:
    checkpoint_dir = output_dir / name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    model.llm.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(model.critic.state_dict(), checkpoint_dir / "trajectory_critic.pt")
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    for sidecar in sidecars or []:
        if sidecar.exists():
            shutil.copy2(sidecar, checkpoint_dir / sidecar.name)
    return checkpoint_dir


def append_training_result(output_dir: Path, result: dict) -> None:
    summary_path = output_dir / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"status": "running", "history": []}
    summary["status"] = "running"
    summary.setdefault("history", []).append(result)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def selection_score(result: dict) -> float:
    return (
        float(result["matched_trajectory_kl"])
        + 0.25 * float(result["matched_delta_l1"])
        + 0.10 * float(result["matched_direction_loss"])
    )


def checkpoint_is_eligible(
    result: dict,
    comment_nll_limit: float,
    min_action_macro_f1: float,
) -> bool:
    return (
        result["stage"] == "warmup"
        or (
            float(result["comment_nll"]) <= comment_nll_limit
            and bool(result["text_sensitive"])
            and float(result["action_macro_f1"]) >= min_action_macro_f1
        )
    )


def _copy_checkpoint_if_needed(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            f"Cannot rebuild missing best checkpoint from {source}."
        )
    current_identity = None
    target_state_path = target / "trainer_state.pt"
    if target_state_path.is_file():
        target_state = torch.load(
            target_state_path,
            map_location="cpu",
            weights_only=False,
        )
        current_identity = (
            str(target_state.get("stage", "")),
            int(target_state.get("epoch", -1)),
        )
    source_state = torch.load(
        source / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_identity = (
        str(source_state.get("stage", "")),
        int(source_state.get("epoch", -1)),
    )
    if current_identity == source_identity:
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def rebuild_best_checkpoints(
    output_dir: Path,
    history: list[dict],
    comment_nll_limit: float,
    min_action_macro_f1: float,
) -> None:
    for stage in ("warmup", "joint"):
        stage_history = [
            row for row in history
            if row.get("stage") == stage
        ]
        if not stage_history:
            continue
        best = min(stage_history, key=selection_score)
        _copy_checkpoint_if_needed(
            output_dir / f"{stage}-epoch-{int(best['epoch'])}",
            output_dir / f"{stage}_best_trajectory_checkpoint",
        )
        eligible = [
            row
            for row in stage_history
            if checkpoint_is_eligible(
                row,
                comment_nll_limit,
                min_action_macro_f1,
            )
        ]
        if not eligible:
            continue
        best_eligible = min(eligible, key=selection_score)
        _copy_checkpoint_if_needed(
            output_dir / f"{stage}-epoch-{int(best_eligible['epoch'])}",
            output_dir / (
                "warmup_best_checkpoint"
                if stage == "warmup"
                else "best_checkpoint"
            ),
        )


def train_stage(
    stage_name: str,
    model: FutureConstrainedPolicyModel,
    tokenizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    gradient_accumulation_steps: int,
    head_lr: float,
    lora_lr: float,
    max_future_weight: float,
    mc_samples: int,
    grad_clip: float,
    warmup_ratio: float,
    anti_shortcut_samples: int,
    max_train_batches: int,
    max_eval_batches: int,
    comment_nll_limit: float,
    min_action_macro_f1: float,
    anti_shortcut_weight: float,
    anti_shortcut_margin: float,
    comment_token_id: int,
    share_token_id: int,
    sidecars: list[Path],
    run_history_prefix: list[dict],
    resume_config: dict,
    train_generator: torch.Generator,
    valid_generator: torch.Generator,
    start_epoch: int = 1,
    resume_state: dict | None = None,
) -> list[dict]:
    joint = stage_name == "joint"
    model.set_lora_trainable(joint)
    parameter_groups = [
        {
            "params": [parameter for parameter in model.critic.parameters() if parameter.requires_grad],
            "lr": head_lr,
        }
    ]
    if joint:
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.llm.parameters()
                    if parameter.requires_grad
                ],
                "lr": lora_lr,
            }
        )
    optimizer = AdamW(parameter_groups, betas=(0.9, 0.95), weight_decay=0.01)
    stage_batches = (
        min(len(train_loader), max_train_batches)
        if max_train_batches > 0
        else len(train_loader)
    )
    updates_per_epoch = math.ceil(stage_batches / gradient_accumulation_steps)
    total_updates = max(updates_per_epoch * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * warmup_ratio),
        num_training_steps=total_updates,
    )
    ramp_steps = max(int(total_updates * 0.2), 1)
    history = []
    prior_stage_history = [
        row for row in run_history_prefix
        if row.get("stage") == stage_name
    ]
    best_score = min(
        (selection_score(row) for row in prior_stage_history),
        default=float("inf"),
    )
    best_eligible_score = min(
        (
            selection_score(row)
            for row in prior_stage_history
            if checkpoint_is_eligible(
                row,
                comment_nll_limit,
                min_action_macro_f1,
            )
        ),
        default=float("inf"),
    )
    nonfinite_steps = 0
    global_step = 0
    if resume_state is not None:
        if str(resume_state["stage"]) != stage_name:
            raise ValueError(
                f"Cannot restore {resume_state['stage']} optimizer in "
                f"{stage_name} stage."
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        global_step = int(resume_state["global_step"])
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if not joint:
            model.llm.eval()
            model.critic.train()
        running = {
            "text": 0.0,
            "response": 0.0,
            "action": 0.0,
            "future": 0.0,
            "contrastive": 0.0,
            "total": 0.0,
        }
        seen_batches = 0
        progress = tqdm(train_loader, desc=f"{stage_name} epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress):
            if max_train_batches > 0 and batch_index >= max_train_batches:
                break
            batch = move_batch(batch, device)
            if not joint and float(batch["future_mask"].sum()) == 0.0:
                continue
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                action_token_mask=batch["action_token_mask"],
                action_class=batch["action_class"],
                current_state=batch["current_state"],
                st_latent=batch["st_latent"],
                future_states=batch["future_states"],
                future_mask=batch["future_mask"],
                mc_samples=mc_samples,
                mc_dropout=True,
                detach_llm_features=not joint,
            )
            if anti_shortcut_weight > 0.0:
                contrastive = model.response_contrastive_loss(
                    prompt_feature=outputs["prompt_feature"],
                    response_feature=outputs["response_feature"],
                    current_state=batch["current_state"],
                    st_latent=batch["st_latent"],
                    future_states=batch["future_states"],
                    future_mask=batch["future_mask"],
                    matched_trajectory_kl=outputs["trajectory_kl"],
                    matched_trajectory_samples=outputs["trajectory_samples"],
                    margin=anti_shortcut_margin,
                    mc_samples=mc_samples,
                    mc_dropout=True,
                    shuffle_index=cross_group_shuffle_index(
                        batch["shuffle_group"],
                        device,
                    ),
                )
            else:
                zero = outputs["future_loss"] * 0.0
                contrastive = {
                    "contrastive_loss": zero,
                    "masked_contrastive_kl": zero,
                    "shuffled_contrastive_kl": zero,
                }
            constrained_future_loss = (
                outputs["future_loss"]
                + anti_shortcut_weight * contrastive["contrastive_loss"]
            )
            weight = (
                future_weight(global_step, ramp_steps, max_future_weight)
                if joint
                else 1.0
            )
            total_loss = (
                outputs["text_loss"] + weight * constrained_future_loss
                if joint
                else constrained_future_loss
            )
            if not bool(torch.isfinite(total_loss)):
                components = {
                    key: float(outputs[key].detach())
                    for key in (
                        "trajectory_kl",
                        "delta_l1",
                        "direction_loss",
                        "curvature_l1",
                        "mc_consistency",
                        "mc_variance",
                    )
                }
                torch.save(
                    {
                        "stage": stage_name,
                        "batch_index": batch_index,
                        "critic_state_dict": model.critic.state_dict(),
                        "prompt_feature": outputs["prompt_feature"].detach().cpu(),
                        "response_feature": outputs["response_feature"].detach().cpu(),
                        "trajectory_samples": outputs["trajectory_samples"].detach().cpu(),
                        "current_state": batch["current_state"].detach().cpu(),
                        "st_latent": batch["st_latent"].detach().cpu(),
                        "future_states": batch["future_states"].detach().cpu(),
                        "future_mask": batch["future_mask"].detach().cpu(),
                        "components": components,
                    },
                    output_dir / f"nonfinite_{stage_name}_batch_{batch_index}.pt",
                )
                raise RuntimeError(
                    f"Non-finite {stage_name} loss at batch {batch_index}: "
                    f"text={float(outputs['text_loss'].detach())}, "
                    f"future={float(outputs['future_loss'].detach())}, "
                    f"components={components}"
                )
            (total_loss / gradient_accumulation_steps).backward()
            seen_batches += 1
            running["text"] += float(outputs["text_loss"].detach().cpu())
            running["response"] += float(outputs["response_loss"].detach().cpu())
            running["action"] += float(outputs["action_loss"].detach().cpu())
            running["future"] += float(outputs["future_loss"].detach().cpu())
            running["contrastive"] += float(
                contrastive["contrastive_loss"].detach().cpu()
            )
            running["total"] += float(total_loss.detach().cpu())

            should_step = seen_batches % gradient_accumulation_steps == 0
            if should_step:
                trainable = [
                    parameter
                    for group in parameter_groups
                    for parameter in group["params"]
                ]
                try:
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        grad_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError:
                    nonfinite_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"[NONFINITE_GRAD] stage={stage_name} "
                        f"batch={batch_index} skipped={nonfinite_steps}",
                        flush=True,
                    )
                    if nonfinite_steps >= 3:
                        raise
                    continue
                optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in trainable):
                    raise RuntimeError(
                        f"Optimizer produced non-finite parameters in {stage_name}."
                    )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(
                total=f"{total_loss.item():.3f}",
                future=f"{outputs['future_loss'].item():.3f}",
                contrast=f"{contrastive['contrastive_loss'].item():.3f}",
                fw=f"{weight:.3f}",
            )

        if seen_batches % gradient_accumulation_steps:
            trainable = [
                parameter
                for group in parameter_groups
                for parameter in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(
                trainable,
                grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in trainable):
                raise RuntimeError(
                    f"Optimizer produced non-finite parameters in {stage_name}."
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        metrics = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            mc_samples=mc_samples,
            anti_shortcut_samples=anti_shortcut_samples,
            max_eval_batches=max_eval_batches,
            comment_token_id=comment_token_id,
            share_token_id=share_token_id,
        )
        result = {
            "stage": stage_name,
            "epoch": epoch,
            "global_step": global_step,
            "train_text_loss": running["text"] / max(seen_batches, 1),
            "train_response_loss": running["response"] / max(seen_batches, 1),
            "train_action_loss": running["action"] / max(seen_batches, 1),
            "train_future_loss": running["future"] / max(seen_batches, 1),
            "train_contrastive_loss": running["contrastive"] / max(seen_batches, 1),
            "train_total_loss": running["total"] / max(seen_batches, 1),
            **metrics,
        }
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        checkpoint_dir = save_checkpoint(
            model,
            tokenizer,
            output_dir,
            f"{stage_name}-epoch-{epoch}",
            {
                "stage": stage_name,
                "epoch": epoch,
                "global_step": global_step,
                "metrics": metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": resume_config,
                "history": run_history_prefix + history,
                "train_generator_state": train_generator.get_state(),
                "valid_generator_state": valid_generator.get_state(),
                "rng_state": capture_rng_state(),
            },
            sidecars=sidecars,
        )
        current_selection_score = selection_score(result)
        if current_selection_score < best_score:
            best_score = current_selection_score
            best_dir = output_dir / f"{stage_name}_best_trajectory_checkpoint"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
        eligible = checkpoint_is_eligible(
            result,
            comment_nll_limit,
            min_action_macro_f1,
        )
        if eligible and current_selection_score < best_eligible_score:
            best_eligible_score = current_selection_score
            best_dir = output_dir / (
                "warmup_best_checkpoint"
                if stage_name == "warmup"
                else "best_checkpoint"
            )
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
        append_training_result(output_dir, result)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--init_adapter", default="")
    parser.add_argument("--fresh_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_len", type=int, default=704)
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--critic_hidden_size", type=int, default=256)
    parser.add_argument("--critic_dropout", type=float, default=0.2)
    parser.add_argument("--mc_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--joint_epochs", type=int, default=2)
    parser.add_argument("--head_warmup_lr", type=float, default=1e-4)
    parser.add_argument("--head_joint_lr", type=float, default=5e-5)
    parser.add_argument("--lora_lr", type=float, default=5e-6)
    parser.add_argument("--lambda_future", type=float, default=0.15)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument(
        "--policy_format",
        choices=sorted(POLICY_FORMATS),
        default=POLICY_FORMAT_ACTION_CONTENT,
    )
    parser.add_argument("--action_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--action_class_balance",
        choices=["none", "inverse_sqrt", "inverse"],
        default="inverse_sqrt",
    )
    parser.add_argument("--min_action_macro_f1", type=float, default=0.0)
    parser.add_argument("--anti_shortcut_weight", type=float, default=0.0)
    parser.add_argument("--anti_shortcut_margin", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--anti_shortcut_samples", type=int, default=512)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--comment_nll_limit", type=float, default=3.7948)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = (
        load_resume_training_state(
            Path(args.resume_checkpoint),
            output_dir,
            args,
        )
        if args.resume_checkpoint
        else None
    )
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer_source = (
        args.resume_checkpoint if args.resume_checkpoint else args.pretrain
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    )
    if resume is not None:
        llm = PeftModel.from_pretrained(
            base_model,
            args.resume_checkpoint,
            is_trainable=True,
        )
    elif args.fresh_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        llm = get_peft_model(base_model, lora_config)
    else:
        if not args.init_adapter:
            raise ValueError("--init_adapter is required unless --fresh_lora is set")
        llm = PeftModel.from_pretrained(
            base_model,
            args.init_adapter,
            is_trainable=True,
        )
    llm.config.use_cache = False
    if device.type == "cuda":
        llm.gradient_checkpointing_enable()
        llm.enable_input_require_grads()
    model = FutureConstrainedPolicyModel(
        llm=llm,
        llm_hidden_size=int(base_model.config.hidden_size),
        latent_dim=args.latent_dim,
        critic_hidden_size=args.critic_hidden_size,
        k_steps=args.k_steps,
        critic_dropout=args.critic_dropout,
        label_smoothing=args.label_smoothing,
        action_loss_weight=args.action_loss_weight,
    ).to(device)
    if resume is not None:
        critic_state = torch.load(
            Path(args.resume_checkpoint) / "trajectory_critic.pt",
            map_location=device,
            weights_only=True,
        )
        model.critic.load_state_dict(critic_state, strict=True)

    train_records = load_and_process_csv_data(args.dataset)
    valid_records = load_and_process_csv_data(args.eval_dataset)
    train_action_targets = [
        1 if normalize_action(record.get("action_type", "")) == "share" else 0
        for record in train_records
    ]
    action_weights = action_class_weights(
        train_action_targets,
        mode=args.action_class_balance,
    )
    model.set_action_class_weights(action_weights)
    train_dataset = PolicyMFDataset(
        train_records,
        tokenizer,
        max_length=args.max_len,
        k_steps=args.k_steps,
        policy_format=args.policy_format,
    )
    valid_dataset = PolicyMFDataset(
        valid_records,
        tokenizer,
        max_length=args.max_len,
        k_steps=args.k_steps,
        policy_format=args.policy_format,
    )
    train_generator = torch.Generator()
    valid_generator = torch.Generator()
    if resume is not None and isinstance(
        resume["state"].get("train_generator_state"),
        torch.Tensor,
    ):
        train_generator.set_state(
            resume["state"]["train_generator_state"]
        )
        train_generator_restored = True
    else:
        train_generator.manual_seed(
            int(resume["fallback_seed"]) if resume is not None else args.seed
        )
        train_generator_restored = False
    if resume is not None and isinstance(
        resume["state"].get("valid_generator_state"),
        torch.Tensor,
    ):
        valid_generator.set_state(
            resume["state"]["valid_generator_state"]
        )
        valid_generator_restored = True
    else:
        valid_generator.manual_seed(
            (
                int(resume["fallback_seed"])
                if resume is not None
                else args.seed
            )
            + 1
        )
        valid_generator_restored = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_policy,
        generator=train_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_policy,
        generator=valid_generator,
    )
    resume_audit = None
    if resume is not None:
        rng_restored = restore_rng_state(
            resume["state"].get("rng_state"),
            int(resume["fallback_seed"]),
        )
        resume_audit = dict(resume["audit"])
        resume_audit.update(
            {
                "rng_state_restored": rng_restored,
                "train_generator_state_restored": train_generator_restored,
                "valid_generator_state_restored": valid_generator_restored,
            }
        )
        print(json.dumps({"resume": resume_audit}, ensure_ascii=False))
    config = vars(args) | {
        "device": str(device),
        "dtype": str(dtype),
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "state_order": "pos,neu,neg",
        "action_class_weights": action_weights.tolist(),
        "resume": resume_audit,
    }
    resume_config = {
        key: getattr(args, key)
        for key in RESUME_CONFIG_KEYS
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    existing_history = (
        list(resume["history"])
        if resume is not None
        else []
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {"status": "running", "history": existing_history},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comment_action_ids = tokenizer.encode("评论", add_special_tokens=False)
    share_action_ids = tokenizer.encode("转发", add_special_tokens=False)
    if not comment_action_ids or not share_action_ids:
        raise ValueError("Action labels must tokenize to at least one token.")
    if comment_action_ids[0] == share_action_ids[0]:
        raise ValueError("Comment and share must have distinct first action tokens.")
    comment_token_id = int(comment_action_ids[0])
    share_token_id = int(share_action_ids[0])
    sidecars = [
        Path(args.dataset).parent / "latent_kmeans.joblib",
        Path(args.init_adapter) / "policy_format.json"
        if args.init_adapter
        else output_dir / "policy_format.json",
    ]
    if not sidecars[-1].exists():
        format_path = output_dir / "policy_format.json"
        format_path.write_text(
            json.dumps(
                {
                    "policy_format": args.policy_format,
                    "action_loss_weight": args.action_loss_weight,
                    "action_class_balance": args.action_class_balance,
                    "action_class_weights": action_weights.tolist(),
                    "comment_token_id": comment_token_id,
                    "share_token_id": share_token_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        sidecars[-1] = format_path
    critic_config_path = output_dir / "trajectory_critic_config.json"
    critic_config_path.write_text(
        json.dumps(
            {
                "llm_hidden_size": int(base_model.config.hidden_size),
                "latent_dim": args.latent_dim,
                "critic_hidden_size": args.critic_hidden_size,
                "k_steps": args.k_steps,
                "dropout": args.critic_dropout,
                "mc_samples": args.mc_samples,
                "anti_shortcut_weight": args.anti_shortcut_weight,
                "anti_shortcut_margin": args.anti_shortcut_margin,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sidecars.append(critic_config_path)
    if resume is not None:
        rebuild_best_checkpoints(
            output_dir,
            existing_history,
            args.comment_nll_limit,
            args.min_action_macro_f1,
        )
    history = list(existing_history)
    warmup_history = train_stage(
        stage_name="warmup",
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
        epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        head_lr=args.head_warmup_lr,
        lora_lr=0.0,
        max_future_weight=1.0,
        mc_samples=args.mc_samples,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        anti_shortcut_samples=args.anti_shortcut_samples,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        comment_nll_limit=args.comment_nll_limit,
        min_action_macro_f1=args.min_action_macro_f1,
        anti_shortcut_weight=args.anti_shortcut_weight,
        anti_shortcut_margin=args.anti_shortcut_margin,
        comment_token_id=comment_token_id,
        share_token_id=share_token_id,
        sidecars=sidecars,
        run_history_prefix=history,
        resume_config=resume_config,
        train_generator=train_generator,
        valid_generator=valid_generator,
        start_epoch=(
            int(resume["warmup_start_epoch"])
            if resume is not None
            else 1
        ),
        resume_state=(
            resume["state"]
            if resume is not None
            and resume["resume_stage"] == "warmup"
            else None
        ),
    )
    history.extend(warmup_history)
    joint_history = train_stage(
        stage_name="joint",
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
        epochs=args.joint_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        head_lr=args.head_joint_lr,
        lora_lr=args.lora_lr,
        max_future_weight=args.lambda_future,
        mc_samples=args.mc_samples,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        anti_shortcut_samples=args.anti_shortcut_samples,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        comment_nll_limit=args.comment_nll_limit,
        min_action_macro_f1=args.min_action_macro_f1,
        anti_shortcut_weight=args.anti_shortcut_weight,
        anti_shortcut_margin=args.anti_shortcut_margin,
        comment_token_id=comment_token_id,
        share_token_id=share_token_id,
        sidecars=sidecars,
        run_history_prefix=history,
        resume_config=resume_config,
        train_generator=train_generator,
        valid_generator=valid_generator,
        start_epoch=(
            int(resume["joint_start_epoch"])
            if resume is not None
            else 1
        ),
        resume_state=(
            resume["state"]
            if resume is not None
            and resume["resume_stage"] == "joint"
            else None
        ),
    )
    history.extend(joint_history)
    eligible_checkpoint = output_dir / "best_checkpoint"
    status = "completed" if eligible_checkpoint.exists() else "failed_gates"
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "history": history,
                "eligible_checkpoint": str(eligible_checkpoint)
                if eligible_checkpoint.exists()
                else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not eligible_checkpoint.exists():
        raise RuntimeError(
            "No joint checkpoint passed comment NLL, action F1 and "
            "text-sensitivity gates."
        )


if __name__ == "__main__":
    main()
