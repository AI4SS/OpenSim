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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from opensim.policy.dataset import PolicyMFDataset, load_and_process_csv_data  # noqa: E402
from opensim.policy.objectives import (  # noqa: E402
    action_class_weights,
    action_predictions,
    binary_macro_f1,
    response_objective,
)
from opensim.policy.prompting import (  # noqa: E402
    POLICY_FORMAT_ACTION_CONTENT,
    POLICY_FORMATS,
    normalize_action,
)
from opensim.training.collate import custom_collate_fn  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


RESUME_CONFIG_KEYS = (
    "pretrain",
    "dataset",
    "eval_dataset",
    "output_dir",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_len",
    "label_smoothing",
    "policy_format",
    "action_loss_weight",
    "action_class_balance",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "num_workers",
    "grad_clip",
    "early_stop_patience",
    "min_delta",
    "gradient_checkpointing",
    "max_train_batches",
    "max_eval_batches",
    "seed",
)


def load_resume_training_state(
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    required_files = (
        checkpoint_path / "adapter_config.json",
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "trainer_state.pt",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete; missing: {missing}"
        )
    state = torch.load(
        checkpoint_path / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    required_state_keys = {
        "epoch",
        "global_step",
        "best_comment_nll",
        "best_selection_score",
        "best_epoch",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
        "history",
        "stale_epochs",
    }
    missing_state = sorted(required_state_keys.difference(state))
    if missing_state:
        raise ValueError(
            f"Resume trainer_state is missing keys: {missing_state}"
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
            "Resume configuration mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    checkpoint_epoch = int(state["epoch"])
    if checkpoint_epoch < 1 or checkpoint_epoch > int(args.epochs):
        raise ValueError(
            "Resume checkpoint epoch must be completed and no greater than the "
            f"requested epochs; got {checkpoint_epoch}/{args.epochs}."
        )
    state_history = state["history"]
    if not isinstance(state_history, list) or not state_history:
        raise ValueError("Resume trainer_state has no completed epoch history.")
    history_epochs = [
        int(row.get("epoch", -1))
        for row in state_history
    ]
    expected_epochs = list(range(1, checkpoint_epoch + 1))
    if history_epochs != expected_epochs:
        raise ValueError(
            "Resume history must contain each completed epoch exactly once; "
            f"got {history_epochs}, expected {expected_epochs}."
        )
    summary_path = output_dir / "train_summary.json"
    summary_missing = not summary_path.is_file()
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary_missing
        else {"status": "running", "history": []}
    )
    if summary.get("status") == "completed":
        raise ValueError("Refusing to resume a completed SFT run.")
    summary_history = summary.get("history")
    if not isinstance(summary_history, list):
        raise ValueError("Resume summary history is not a list.")
    summary_epochs = [
        int(row.get("epoch", -1))
        for row in summary_history
    ]
    allowed_summary_epochs = (
        history_epochs,
        history_epochs[:-1],
    )
    if not summary_missing and summary_epochs not in allowed_summary_epochs:
        raise ValueError(
            "Resume summary must match the checkpoint history or lag only "
            f"its final atomic write; got {summary_epochs}, expected "
            f"{history_epochs} or {history_epochs[:-1]}."
        )
    best_epoch = int(state["best_epoch"])
    if best_epoch < 1 or best_epoch > checkpoint_epoch:
        raise ValueError(
            f"Invalid best_epoch={best_epoch} at epoch={checkpoint_epoch}."
        )
    if (
        summary_epochs == history_epochs
        and int(summary.get("best_epoch", -1)) != best_epoch
    ):
        raise ValueError("Resume checkpoint and summary disagree on best_epoch.")
    if not state["optimizer_state_dict"].get("param_groups"):
        raise ValueError("Resume checkpoint has no optimizer parameter groups.")
    if "last_epoch" not in state["scheduler_state_dict"]:
        raise ValueError("Resume checkpoint has no scheduler last_epoch.")

    fallback_seed = int(args.seed) + checkpoint_epoch
    audit = {
        "checkpoint": str(checkpoint_path),
        "from_epoch": checkpoint_epoch,
        "from_global_step": int(state["global_step"]),
        "start_epoch": checkpoint_epoch + 1,
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "rng_state_available": isinstance(state.get("rng_state"), dict),
        "train_generator_state_available": isinstance(
            state.get("train_generator_state"),
            torch.Tensor,
        ),
        "deterministic_fallback_seed": fallback_seed,
        "summary_missing_repaired": summary_missing,
        "summary_lag_repaired": (
            not summary_missing
            and summary_epochs == history_epochs[:-1]
        ),
    }
    return {
        "state": state,
        "history": state_history,
        "start_epoch": checkpoint_epoch + 1,
        "stale_epochs": int(state["stale_epochs"]),
        "fallback_seed": fallback_seed,
        "audit": audit,
    }


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


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    *,
    comment_token_id: int,
    share_token_id: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = {
        "overall": [0.0, 0],
        "comment": [0.0, 0],
        "share": [0.0, 0],
    }
    row_loss_sum = 0.0
    row_count = 0
    action_loss_sum = 0.0
    action_token_count = 0
    action_targets: list[int] = []
    action_predictions_all: list[int] = []
    for batch_index, batch in enumerate(
        tqdm(dataloader, desc="Validate", leave=False)
    ):
        if max_batches > 0 and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        valid_rows = shift_labels.ne(-100)
        row_losses = (losses * valid_rows).sum(dim=1) / valid_rows.sum(dim=1).clamp_min(1)
        row_loss_sum += float(row_losses.sum().cpu())
        row_count += int(row_losses.numel())

        action_mask = batch["action_token_mask"][:, 1:].to(device).bool() & valid_rows
        action_loss_sum += float((losses * action_mask).sum().cpu())
        action_token_count += int(action_mask.sum().cpu())
        valid_action_rows = batch["action_decision_index"].ge(0)
        if bool(valid_action_rows.any()):
            predicted = action_predictions(
                logits,
                batch["action_decision_index"].to(device),
                comment_token_id,
                share_token_id,
            )
            action_targets.extend(batch["action_class"][valid_action_rows].tolist())
            action_predictions_all.extend(predicted[valid_action_rows.to(device)].cpu().tolist())

        for idx, action in enumerate(batch["action_type"]):
            valid = shift_labels[idx] != -100
            loss_sum = float(losses[idx][valid].sum().cpu())
            token_count = int(valid.sum().cpu())
            bucket = "share" if action == "转发" else "comment"
            totals["overall"][0] += loss_sum
            totals["overall"][1] += token_count
            totals[bucket][0] += loss_sum
            totals[bucket][1] += token_count

    metrics: dict[str, float] = {}
    for name, (loss_sum, token_count) in totals.items():
        nll = loss_sum / max(token_count, 1)
        metrics[f"{name}_nll"] = nll
        metrics[f"{name}_ppl"] = math.exp(min(nll, 20.0))
        metrics[f"{name}_tokens"] = token_count
    metrics["row_nll"] = row_loss_sum / max(row_count, 1)
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
    metrics["action_rows"] = len(action_targets)
    model.train()
    return metrics


def save_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    name: str,
    state: dict,
    sidecars: list[Path] | None = None,
    is_best: bool = False,
) -> Path:
    checkpoint_dir = output_dir / name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    for sidecar in sidecars or []:
        if sidecar.exists():
            shutil.copy2(sidecar, checkpoint_dir / sidecar.name)

    last_dir = output_dir / "last_checkpoint"
    if last_dir.exists():
        shutil.rmtree(last_dir)
    shutil.copytree(checkpoint_dir, last_dir)
    if is_best:
        best_dir = output_dir / "best_checkpoint"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
    return checkpoint_dir


def _copy_sft_checkpoint_alias_if_needed(
    source: Path,
    target: Path,
) -> None:
    required = (
        source / "adapter_config.json",
        source / "adapter_model.safetensors",
        source / "trainer_state.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot rebuild SFT checkpoint alias; missing: {missing}"
        )
    source_state = torch.load(
        source / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_epoch = int(source_state.get("epoch", -1))
    target_epoch = -1
    target_state_path = target / "trainer_state.pt"
    if target_state_path.is_file():
        target_state = torch.load(
            target_state_path,
            map_location="cpu",
            weights_only=False,
        )
        target_epoch = int(target_state.get("epoch", -1))
    target_complete = all(
        (target / path.name).is_file()
        for path in required
    )
    if target_complete and target_epoch == source_epoch:
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def rebuild_sft_checkpoint_aliases(
    output_dir: Path,
    resume_checkpoint: Path,
    state: dict,
) -> None:
    _copy_sft_checkpoint_alias_if_needed(
        resume_checkpoint,
        output_dir / "last_checkpoint",
    )
    best_epoch = int(state["best_epoch"])
    _copy_sft_checkpoint_alias_if_needed(
        output_dir / f"checkpoint-epoch-{best_epoch}",
        output_dir / "best_checkpoint",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=8e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--max_len", type=int, default=768)
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
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=1)
    parser.add_argument("--min_delta", type=float, default=0.01)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Trade compute for activation memory when training larger bases.",
    )
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
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
    if resume is not None:
        rebuild_sft_checkpoint_aliases(
            output_dir,
            Path(args.resume_checkpoint).resolve(),
            resume["state"],
        )
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_bf16 = device.type == "cuda" and cuda_major >= 8 and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)

    tokenizer_source = (
        args.resume_checkpoint if args.resume_checkpoint else args.pretrain
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=model_dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    if resume is not None:
        model = PeftModel.from_pretrained(
            base_model,
            args.resume_checkpoint,
            is_trainable=True,
        ).to(device)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base_model, lora_config).to(device)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_raw = load_and_process_csv_data(args.dataset)
    valid_raw = load_and_process_csv_data(args.eval_dataset)
    train_action_targets = [
        1 if normalize_action(record.get("action_type", "")) == "share" else 0
        for record in train_raw
    ]
    action_weights = action_class_weights(
        train_action_targets,
        mode=args.action_class_balance,
    ).to(device)
    train_dataset = PolicyMFDataset(
        train_raw,
        tokenizer,
        max_length=args.max_len,
        k_steps=10,
        policy_format=args.policy_format,
    )
    valid_dataset = PolicyMFDataset(
        valid_raw,
        tokenizer,
        max_length=args.max_len,
        k_steps=10,
        policy_format=args.policy_format,
    )
    train_generator = torch.Generator()
    if resume is not None and isinstance(
        resume["state"].get("train_generator_state"),
        torch.Tensor,
    ):
        train_generator.set_state(resume["state"]["train_generator_state"])
        train_generator_restored = True
    else:
        train_generator.manual_seed(
            int(resume["fallback_seed"]) if resume is not None else args.seed
        )
        train_generator_restored = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        generator=train_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    best_selection_score = float("inf")
    best_comment_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    history: list[dict] = []
    start_epoch = 1
    resume_audit = None
    if resume is not None:
        state = resume["state"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best_selection_score = float(state["best_selection_score"])
        best_comment_nll = float(state["best_comment_nll"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(resume["stale_epochs"])
        global_step = int(state["global_step"])
        history = list(resume["history"])
        start_epoch = int(resume["start_epoch"])
        rng_restored = restore_rng_state(
            state.get("rng_state"),
            int(resume["fallback_seed"]),
        )
        resume_audit = dict(resume["audit"])
        resume_audit.update(
            {
                "rng_state_restored": rng_restored,
                "train_generator_state_restored": (
                    train_generator_restored
                ),
            }
        )
        if (
            start_epoch >= 2
            and stale_epochs >= args.early_stop_patience
        ):
            start_epoch = args.epochs + 1
            resume_audit["early_stop_already_satisfied"] = True
        print(json.dumps({"resume": resume_audit}, ensure_ascii=False))

    config = vars(args) | {
        "device": str(device),
        "dtype": str(model_dtype),
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "action_class_weights": action_weights.detach().cpu().tolist(),
        "resume": resume_audit,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    running_summary = {
        "history": history,
        "status": "running",
        "resume": resume_audit,
    }
    if history:
        running_summary.update(
            {
                "best_epoch": best_epoch,
                "best_comment_nll": best_comment_nll,
                "best_selection_score": best_selection_score,
            }
        )
    (output_dir / "train_summary.json").write_text(
        json.dumps(running_summary, ensure_ascii=False, indent=2),
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
    ]
    format_path = output_dir / "policy_format.json"
    format_path.write_text(
        json.dumps(
            {
                "policy_format": args.policy_format,
                "action_loss_weight": args.action_loss_weight,
                "action_class_balance": args.action_class_balance,
                "action_class_weights": action_weights.detach().cpu().tolist(),
                "comment_token_id": comment_token_id,
                "share_token_id": share_token_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sidecars.append(format_path)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress):
            if (
                args.max_train_batches > 0
                and batch_idx >= args.max_train_batches
            ):
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            losses = response_objective(
                logits,
                labels,
                batch["action_token_mask"].to(device),
                action_class=batch["action_class"].to(device),
                action_weights=action_weights,
                label_smoothing=args.label_smoothing,
                action_loss_weight=args.action_loss_weight,
            )
            loss = losses["total_loss"]
            (loss / args.gradient_accumulation_steps).backward()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

            should_step = (
                (batch_idx + 1) % args.gradient_accumulation_steps == 0
                or batch_idx + 1 == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                action=f"{losses['action_loss'].item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            comment_token_id=comment_token_id,
            share_token_id=share_token_id,
            max_batches=args.max_eval_batches,
        )
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_sum / max(train_batches, 1),
            **valid_metrics,
        }
        selection_score = (
            valid_metrics["comment_nll"]
            + args.action_loss_weight * valid_metrics["action_nll"]
            + (1.0 - valid_metrics["action_macro_f1"])
        )
        epoch_result["selection_score"] = selection_score
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        improved = selection_score < best_selection_score - args.min_delta
        if improved:
            best_selection_score = selection_score
            best_comment_nll = valid_metrics["comment_nll"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_comment_nll": best_comment_nll,
            "best_selection_score": best_selection_score,
            "best_epoch": best_epoch,
            "metrics": valid_metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "history": history,
            "stale_epochs": stale_epochs,
            "train_generator_state": train_generator.get_state(),
            "rng_state": capture_rng_state(),
            "resume": resume_audit,
        }
        save_checkpoint(
            model,
            tokenizer,
            output_dir,
            f"checkpoint-epoch-{epoch}",
            state,
            sidecars=sidecars,
            is_best=improved,
        )
        (output_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_comment_nll": best_comment_nll,
                    "best_selection_score": best_selection_score,
                    "history": history,
                    "status": "running",
                    "resume": resume_audit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if epoch >= 2 and stale_epochs >= args.early_stop_patience:
            print(
                f"[EARLY STOP] no selection-score improvement for "
                f"{stale_epochs} epoch(s)"
            )
            break

    summary = {
        "best_epoch": best_epoch,
        "best_comment_nll": best_comment_nll,
        "best_selection_score": best_selection_score,
        "history": history,
        "status": "completed",
        "resume": resume_audit,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
