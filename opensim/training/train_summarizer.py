#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA SFT trainer for the mean-field summarizer."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def apply_chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    template_kwargs = (
        {"enable_thinking": False}
        if "qwen3" in tokenizer_name or "enable_thinking" in chat_template
        else {}
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:
            pass
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


class PromptTargetDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = str(row["prompt"])
        target = str(row["target"]).strip()
        prompt_text = apply_chat(self.tokenizer, prompt)
        eos = self.tokenizer.eos_token or "<|im_end|>"

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target + eos, add_special_tokens=False).input_ids
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id]
        if len(target_ids) >= self.max_len:
            target_ids = target_ids[: self.max_len - 1]
        max_prompt_len = max(self.max_len - len(target_ids), 1)
        # Keep the prompt tail so the assistant generation marker is preserved.
        prompt_ids = prompt_ids[-max_prompt_len:]
        ids = prompt_ids + target_ids
        labels_list = [-100] * len(prompt_ids) + list(target_ids)
        if len(ids) > self.max_len:
            overflow = len(ids) - self.max_len
            ids = ids[overflow:]
            labels_list = labels_list[overflow:]
        pad_len = self.max_len - len(ids)
        pad_id = self.tokenizer.pad_token_id
        ids = ids + [pad_id] * pad_len
        labels_list = labels_list + [-100] * pad_len
        attention = [1] * (self.max_len - pad_len) + [0] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(attention, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "action_type": row.get("action_type", ""),
            "event_id": row.get("event_id", ""),
            "day_index": row.get("day_index", ""),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "action_type": [item["action_type"] for item in batch],
        "event_id": [item["event_id"] for item in batch],
        "day_index": [item["day_index"] for item in batch],
    }


def token_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {"overall": [0.0, 0.0], "comment": [0.0, 0.0], "share": [0.0, 0.0]}
    for batch in tqdm(loader, desc="Validate", leave=False):
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
        for idx, action in enumerate(batch["action_type"]):
            valid = shift_labels[idx] != -100
            loss_sum = float(losses[idx][valid].sum().detach().cpu())
            token_count = float(valid.sum().detach().cpu())
            totals["overall"][0] += loss_sum
            totals["overall"][1] += token_count
            bucket = "share" if str(action).lower() == "share" else "comment"
            totals.setdefault(bucket, [0.0, 0.0])
            totals[bucket][0] += loss_sum
            totals[bucket][1] += token_count
    out: dict[str, float] = {}
    for name, (loss_sum, token_count) in totals.items():
        nll = loss_sum / max(token_count, 1.0)
        out[f"{name}_nll"] = float(nll)
        out[f"{name}_ppl"] = float(math.exp(min(nll, 20.0)))
        out[f"{name}_tokens"] = float(token_count)
    model.train()
    return out


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    name: str,
    state: dict[str, Any],
    is_best: bool,
) -> None:
    target = output_dir / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    torch.save(state, target / "trainer_state.pt")

    last = output_dir / "last_checkpoint"
    if last.exists():
        shutil.rmtree(last)
    shutil.copytree(target, last)
    if is_best:
        best = output_dir / "best_checkpoint"
        if best.exists():
            shutil.rmtree(best)
        shutil.copytree(target, best)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_bf16 = device.type == "cuda" and cuda_major >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    )
    base_model.config.use_cache = False
    if device.type == "cuda" and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config).to(device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_rows = read_jsonl(Path(args.train_data), args.max_train_rows)
    valid_rows = read_jsonl(Path(args.valid_data), args.max_valid_rows)
    train_ds = PromptTargetDataset(train_rows, tokenizer, args.max_len)
    valid_ds = PromptTargetDataset(valid_rows, tokenizer, args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = max(updates_per_epoch * args.epochs, 1)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    config = vars(args) | {
        "device": str(device),
        "dtype": str(dtype),
        "train_rows": len(train_ds),
        "valid_rows": len(valid_ds),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    best_nll = float("inf")
    best_epoch = 0
    stale = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            loss = token_loss(logits, labels, args.label_smoothing)
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
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        metrics = evaluate(model, valid_loader, device)
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_sum / max(train_batches, 1),
            **metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        improved = metrics["overall_nll"] < best_nll - args.min_delta
        if improved:
            best_nll = metrics["overall_nll"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_overall_nll": best_nll,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }
        save_checkpoint(model, tokenizer, output_dir, f"checkpoint-epoch-{epoch}", state, improved)
        (output_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_overall_nll": best_nll,
                    "history": history,
                    "status": "running",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if epoch >= 2 and stale >= args.early_stop_patience:
            print(f"[EARLY STOP] no overall NLL improvement for {stale} epoch(s)", flush=True)
            break

    summary = {
        "best_epoch": best_epoch,
        "best_overall_nll": best_nll,
        "history": history,
        "status": "completed",
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
