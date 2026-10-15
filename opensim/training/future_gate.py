#!/usr/bin/env python3
"""Re-evaluate future-policy checkpoints with aligned feature interventions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from opensim.policy.dataset import (
    PolicyMFDataset,
    load_and_process_csv_data,
)
from opensim.policy.model import FutureConstrainedPolicyModel
from opensim.training.train_policy_future import (
    collate_policy,
    evaluate,
)


PROTOCOL = "cross_event_day_feature_intervention"
FINGERPRINT_FILES = (
    "adapter_model.safetensors",
    "trajectory_critic.pt",
)


def validate_intervention_batch_size(batch_size: int) -> int:
    """Require at least one cross-row negative for shuffled interventions."""

    value = int(batch_size)
    if value < 2:
        raise ValueError(
            "Future-gate intervention batch size must be at least 2; "
            "batch_size=1 makes the shuffled response identical to matched."
        )
    return value


def resolve_candidate_checkpoints(
    checkpoint_root: Path,
    requested: list[str],
) -> list[Path]:
    candidates = (
        [Path(value) for value in requested]
        if requested
        else sorted(checkpoint_root.glob("joint-epoch-*"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No joint-epoch-* checkpoints under {checkpoint_root}"
        )
    missing = [path for path in candidates if not path.is_dir()]
    if missing:
        raise FileNotFoundError(missing[0])
    return candidates


def checkpoint_fingerprint(checkpoint: Path) -> dict[str, str]:
    fingerprint = {}
    for name in FINGERPRINT_FILES:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        fingerprint[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint


def load_checkpoint(
    checkpoint: Path,
    pretrain: str,
    device: torch.device,
) -> tuple[AutoTokenizer, FutureConstrainedPolicyModel]:
    policy_metadata = json.loads(
        (checkpoint / "policy_format.json").read_text(encoding="utf-8")
    )
    critic_config = json.loads(
        (checkpoint / "trajectory_critic_config.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrain,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    )
    llm = PeftModel.from_pretrained(
        base_model,
        checkpoint,
        local_files_only=True,
        is_trainable=False,
    )
    model = FutureConstrainedPolicyModel(
        llm=llm,
        llm_hidden_size=int(critic_config["llm_hidden_size"]),
        latent_dim=int(critic_config["latent_dim"]),
        critic_hidden_size=int(critic_config["critic_hidden_size"]),
        k_steps=int(critic_config["k_steps"]),
        critic_dropout=float(critic_config["dropout"]),
        action_loss_weight=float(policy_metadata.get("action_loss_weight", 1.0)),
    )
    action_weights = policy_metadata.get("action_class_weights")
    if action_weights:
        model.set_action_class_weights(torch.tensor(action_weights))
    critic_state = torch.load(
        checkpoint / "trajectory_critic.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.critic.load_state_dict(critic_state)
    model.to(device).eval()
    return tokenizer, model


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    pretrain: str,
    records: list[dict],
    device: torch.device,
    max_len: int,
    k_steps: int,
    batch_size: int,
    num_workers: int,
    mc_samples: int,
    anti_shortcut_samples: int,
) -> dict[str, float]:
    tokenizer, model = load_checkpoint(checkpoint, pretrain, device)
    metadata = json.loads(
        (checkpoint / "policy_format.json").read_text(encoding="utf-8")
    )
    dataset = PolicyMFDataset(
        records,
        tokenizer,
        max_length=max_len,
        k_steps=k_steps,
        policy_format=metadata["policy_format"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_policy,
        generator=torch.Generator().manual_seed(43),
    )
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        mc_samples=mc_samples,
        anti_shortcut_samples=anti_shortcut_samples,
        max_eval_batches=0,
        comment_token_id=int(metadata["comment_token_id"]),
        share_token_id=int(metadata["share_token_id"]),
    )
    del model, tokenizer, dataloader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def promote_checkpoint(
    checkpoint: Path,
    destination: Path,
    metrics: dict[str, float],
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(checkpoint, destination)
    state_path = destination / "trainer_state.pt"
    state = (
        torch.load(state_path, map_location="cpu", weights_only=False)
        if state_path.exists()
        else {}
    )
    state["metrics"] = metrics
    state["gate_protocol"] = PROTOCOL
    state["requalified_from"] = str(checkpoint)
    torch.save(state, state_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=704)
    parser.add_argument("--k-steps", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--anti-shortcut-samples", type=int, default=1024)
    parser.add_argument("--comment-nll-limit", type=float, default=4.5)
    parser.add_argument("--min-action-macro-f1", type=float, default=0.55)
    parser.add_argument("--min-relative-gap", type=float, default=0.05)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Evaluate only this checkpoint path; repeat for multiple "
            "checkpoints. By default all joint-epoch-* directories are used."
        ),
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument(
        "--require-current-best-match",
        action="store_true",
        help=(
            "Before promotion, require the selected checkpoint to have the "
            "same adapter and critic hashes as the existing best_checkpoint. "
            "Use this when simulation artifacts already exist."
        ),
    )
    args = parser.parse_args()
    args.batch_size = validate_intervention_batch_size(args.batch_size)

    checkpoint_root = Path(args.checkpoint_root)
    candidates = resolve_candidate_checkpoints(
        checkpoint_root,
        args.candidate,
    )
    records = load_and_process_csv_data(args.valid_csv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    for checkpoint in candidates:
        metrics = evaluate_checkpoint(
            checkpoint=checkpoint,
            pretrain=args.pretrain,
            records=records,
            device=device,
            max_len=args.max_len,
            k_steps=args.k_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mc_samples=args.mc_samples,
            anti_shortcut_samples=args.anti_shortcut_samples,
        )
        gates = {
            "finite": all(
                math.isfinite(float(value))
                for value in metrics.values()
                if isinstance(value, (int, float))
            ),
            "comment_nll": (
                metrics["comment_nll"] <= args.comment_nll_limit
            ),
            "action_macro_f1": (
                metrics["action_macro_f1"] >= args.min_action_macro_f1
            ),
            "masked_gap": metrics["masked_gap"] >= args.min_relative_gap,
            "shuffled_gap": metrics["shuffled_gap"] >= args.min_relative_gap,
        }
        selection_score = (
            metrics["matched_trajectory_kl"]
            + 0.25 * metrics["matched_delta_l1"]
            + 0.10 * metrics["matched_direction_loss"]
        )
        results.append(
            {
                "checkpoint": str(checkpoint),
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
                "selection_score": selection_score,
            }
        )

    eligible = [result for result in results if result["passed"]]
    selected = (
        min(eligible, key=lambda result: result["selection_score"])
        if eligible
        else None
    )
    current_best = checkpoint_root / "best_checkpoint"
    current_best_matches_selected = None
    if selected and current_best.exists():
        current_best_matches_selected = (
            checkpoint_fingerprint(current_best)
            == checkpoint_fingerprint(Path(selected["checkpoint"]))
        )
    promotion_blocked = bool(
        selected
        and args.promote
        and args.require_current_best_match
        and current_best_matches_selected is not True
    )
    if selected and args.promote and not promotion_blocked:
        selected_metrics = dict(selected["metrics"])
        selected_metrics["text_sensitive"] = 1.0
        selected_metrics["gate_protocol"] = PROTOCOL
        promote_checkpoint(
            Path(selected["checkpoint"]),
            checkpoint_root / "best_checkpoint",
            selected_metrics,
        )
        summary_path = checkpoint_root / "train_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        summary["status"] = "completed"
        summary["eligible_checkpoint"] = str(
            checkpoint_root / "best_checkpoint"
        )
        summary["gate_recheck"] = {
            "protocol": PROTOCOL,
            "selected": selected["checkpoint"],
            "results": results,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "protocol": PROTOCOL,
        "interventions": {
            "matched": "teacher-forced response feature",
            "masked": "zero response feature",
            "shuffled": "response feature from a different event/day",
        },
        "checkpoint_root": str(checkpoint_root),
        "batch_size": args.batch_size,
        "anti_shortcut_samples": args.anti_shortcut_samples,
        "results": results,
        "selected": selected["checkpoint"] if selected else "",
        "current_best_before_promotion": str(current_best),
        "current_best_matches_selected": current_best_matches_selected,
        "promotion_blocked": promotion_blocked,
        "promoted": bool(selected and args.promote and not promotion_blocked),
        "passed": bool(selected and not promotion_blocked),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if promotion_blocked:
        return 5
    return 0 if selected else 4


if __name__ == "__main__":
    raise SystemExit(main())
