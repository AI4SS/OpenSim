from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from opensim.state_transition.dataset import (  # noqa: E402
    LatentStateEventDataset,
    TextFeatureEncoder,
    collate_latent_events,
)
from opensim.state_transition.model import (  # noqa: E402
    FutureAwareLatentSTNet,
)


@dataclass
class LossWeights:
    daily: float = 1.0
    cumulative: float = 0.5
    future_daily: float = 0.8
    future_cumulative: float = 0.4
    individual: float = 0.3
    future_direction: float = 0.15
    future_delta: float = 0.35
    future_impact: float = 0.1
    entropy: float = 0.005


@dataclass
class TrainConfig:
    macro_dir: str
    micro_dir: str
    output_dir: str
    agents_per_day: int = 128
    k_steps: int = 3
    d_model: int = 192
    latent_dim: int = 64
    nhead: int = 6
    num_layers: int = 2
    dropout: float = 0.1
    feature_dropout: float = 0.0
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    valid_events: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder_path: str | None = "Qwen/Qwen3-1.7B"
    text_cache_db: str | None = None
    text_encoder_device: str = "cpu"
    text_max_length: int = 128
    text_batch_size: int = 32
    semantic_agent_profiles: bool = False
    macro_label_smoothing: float = 0.0
    future_label_smoothing: float = 0.0
    weak_label_smoothing: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def event_ids(macro_dir: str | Path, micro_dir: str | Path) -> list[str]:
    macro_dir = Path(macro_dir)
    micro_dir = Path(micro_dir)
    ids = []
    for path in sorted(macro_dir.glob("*.jsonl")):
        if (micro_dir / f"{path.stem}.json").exists():
            ids.append(path.stem)
    if not ids:
        raise ValueError(f"No paired events found in {macro_dir} and {micro_dir}")
    return ids


def split_events(ids: list[str], valid_count: int, seed: int) -> tuple[list[str], list[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    valid_count = min(max(valid_count, 1), max(len(ids) - 1, 1))
    valid = sorted(ids[:valid_count])
    train = sorted(ids[valid_count:])
    return train, valid


def smooth_distribution(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return target
    smoothing = min(max(smoothing, 0.0), 0.45)
    uniform = torch.full_like(target, 1.0 / target.shape[-1])
    return target * (1.0 - smoothing) + uniform * smoothing


def masked_kl(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    kl = target.clamp_min(eps) * (target.clamp_min(eps).log() - pred.clamp_min(eps).log())
    kl = kl.sum(dim=-1) * mask
    return kl.sum() / mask.sum().clamp_min(1.0)


def cumulative_from_daily_mask_aware(
    daily_pred: torch.Tensor,
    update_weight: torch.Tensor,
    day_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate daily predictions; zero evidence days preserve cumulative state."""
    batch_size, time_steps, _ = daily_pred.shape
    running = daily_pred.new_zeros(batch_size, 3)
    total = daily_pred.new_zeros(batch_size, 1)
    outputs = []
    for t in range(time_steps):
        valid = day_mask[:, t].unsqueeze(-1)
        weight = update_weight[:, t].clamp_min(0.0).unsqueeze(-1) * valid
        running = running * total + daily_pred[:, t] * weight
        total = total + weight
        current = running / total.clamp_min(1.0)
        outputs.append(current)
        running = current
    return torch.stack(outputs, dim=1)


def individual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    agent_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = confidence * agent_mask
    kl = target.clamp_min(eps) * (target.clamp_min(eps).log() - pred.clamp_min(eps).log())
    kl = kl.sum(dim=-1) * mask
    return kl.sum() / mask.sum().clamp_min(1.0)


def future_direction_loss(
    future_pred: torch.Tensor,
    future_target: torch.Tensor,
    current_pred: torch.Tensor,
    current_target: torch.Tensor,
    future_mask: torch.Tensor,
) -> torch.Tensor:
    pred_delta = future_pred - current_pred.unsqueeze(2)
    target_delta = future_target - current_target.unsqueeze(2)
    loss = (pred_delta - target_delta).abs().sum(dim=-1) * future_mask
    return loss.sum() / future_mask.sum().clamp_min(1.0)


def future_delta_loss(pred: torch.Tensor, target: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1) * future_mask
    return loss.sum() / future_mask.sum().clamp_min(1.0)


def future_impact_loss(pred: torch.Tensor, target: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none") * future_mask
    return loss.sum() / future_mask.sum().clamp_min(1.0)


def entropy_loss(pred: torch.Tensor, day_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    entropy = -(pred.clamp_min(eps) * pred.clamp_min(eps).log()).sum(dim=-1)
    # Mildly penalize overconfident collapse, so minimize negative entropy.
    return -(entropy * day_mask).sum() / day_mask.sum().clamp_min(1.0)


def batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    weights: LossWeights,
    device: torch.device,
    gamma: float,
    macro_label_smoothing: float,
    future_label_smoothing: float,
    weak_label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    day_features = batch["day_features"].to(device)
    agent_features = batch["agent_features"].to(device)
    agent_mask = batch["agent_mask"].to(device)
    day_mask = batch["day_mask"].to(device)
    weak_labels = batch["weak_labels"].to(device)
    weak_conf = batch["weak_conf"].to(device)
    daily_gt = batch["daily_gt"].to(device)
    cumulative_gt = batch["cumulative_gt"].to(device)
    future_daily_gt = batch["future_daily_gt"].to(device)
    future_cumulative_gt = batch["future_cumulative_gt"].to(device)
    future_daily_delta_gt = batch["future_daily_delta_gt"].to(device)
    future_impact_gt = batch["future_impact_gt"].to(device)
    future_mask = batch["future_mask"].to(device)
    daily_target_mask = batch.get("daily_target_mask", day_mask).to(device)
    cumulative_target_mask = batch.get("cumulative_target_mask", day_mask).to(device)
    future_daily_mask = batch.get("future_daily_mask", future_mask).to(device)
    future_cumulative_mask = batch.get("future_cumulative_mask", future_mask).to(device)
    update_weight = batch["update_weight"].to(device)

    daily_target = smooth_distribution(daily_gt, macro_label_smoothing)
    cumulative_target = smooth_distribution(cumulative_gt, macro_label_smoothing)
    future_daily_target = smooth_distribution(future_daily_gt, future_label_smoothing)
    future_cumulative_target = smooth_distribution(future_cumulative_gt, future_label_smoothing)
    weak_target = smooth_distribution(weak_labels, weak_label_smoothing)

    outputs = model(
        day_features=day_features,
        agent_features=agent_features,
        agent_mask=agent_mask,
        day_mask=day_mask,
    )
    daily_pred = outputs["daily_pred"]
    cumulative_pred = cumulative_from_daily_mask_aware(daily_pred, update_weight, day_mask)
    future_daily_pred = outputs["future_daily_pred"]
    future_cumulative_pred = outputs["future_cumulative_pred"]
    future_delta_pred = outputs["future_delta_pred"]
    future_impact_pred = outputs["future_impact_pred"]

    daily_loss_mask = day_mask * daily_target_mask
    cumulative_loss_mask = day_mask * cumulative_target_mask
    daily = masked_kl(daily_pred, daily_target, daily_loss_mask)
    cumulative = masked_kl(cumulative_pred, cumulative_target, cumulative_loss_mask)
    individual = individual_loss(outputs["state_dist"], weak_target, weak_conf, agent_mask)

    K = future_daily_pred.shape[2]
    gamma_weights = torch.tensor([gamma**k for k in range(K)], device=device, dtype=future_mask.dtype)
    gamma_view = gamma_weights.view(1, 1, K)
    weighted_future_daily_mask = future_daily_mask * gamma_view
    weighted_future_cumulative_mask = future_cumulative_mask * gamma_view
    current_daily_mask = daily_target_mask.unsqueeze(-1)
    current_cumulative_mask = cumulative_target_mask.unsqueeze(-1)
    direction_delta_mask = weighted_future_daily_mask * current_daily_mask
    weighted_future_impact_mask = (
        future_mask
        * future_daily_mask
        * future_cumulative_mask
        * current_daily_mask
        * current_cumulative_mask
        * gamma_view
    )
    future_daily = masked_kl(
        future_daily_pred,
        future_daily_target,
        weighted_future_daily_mask,
    )
    future_cumulative = masked_kl(
        future_cumulative_pred,
        future_cumulative_target,
        weighted_future_cumulative_mask,
    )
    direction = future_direction_loss(
        future_daily_pred,
        future_daily_gt,
        daily_pred,
        daily_gt,
        direction_delta_mask,
    )
    delta = future_delta_loss(
        future_delta_pred,
        future_daily_delta_gt,
        direction_delta_mask,
    )
    impact = future_impact_loss(
        future_impact_pred,
        future_impact_gt,
        weighted_future_impact_mask,
    )
    ent = entropy_loss(daily_pred, day_mask)

    loss = (
        weights.daily * daily
        + weights.cumulative * cumulative
        + weights.future_daily * future_daily
        + weights.future_cumulative * future_cumulative
        + weights.individual * individual
        + weights.future_direction * direction
        + weights.future_delta * delta
        + weights.future_impact * impact
        + weights.entropy * ent
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "daily_kl": float(daily.detach().cpu()),
        "cum_kl": float(cumulative.detach().cpu()),
        "future_daily_kl": float(future_daily.detach().cpu()),
        "future_cum_kl": float(future_cumulative.detach().cpu()),
        "individual_kl": float(individual.detach().cpu()),
        "future_direction_l1": float(direction.detach().cpu()),
        "future_delta_l1": float(delta.detach().cpu()),
        "future_impact_l1": float(impact.detach().cpu()),
        "entropy_loss": float(ent.detach().cpu()),
    }
    return loss, metrics


def average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    weights: LossWeights,
    device: torch.device,
    gamma: float,
    macro_label_smoothing: float,
    future_label_smoothing: float,
    weak_label_smoothing: float,
    grad_clip: float,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics: list[dict[str, float]] = []
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        loss, batch_metrics = batch_loss(
            model,
            batch,
            weights,
            device,
            gamma,
            macro_label_smoothing,
            future_label_smoothing,
            weak_label_smoothing,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        metrics.append(batch_metrics)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items() if k in {"loss", "future_daily_kl", "daily_kl"}})
    return average_metrics(metrics)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    weights: LossWeights,
    train_events: list[str],
    valid_events: list[str],
    metrics: dict[str, float],
    epoch: int,
    dims: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "loss_weights": asdict(weights),
            "train_events": train_events,
            "valid_events": valid_events,
            "metrics": metrics,
            "dims": dims,
            "format": "future_aware_latent_st",
        },
        path,
    )


def validation_score(metrics: dict[str, float]) -> float:
    return (
        metrics.get("daily_kl", math.inf)
        + metrics.get("future_daily_kl", math.inf)
        + 0.25 * metrics.get("future_delta_l1", math.inf)
    )


def train_one_split(
    args: argparse.Namespace,
    cfg: TrainConfig,
    weights: LossWeights,
    train_ids: list[str],
    valid_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    cfg = replace(cfg, output_dir=str(output_dir))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    if cfg.text_encoder_path and num_workers != 0:
        print("text encoder is loaded inside the dataset; forcing --num-workers 0")
        num_workers = 0
    text_feature_encoder = TextFeatureEncoder(
        model_path=cfg.text_encoder_path,
        cache_db=cfg.text_cache_db,
        device=cfg.text_encoder_device,
        max_length=cfg.text_max_length,
        batch_size=cfg.text_batch_size,
    )
    (output_dir / "split.json").write_text(
        json.dumps(
            {"train_events": train_ids, "valid_events": valid_ids},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_ds = LatentStateEventDataset(
        cfg.macro_dir,
        cfg.micro_dir,
        event_ids=train_ids,
        agents_per_day=cfg.agents_per_day,
        k_steps=cfg.k_steps,
        seed=cfg.seed,
        text_encoder_path=cfg.text_encoder_path,
        text_cache_db=cfg.text_cache_db,
        text_encoder_device=cfg.text_encoder_device,
        text_max_length=cfg.text_max_length,
        text_batch_size=cfg.text_batch_size,
        semantic_agent_profiles=cfg.semantic_agent_profiles,
        text_feature_encoder=text_feature_encoder,
    )
    valid_ds = LatentStateEventDataset(
        cfg.macro_dir,
        cfg.micro_dir,
        event_ids=valid_ids,
        agents_per_day=cfg.agents_per_day,
        k_steps=cfg.k_steps,
        seed=cfg.seed,
        text_encoder_path=cfg.text_encoder_path,
        text_cache_db=cfg.text_cache_db,
        text_encoder_device=cfg.text_encoder_device,
        text_max_length=cfg.text_max_length,
        text_batch_size=cfg.text_batch_size,
        semantic_agent_profiles=cfg.semantic_agent_profiles,
        text_feature_encoder=text_feature_encoder,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_latent_events,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_latent_events,
        pin_memory=device.type == "cuda",
    )

    model = FutureAwareLatentSTNet(
        day_dim=train_ds.dims.day_dim,
        agent_dim=train_ds.dims.agent_dim,
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feature_dropout=cfg.feature_dropout,
        max_len=args.max_len,
        k_steps=cfg.k_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dims = {"day_dim": train_ds.dims.day_dim, "agent_dim": train_ds.dims.agent_dim}
    meta = {
        "config": asdict(cfg),
        "loss_weights": asdict(weights),
        "dims": dims,
        "train_events": train_ids,
        "valid_events": valid_ids,
        "score": "daily_kl + future_daily_kl + 0.25 * future_delta_l1",
        "regularization": {
            "dropout": cfg.dropout,
            "feature_dropout": cfg.feature_dropout,
            "weight_decay": cfg.weight_decay,
            "macro_label_smoothing": cfg.macro_label_smoothing,
            "future_label_smoothing": cfg.future_label_smoothing,
            "weak_label_smoothing": cfg.weak_label_smoothing,
        },
        "early_stopping": {
            "patience": args.early_stop_patience,
            "min_epochs": args.min_epochs,
            "min_delta": args.min_delta,
        },
        "success_criteria": [
            "valid future_daily_kl decreases or remains stable",
            "valid future_delta_l1 learns future direction beyond persistence",
            "valid future_impact_l1 remains finite",
            "daily/cumulative KL are finite",
            "checkpoint contains the latent state transition model",
        ],
    }
    (output_dir / "train_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    stale_epochs = 0
    stopped_reason = "max_epochs"
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            args.gamma,
            cfg.macro_label_smoothing,
            cfg.future_label_smoothing,
            cfg.weak_label_smoothing,
            cfg.grad_clip,
            desc=f"train {epoch}/{cfg.epochs}",
        )
        with torch.no_grad():
            valid_metrics = run_epoch(
                model,
                valid_loader,
                None,
                weights,
                device,
                args.gamma,
                cfg.macro_label_smoothing,
                cfg.future_label_smoothing,
                cfg.weak_label_smoothing,
                cfg.grad_clip,
                desc=f"valid {epoch}/{cfg.epochs}",
            )
        score = validation_score(valid_metrics)
        improved = score < best_score - args.min_delta
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "score": score,
            "best_score": min(best_score, score),
            "improved": improved,
        }
        history.append(row)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            cfg,
            weights,
            train_ids,
            valid_ids,
            valid_metrics,
            epoch,
            dims,
        )
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = valid_metrics
            stale_epochs = 0
            save_checkpoint(
                output_dir / "future_aware_latent_st_best.pt",
                model,
                optimizer,
                cfg,
                weights,
                train_ids,
                valid_ids,
                valid_metrics,
                epoch,
                dims,
            )
        else:
            stale_epochs += 1

        print(json.dumps(row, ensure_ascii=False, indent=2))
        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            stopped_reason = "early_stop"
            break

    summary = {
        "output_dir": str(output_dir),
        "train_event_count": len(train_ids),
        "valid_event_count": len(valid_ids),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_valid": best_metrics,
        "stopped_epoch": history[-1]["epoch"] if history else 0,
        "stopped_reason": stopped_reason,
    }
    (output_dir / "early_stop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        macro_dir=args.macro_dir,
        micro_dir=args.micro_dir,
        output_dir=args.output_dir,
        agents_per_day=args.agents_per_day,
        k_steps=args.k_steps,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_dropout=args.feature_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        valid_events=args.valid_events,
        seed=args.seed,
        device=args.device,
        text_encoder_path=args.text_encoder_path,
        text_cache_db=args.text_cache_db,
        text_encoder_device=args.text_encoder_device,
        text_max_length=args.text_max_length,
        text_batch_size=args.text_batch_size,
        semantic_agent_profiles=args.semantic_agent_profiles,
        macro_label_smoothing=args.macro_label_smoothing,
        future_label_smoothing=args.future_label_smoothing,
        weak_label_smoothing=args.weak_label_smoothing,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.text_encoder_path and cfg.text_cache_db is None:
        cfg.text_cache_db = str(output_dir / "text_embedding_cache.sqlite")

    weights = LossWeights(
        daily=args.lambda_daily,
        cumulative=args.lambda_cumulative,
        future_daily=args.lambda_future_daily,
        future_cumulative=args.lambda_future_cumulative,
        individual=args.lambda_individual,
        future_direction=args.lambda_future_direction,
        future_delta=args.lambda_future_delta,
        future_impact=args.lambda_future_impact,
        entropy=args.lambda_entropy,
    )
    ids = event_ids(cfg.macro_dir, cfg.micro_dir)

    train_ids, valid_ids = split_events(ids, cfg.valid_events, cfg.seed)
    summary = train_one_split(args, cfg, weights, train_ids, valid_ids, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal latent state transition model.")
    parser.add_argument("--macro-dir", default="data/train/macro")
    parser.add_argument("--micro-dir", default="data/train/micro")
    parser.add_argument("--output-dir", default="artifacts/state_transition")
    parser.add_argument("--agents-per-day", type=int, default=128)
    parser.add_argument("--k-steps", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--valid-events", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-encoder-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--text-cache-db", default=None)
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--semantic-agent-profiles", action="store_true")
    parser.add_argument("--macro-label-smoothing", type=float, default=0.0)
    parser.add_argument("--future-label-smoothing", type=float, default=0.0)
    parser.add_argument("--weak-label-smoothing", type=float, default=0.0)

    parser.add_argument("--lambda-daily", type=float, default=1.0)
    parser.add_argument("--lambda-cumulative", type=float, default=0.5)
    parser.add_argument("--lambda-future-daily", type=float, default=0.8)
    parser.add_argument("--lambda-future-cumulative", type=float, default=0.4)
    parser.add_argument("--lambda-individual", type=float, default=0.3)
    parser.add_argument("--lambda-future-direction", type=float, default=0.15)
    parser.add_argument("--lambda-future-delta", type=float, default=0.35)
    parser.add_argument("--lambda-future-impact", type=float, default=0.1)
    parser.add_argument("--lambda-entropy", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
