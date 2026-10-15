from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from opensim.state_transition.dataset import (  # noqa: E402
    LatentStateEventDataset,
    SHARE_TEXTS,
    clean_text,
    content_type,
    macro_source_t_row_positions,
    dist_from_dict,
    profile_text,
    record_row_position,
    sample_records_by_day as _sample_records_by_day,
    target_mask_from_diagnostics,
)
from opensim.state_transition.model import FutureAwareLatentSTNet  # noqa: E402


COMMENT_CONTAMINATION_RE = re.compile(
    r"不要编号|解释[，,]不要|请输出|禁止输出|作为AI|作为人工智能|系统提示|"
    r"instruction|prompt|回答格式",
    re.IGNORECASE,
)


def to_policy_order(dist_pos_neu_neg: list[float] | torch.Tensor) -> dict[str, float]:
    if isinstance(dist_pos_neu_neg, torch.Tensor):
        values = dist_pos_neu_neg.detach().float().cpu().tolist()
    else:
        values = [float(value) for value in dist_pos_neu_neg]
    pos, neu, neg = values
    return {"pos": pos, "neg": neg, "neu": neu}


def sample_records_by_day(
    event_id: str,
    micro_rows: list[dict[str, Any]],
    agents_per_day: int,
    seed: int,
    macro_rows: list[dict[str, Any]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Use the exact sampling helper consumed by the ST dataset."""
    return _sample_records_by_day(
        event_id,
        micro_rows,
        agents_per_day,
        seed,
        macro_rows=macro_rows,
    )


def root_posts_by_day(
    micro_rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]] | None = None,
) -> dict[int, str]:
    source_t_positions = (
        macro_source_t_row_positions(macro_rows) if macro_rows is not None else None
    )
    posts: dict[int, str] = {}
    for record in micro_rows:
        if content_type(record) != "post":
            continue
        row_position = record_row_position(record, source_t_positions)
        if row_position is None:
            continue
        posts.setdefault(row_position, clean_text(record.get("text") or record.get("content")))
    return posts


def historical_mf_summary(macro_rows: list[dict[str, Any]], day_idx: int) -> str:
    """Return only mean-field context available before the current day."""
    input_info = macro_rows[day_idx].get("input") or {}
    previous = clean_text(input_info.get("previous_mf_summary"))
    if previous:
        return previous
    if day_idx <= 0:
        return ""
    return clean_text(macro_rows[day_idx - 1].get("mf_summary"))


def historical_representative_comments(
    macro_rows: list[dict[str, Any]],
    day_idx: int,
    limit: int = 3,
) -> str:
    if day_idx <= 0:
        return ""
    comments = macro_rows[day_idx - 1].get("representative_comments") or []
    return "；".join(clean_text(text) for text in comments[:limit] if clean_text(text))


def action_target(record: dict[str, Any], canonical_share: bool) -> tuple[str, str] | None:
    source = content_type(record)
    text = clean_text(record.get("text") or record.get("content") or record.get("original_text"))
    if source == "share":
        if canonical_share or text in SHARE_TEXTS:
            return "转发", "转发微博"
        return "转发", text
    if source == "comment":
        if not text or text in SHARE_TEXTS:
            return None
        if COMMENT_CONTAMINATION_RE.search(text):
            return None
        return "评论", text
    return None


def descriptor_from_latent(
    state_dist: torch.Tensor,
    future_delta: torch.Tensor | None,
    future_impact: torch.Tensor | None,
    latent_code: str | None = None,
) -> str:
    pos, neu, neg = state_dist.detach().float().cpu().tolist()
    named = [("积极", pos), ("中立", neu), ("消极", neg)]
    label, confidence = max(named, key=lambda item: item[1])
    if confidence >= 0.70:
        stance = f"强{label}"
    elif confidence >= 0.50:
        stance = f"偏{label}"
    else:
        stance = "混合观望"

    impact_text = "未知"
    if future_impact is not None and future_impact.numel() > 0:
        impact = float(future_impact.detach().float().mean().cpu())
        if impact >= 0.25:
            impact_text = "高"
        elif impact >= 0.10:
            impact_text = "中"
        else:
            impact_text = "低"

    trend_text = "不明显"
    if future_delta is not None and future_delta.numel() > 0:
        delta = future_delta.detach().float().mean(dim=0).cpu().tolist()
        names = ["积极", "中立", "消极"]
        trend_idx = max(range(3), key=lambda idx: abs(delta[idx]))
        if abs(delta[trend_idx]) >= 0.03:
            trend_text = f"{names[trend_idx]}{'上升' if delta[trend_idx] > 0 else '下降'}"

    descriptor = (
        f"{stance}；态度分布：积极{pos:.0%}，中立{neu:.0%}，消极{neg:.0%}；"
        f"未来影响强度：{impact_text}；未来变化倾向：{trend_text}"
    )
    if latent_code:
        return f"潜在状态码：{latent_code}；{descriptor}"
    return descriptor


def balance_action_rows(
    rows: list[dict[str, Any]],
    share_to_comment_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if share_to_comment_ratio < 0:
        return rows

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["event_id"]), int(row["day_index"]))
        groups.setdefault(key, []).append(row)

    balanced: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        comments = [row for row in group if row["action_type"] == "评论"]
        shares = [row for row in group if row["action_type"] == "转发"]
        if comments:
            max_shares = max(1, math.ceil(len(comments) * share_to_comment_ratio))
        else:
            max_shares = min(2, len(shares))
        if len(shares) > max_shares:
            rng_seed = seed + int(hashlib.md5(f"{key[0]}:{key[1]}:share".encode("utf-8")).hexdigest()[:8], 16)
            shares = random.Random(rng_seed).sample(shares, max_shares)
        balanced.extend(comments)
        balanced.extend(shares)
    return balanced


def fit_and_apply_latent_codes(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    n_clusters: int,
    seed: int,
    output_dir: Path,
) -> None:
    train_latents = np.asarray([row["_latent_vector"] for row in train_rows], dtype=np.float32)
    valid_latents = np.asarray([row["_latent_vector"] for row in valid_rows], dtype=np.float32)
    n_clusters = min(n_clusters, len(train_latents))
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(2048, len(train_latents)),
        random_state=seed,
        n_init="auto",
    )
    train_codes = clusterer.fit_predict(train_latents)
    valid_codes = clusterer.predict(valid_latents)

    for rows, codes in ((train_rows, train_codes), (valid_rows, valid_codes)):
        for row, code in zip(rows, codes):
            latent_vector = row.pop("_latent_vector")
            latent_code = f"L{int(code):02d}"
            row["latent_code"] = latent_code
            row["st_latent"] = json.dumps(latent_vector, ensure_ascii=False)
            row["latent_state_descriptor"] = (
                f"潜在状态码：{latent_code}；{row['latent_state_descriptor']}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clusterer, output_dir / "latent_kmeans.joblib")


def future_distribution(
    macro_rows: list[dict[str, Any]],
    day_idx: int,
    step_idx: int,
    target_name: str,
) -> dict[str, float] | None:
    target_idx = day_idx + step_idx + 1
    if target_idx >= len(macro_rows):
        return None
    target_mask_name = (
        "daily_target_mask" if target_name == "daily" else "cumulative_target_mask"
    )
    target_diagnostics = macro_rows[target_idx].get("diagnostics") or {}
    if target_mask_from_diagnostics(target_diagnostics, target_mask_name) <= 0.0:
        return None
    key = "daily_state_distribution" if target_name == "daily" else "cumulative_state_distribution"
    return to_policy_order(dist_from_dict(macro_rows[target_idx].get(key)))


def load_st_model(checkpoint_path: Path, device: torch.device) -> tuple[FutureAwareLatentSTNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dims = checkpoint["dims"]
    model = FutureAwareLatentSTNet(
        day_dim=int(dims["day_dim"]),
        agent_dim=int(dims["agent_dim"]),
        d_model=int(config.get("d_model", 192)),
        latent_dim=int(config.get("latent_dim", 64)),
        nhead=int(config.get("nhead", 6)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
        feature_dropout=0.0,
        max_len=int(config.get("max_len", 64)),
        k_steps=int(config.get("k_steps", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def build_rows_for_events(
    event_ids: list[str],
    args: argparse.Namespace,
    model: FutureAwareLatentSTNet,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    config = checkpoint["config"]
    dataset = LatentStateEventDataset(
        macro_dir=args.macro_dir,
        micro_dir=args.micro_dir,
        event_ids=event_ids,
        agents_per_day=args.agents_per_day,
        k_steps=int(config.get("k_steps", 3)),
        seed=args.seed,
        text_encoder_path=config.get("text_encoder_path"),
        text_cache_db=config.get("text_cache_db"),
        text_encoder_device=args.text_encoder_device,
        text_max_length=int(config.get("text_max_length", 64)),
        text_batch_size=int(config.get("text_batch_size", 16)),
        semantic_agent_profiles=bool(config.get("semantic_agent_profiles", False)),
    )

    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        event_id = item["event_id"]
        macro_path = Path(args.macro_dir) / f"{event_id}.jsonl"
        micro_path = Path(args.micro_dir) / f"{event_id}.json"
        macro_rows = [json.loads(line) for line in macro_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        micro_rows = json.loads(micro_path.read_text(encoding="utf-8"))
        sampled = sample_records_by_day(
            event_id,
            micro_rows,
            args.agents_per_day,
            args.seed,
            macro_rows=macro_rows,
        )
        posts_by_day = root_posts_by_day(micro_rows, macro_rows=macro_rows)

        with torch.no_grad():
            outputs = model(
                day_features=item["day_features"].unsqueeze(0).to(device),
                agent_features=item["agent_features"].unsqueeze(0).to(device),
                agent_mask=item["agent_mask"].unsqueeze(0).to(device),
                day_mask=torch.ones(1, item["seq_len"], dtype=torch.float32, device=device),
            )

        state_dist = outputs["state_dist"][0].cpu()
        state_latent = outputs["state_latent"][0].cpu()
        future_delta = outputs.get("future_delta_pred")
        future_impact = outputs.get("future_impact_pred")
        if future_delta is not None:
            future_delta = future_delta[0].cpu()
        if future_impact is not None:
            future_impact = future_impact[0].cpu()

        for day_idx, macro_row in enumerate(macro_rows):
            records = sampled.get(day_idx, [])
            if not records:
                continue
            topic = clean_text(macro_row.get("topic") or event_id)
            input_info = macro_row.get("input") or {}
            mf_summary = historical_mf_summary(macro_rows, day_idx)
            root_post = posts_by_day.get(day_idx) or clean_text(input_info.get("event"))
            hot_comment = historical_representative_comments(macro_rows, day_idx)
            disturb_names = "、".join(clean_text(text) for text in (input_info.get("disturb_names") or []))
            related = (
                f"day={day_idx}; has_disturb={bool(input_info.get('has_disturb'))}; "
                f"disturb={disturb_names or '无'}; days_since_disturb={input_info.get('days_since_disturb')}"
            )

            for agent_idx, record in enumerate(records):
                target_pair = action_target(record, canonical_share=args.canonical_share_target)
                if target_pair is None:
                    continue
                action, target_text = target_pair
                current = to_policy_order(state_dist[day_idx, agent_idx])
                descriptor = descriptor_from_latent(
                    state_dist[day_idx, agent_idx],
                    future_delta[day_idx] if future_delta is not None else None,
                    future_impact[day_idx] if future_impact is not None else None,
                )

                out: dict[str, Any] = {
                    "event_id": event_id,
                    "day_index": day_idx,
                    "uid": clean_text(record.get("uid") or record.get("user_id")),
                    "record_id": clean_text(record.get("record_id") or record.get("id")),
                    "action_type": action,
                    "profile_text": profile_text(record) or "未知画像",
                    "topic": topic,
                    "root_post": root_post,
                    "hot_comment": hot_comment,
                    "batch_mf": mf_summary,
                    "related_cases_info": related,
                    "latent_state_descriptor": descriptor,
                    "_latent_vector": state_latent[day_idx, agent_idx].float().tolist(),
                    "real_comments": target_text,
                    "pre_pos": current["pos"],
                    "pre_neg": current["neg"],
                    "pre_neu": current["neu"],
                }
                for step_idx in range(args.k_steps):
                    future = future_distribution(macro_rows, day_idx, step_idx, args.future_target)
                    is_valid = future is not None
                    if future is None:
                        future = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
                    out[f"dist_t{step_idx}_pos"] = future["pos"]
                    out[f"dist_t{step_idx}_neg"] = future["neg"]
                    out[f"dist_t{step_idx}_neu"] = future["neu"]
                    out[f"future_valid_t{step_idx}"] = int(is_valid)
                rows.append(out)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro_dir", default="data/train/macro")
    parser.add_argument("--micro_dir", default="data/train/micro")
    parser.add_argument("--st_checkpoint", default="artifacts/state_transition/future_aware_latent_st_best.pt")
    parser.add_argument("--output_dir", default="artifacts/prepared_policy")
    parser.add_argument("--agents_per_day", type=int, default=128)
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--future_target", choices=["daily", "cumulative"], default="cumulative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default="cuda")
    parser.add_argument("--canonical_share_target", action="store_true", default=True)
    parser.add_argument("--latent_clusters", type=int, default=32)
    parser.add_argument("--share_to_comment_ratio", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, checkpoint = load_st_model(Path(args.st_checkpoint), device)
    train_events = list(checkpoint.get("train_events") or [])
    valid_events = list(checkpoint.get("valid_events") or [])
    if not train_events or not valid_events:
        raise ValueError("ST checkpoint must contain train_events and valid_events for event-level policy split.")

    train_rows = build_rows_for_events(train_events, args, model, checkpoint, device)
    valid_rows = build_rows_for_events(valid_events, args, model, checkpoint, device)

    output_dir = Path(args.output_dir)
    original_train_rows = len(train_rows)
    original_valid_rows = len(valid_rows)
    train_rows = balance_action_rows(train_rows, args.share_to_comment_ratio, args.seed)
    valid_rows = balance_action_rows(valid_rows, 1.0, args.seed)
    fit_and_apply_latent_codes(
        train_rows,
        valid_rows,
        args.latent_clusters,
        args.seed,
        output_dir,
    )
    train_path = output_dir / "policy_train.csv"
    valid_path = output_dir / "policy_valid.csv"
    write_csv(train_path, train_rows)
    write_csv(valid_path, valid_rows)

    summary = {
        "st_checkpoint": args.st_checkpoint,
        "future_target": args.future_target,
        "k_steps": args.k_steps,
        "agents_per_day": args.agents_per_day,
        "train_events": train_events,
        "valid_events": valid_events,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "original_train_rows": original_train_rows,
        "original_valid_rows": original_valid_rows,
        "share_to_comment_ratio": args.share_to_comment_ratio,
        "latent_clusters": args.latent_clusters,
        "state_order_for_model": "pos,neu,neg (positive, neutral, negative)",
        "latent_bridge": "raw ST state_latent[64] plus KMeans code and interpretable descriptor",
        "prompt_causal_boundary": (
            "current external event/root post + historical previous_mf_summary/"
            "previous representative comments; no current-day mf_summary or "
            "current-day representative comments in prompt"
        ),
        "train_csv": str(train_path),
        "valid_csv": str(valid_path),
    }
    (output_dir / "policy_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
