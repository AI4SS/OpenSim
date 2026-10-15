#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare training examples for the mean-field summarizer from final train data."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SHARE_TEXTS = {"转发", "转发微博", "分享", "转发了", "已转发"}
COMMENT_CONTAMINATION_RE = re.compile(
    r"不要编号|解释[，,]不要|请输出|禁止输出|作为AI|作为人工智能|系统提示|instruction|prompt|回答格式",
    re.IGNORECASE,
)


def clean_text(value: Any, limit: int = 0) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def content_type(record: dict[str, Any]) -> str:
    return str(record.get("content_type") or record.get("source_content_type") or "").lower()


def parse_t_index(value: Any) -> int | None:
    """Parse a timestep token without inventing a value for missing metadata."""
    if value is None or value == "":
        return None
    value = str(value)
    digits = "".join(ch for ch in value if ch.isdigit())
    return max(int(digits), 1) if digits else None


def extract_t_index(record: dict[str, Any]) -> int:
    """Read the source timestep, preferring explicit ``source_t`` metadata."""
    parsed = parse_t_index(record.get("source_t") or record.get("t"))
    return parsed if parsed is not None else 1


def macro_source_t_to_day_index(macro_rows: list[dict[str, Any]]) -> dict[int, int]:
    """Map preserved source timesteps to compact macro row indices.

    Cleaned data can preserve non-contiguous source timesteps.  The macro
    ``diagnostics.source_t`` field is the authoritative join key; the list
    position is the compact day index consumed by this builder.  An empty
    mapping preserves the historical ``t - 1`` fallback for legacy files.
    """
    mapping: dict[int, int] = {}
    missing_source_t: list[int] = []
    for day_idx, row in enumerate(macro_rows):
        diagnostics = row.get("diagnostics") or {}
        raw_source_t = (
            diagnostics.get("source_t")
            if isinstance(diagnostics, dict)
            else None
        )
        if raw_source_t is None:
            raw_source_t = row.get("source_t")
        source_t = parse_t_index(raw_source_t)
        if source_t is None:
            missing_source_t.append(day_idx)
            continue
        previous = mapping.get(source_t)
        if previous is not None and previous != day_idx:
            raise ValueError(
                f"duplicate macro source_t={source_t}: rows {previous} and {day_idx}"
            )
        mapping[source_t] = day_idx
    if mapping and missing_source_t:
        raise ValueError(
            "macro source_t mapping is incomplete at compact day indices "
            f"{missing_source_t}"
        )
    return mapping


def _record_day_index(
    record: dict[str, Any], source_t_to_day_index: dict[int, int] | None = None
) -> int:
    source_t = extract_t_index(record)
    if source_t_to_day_index:
        if source_t not in source_t_to_day_index:
            raise ValueError(
                f"record source_t={source_t} is absent from the explicit macro source_t mapping"
            )
        return source_t_to_day_index[source_t]
    return source_t - 1


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default




def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def posts_by_day(
    micro_rows: list[dict[str, Any]],
    source_t_to_day_index: dict[int, int] | None = None,
) -> dict[int, str]:
    out: dict[int, list[str]] = {}
    for record in micro_rows:
        if content_type(record) != "post":
            continue
        day_idx = _record_day_index(record, source_t_to_day_index)
        text = clean_text(record.get("text") or record.get("content") or record.get("source_text"), 180)
        if text:
            out.setdefault(day_idx, []).append(text)
    return {day: "；".join(values[:3]) for day, values in out.items()}


def records_by_day(
    micro_rows: list[dict[str, Any]],
    source_t_to_day_index: dict[int, int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for record in micro_rows:
        if content_type(record) == "post":
            continue
        day_idx = _record_day_index(record, source_t_to_day_index)
        out.setdefault(day_idx, []).append(record)
    for values in out.values():
        values.sort(key=lambda r: safe_int(r.get("source_original_index") or r.get("create_time")))
    return out


def event_context(event_id: str, macro_row: dict[str, Any], day_posts: str) -> str:
    topic = clean_text(macro_row.get("topic") or event_id, 80)
    input_info = macro_row.get("input") or {}
    env = clean_text(input_info.get("event"), 360)
    if day_posts:
        post_text = f"当天外生帖子/干预：{day_posts}"
        return f"话题：{topic}。{env}。{post_text}" if env else f"话题：{topic}。{post_text}"
    return f"话题：{topic}。{env or '当前无新的外生帖子，讨论延续前序事件。'}"


def previous_mf_summary(macro_rows: list[dict[str, Any]], day_idx: int) -> str:
    input_info = macro_rows[day_idx].get("input") or {}
    previous = clean_text(input_info.get("previous_mf_summary"), 600)
    if previous:
        return previous
    if day_idx <= 0:
        return ""
    return clean_text(macro_rows[day_idx - 1].get("mf_summary"), 600)


def action_target(record: dict[str, Any], canonical_share: bool, skip_trivial_share: bool = False) -> tuple[str, str] | None:
    source = content_type(record)
    text = clean_text(record.get("text") or record.get("content") or record.get("original_text"), 180)
    if source == "share":
        if skip_trivial_share and (not text or text in SHARE_TEXTS):
            return None
        if canonical_share or text in SHARE_TEXTS:
            return "share", "转发微博"
        return "share", text
    if source == "comment":
        if not text or text in SHARE_TEXTS:
            return None
        if COMMENT_CONTAMINATION_RE.search(text):
            return None
        return "comment", text
    return None




def discussion_sample(records: list[dict[str, Any]], max_items: int, seed: int) -> str:
    usable = []
    for record in records:
        target = action_target(record, canonical_share=True)
        if target is None:
            continue
        action, text = target
        prefix = "转发" if action == "share" else "评论"
        usable.append(f"- [{prefix}] {clean_text(text, 100)}")
    if len(usable) > max_items:
        usable = random.Random(seed).sample(usable, max_items)
    return "\n".join(usable)




def build_summarizer_prompt(prev_summary: str, context: str, discussion: str) -> str:
    return "\n".join(
        [
            "你是社交媒体舆论场摘要器。请只基于上一轮摘要、当前事件环境和本日讨论样本，生成本日舆论环境总结。",
            "摘要要覆盖主要立场、情绪、行为倾向和讨论焦点，不要编造未出现的信息。",
            "",
            f"事件环境：{context}",
            f"上一轮摘要：{prev_summary or '暂无。'}",
            "本日讨论样本：",
            discussion or "本日没有有效讨论样本。",
            "",
            "输出不超过220字的中文摘要。",
        ]
    )


def build_rows_for_event(
    event_id: str,
    micro_path: Path,
    macro_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    micro_rows = json.loads(micro_path.read_text(encoding="utf-8"))
    macro_rows = read_jsonl(macro_path)
    source_t_to_day_index = macro_source_t_to_day_index(macro_rows)
    day_posts = posts_by_day(micro_rows, source_t_to_day_index)
    day_records = records_by_day(micro_rows, source_t_to_day_index)
    summarizer_rows: list[dict[str, Any]] = []
    for day_idx, macro_row in enumerate(macro_rows):
        records = day_records.get(day_idx, [])
        if not records:
            continue
        context = event_context(event_id, macro_row, day_posts.get(day_idx, ""))
        prev_mf = previous_mf_summary(macro_rows, day_idx)
        discussion = discussion_sample(
            records,
            args.summary_discussion_items,
            args.seed + int(hashlib.md5(f"{event_id}:{day_idx}:summary".encode("utf-8")).hexdigest()[:8], 16),
        )
        target_summary = clean_text(macro_row.get("mf_summary"), 900)
        if discussion and target_summary:
            summarizer_rows.append(
                {
                    "task": "summarizer",
                    "event_id": event_id,
                    "day_index": day_idx,
                    "prompt": build_summarizer_prompt(prev_mf, context, discussion),
                    "target": target_summary,
                }
            )
    return summarizer_rows


def event_split(events: list[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    events = sorted(events)
    rng = random.Random(seed)
    shuffled = list(events)
    rng.shuffle(shuffled)
    valid_n = max(1, round(len(shuffled) * valid_ratio))
    valid = set(shuffled[:valid_n])
    train_events = [event for event in events if event not in valid]
    valid_events = [event for event in events if event in valid]
    return train_events, valid_events


def event_split_from_config(
    events: list[str], split_config: Path
) -> tuple[list[str], list[str]]:
    """Read an explicit event split from an ST training config.

    This opt-in path lets the auxiliary summarizer share the explicit ST
    train/valid event boundary.  Omitting ``--split-config`` retains the
    historical seeded event split behavior.
    """
    payload = json.loads(split_config.read_text(encoding="utf-8"))
    source = payload
    if not {"train_events", "valid_events"}.issubset(source) and isinstance(
        payload.get("config"), dict
    ):
        source = payload["config"] | {
            key: value for key, value in payload.items() if key != "config"
        }
    train_raw = source.get("train_events")
    valid_raw = source.get("valid_events")
    if not isinstance(train_raw, list) or not isinstance(valid_raw, list):
        raise ValueError(
            f"split config must provide train_events and valid_events: {split_config}"
        )

    available = set(events)
    train = {str(event) for event in train_raw}
    valid = {str(event) for event in valid_raw}
    overlap = train & valid
    unknown = (train | valid) - available
    missing = available - (train | valid)
    if overlap or unknown or missing:
        raise ValueError(
            "split config does not exactly cover input events: "
            f"overlap={sorted(overlap)}, unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    return sorted(train), sorted(valid)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")




def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-dir", default="data/train/micro")
    parser.add_argument("--macro-dir", default="data/train/macro")
    parser.add_argument("--output-dir", default="artifacts/prepared_summarizer")
    parser.add_argument("--summary-discussion-items", type=int, default=64)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Optional ST train_config.json providing train_events/valid_events.",
    )
    args = parser.parse_args()

    micro_dir = Path(args.micro_dir)
    macro_dir = Path(args.macro_dir)
    events = sorted(path.stem for path in micro_dir.glob("*.json") if (macro_dir / f"{path.stem}.jsonl").exists())
    if args.split_config is not None:
        train_events, valid_events = event_split_from_config(events, args.split_config)
        split_source = str(args.split_config)
    else:
        train_events, valid_events = event_split(events, args.valid_ratio, args.seed)
        split_source = "seeded_event_split"
    splits = {"train": train_events, "valid": valid_events}
    output_dir = Path(args.output_dir)

    summary: dict[str, Any] = {
        "events_total": len(events),
        "train_events": train_events,
        "valid_events": valid_events,
        "summary_discussion_items": args.summary_discussion_items,
        "split_source": split_source,
        "causal_boundary": "summarizer uses previous summary, current event context and discussion observed after the discussion; current summary is a supervised target",
    }

    for split, split_events in splits.items():
        rows: list[dict[str, Any]] = []
        for event_id in split_events:
            rows.extend(build_rows_for_event(
                event_id, micro_dir / f"{event_id}.json", macro_dir / f"{event_id}.jsonl", args
            ))
        write_jsonl(output_dir / f"summarizer_{split}.jsonl", rows)
        summary[f"{split}_rows"] = len(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summarizer_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
