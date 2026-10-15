from __future__ import annotations

import argparse
import csv
import fcntl
import inspect
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .judging import judge_batch



DIMENSIONS = [
    "rumor",
    "sentiment_state",
    "sentiment_tendency",
    "behavior_type",
    "stance",
    "belief_degree",
    "subjectivity",
    "intent_classification",
]

EXTRA_DIMS = ["keywords"]

LABELS = {
    "rumor": ["spread", "counter"],
    "sentiment_state": ["angry", "calm", "happy", "sad", "fear", "surprise"],
    "sentiment_tendency": ["positive", "negative", "neutral"],
    "behavior_type": ["comment", "share"],
    "stance": ["support", "oppose", "neutral"],
    "belief_degree": ["believe", "doubt"],
    "subjectivity": ["subjective", "objective"],
    "intent_classification": ["question", "promotion", "opinion"],
}

BASE_FIELDS = [
    "mode",
    "event",
    "time_step",
    "agent_index",
    "is_warmup",
    "row_uid",
    "source_record_id",
    "source_mid",
    "source_content_type",
    "generated_action_hint",
    "real_text",
    "generated_text",
    "is_invalid_generation",
    "invalid_reason",
    "semantic_similarity",
    "fluency_score",
    "short_reason",
    "llm_model",
]

OUTPUT_FIELDS = (
    BASE_FIELDS
    + [f"real_{d}" for d in DIMENSIONS + EXTRA_DIMS]
    + [f"gen_{d}" for d in DIMENSIONS + EXTRA_DIMS]
)


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {(k or "").lstrip("\ufeff"): v for k, v in row.items()}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [clean_row(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid = obj.get("text_uid")
            if uid:
                if uid in out and is_complete_record(out[uid]) and out[uid] != obj:
                    raise ValueError(f"Accepted cache record was replaced: {uid}")
                out[uid] = obj
    return out


def is_complete_record(record: Dict[str, Any]) -> bool:
    if not record:
        return False
    for dim in DIMENSIONS:
        if str(record.get(dim) or "") not in LABELS[dim]:
            return False
    return True


def short_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def sha_uid(*parts: Any, size: int = 16) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:size]


def row_uid(mode: str, row: Dict[str, str]) -> str:
    from .likelihood import row_uid as exact_row_uid
    return exact_row_uid(row)


def text_uid(mode: str, event: str, side: str, text: str, action_hint: str = "") -> str:
    return sha_uid(mode, event, side, text, size=20)


def load_meta(index_csv: Path) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    if index_csv.exists():
        for row in read_csv_rows(index_csv):
            stem = row.get("standardized_stem") or row.get("event") or row.get("event_slug")
            if stem:
                meta[stem] = {key: row[key] for key in (
                    "event_title_zh", "event_title_en", "topic", "description"
                ) if row.get(key)}
    return meta


def build_text_records(mode: str, rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    texts: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        event = row.get("event", "")
        row["row_uid"] = row_uid(mode, row)
        real_text = str(row.get("real_comment") or row.get("source_text") or "")
        gen_text = str(row.get("generated_comment") or "")
        real_uid = text_uid(mode, event, "real", real_text)
        gen_uid = text_uid(mode, event, "gen", gen_text)
        row["real_text_uid"] = real_uid
        row["gen_text_uid"] = gen_uid
        pair_rows.append(row)
        texts.setdefault(
            real_uid,
            {
                "text_uid": real_uid,
                "mode": mode,
                "event": event,
                "side": "real",
                "time_step": row.get("time_step", ""),
                "text": real_text,
            },
        )
        texts.setdefault(
            gen_uid,
            {
                "text_uid": gen_uid,
                "mode": mode,
                "event": event,
                "side": "gen",
                "time_step": row.get("time_step", ""),
                "text": gen_text,
            },
        )
    return pair_rows, list(texts.values())


def build_prompt(event: str, event_meta: Dict[str, str], batch: List[Dict[str, Any]]) -> str:
    topic = event_meta.get("topic") or event_meta.get("event_title_zh") or event
    desc = short_text(event_meta.get("description", ""), 260)
    rows = [
        {
            "text_id": r["text_uid"],
            "t": r.get("time_step", ""),
            "text": short_text(r.get("text", ""), 320),
        }
        for r in batch
    ]
    schema = {
        "items": [
            {
                "text_id": "same text_id",
                "labels": {d: "one allowed label" for d in DIMENSIONS},
                "keywords": ["0-5 concise Chinese keywords"],
                "is_invalid": False,
                "invalid_reason": "",
                "fluency_score": 1.0,
                "short_reason": "short Chinese reason",
            }
        ]
    }
    label_text = "\n".join(f"- {dim}: {labels}" for dim, labels in LABELS.items())
    return (
        "你是中文社交媒体舆论内容评估员。只输出严格 JSON，不要 Markdown，不要解释。\n"
        "任务：对每条微博/评论文本独立分类。不要在不同文本之间做配对比较，"
        "也不要为了保持一致而修改任意一条文本的标签。\n"
        f"事件：{topic}\n事件ID：{event}\n"
        f"事件描述：{desc}\n\n"
        "分类维度和允许标签：\n"
        + label_text
        + "\n\n"
        "判定规则：\n"
        "1. rumor: counter 表示辟谣、怀疑、要求证据、不相信；其他传播、表达情绪或转发为 spread。\n"
        "2. sentiment_state 中普通平静或中性表达统一标为 calm，不要输出 neutral。\n"
        "3. behavior_type: 只能根据文本本身判断；'转发微博'、'转发'、只扩散信息且无个人评价通常为 share；评价、提问、表达态度为 comment。\n"
        "4. belief_degree: believe 表示相信或正常传播；doubt 表示怀疑、求证、指出虚假。\n"
        "5. subjectivity: 情绪、观点、立场为 subjective；事实陈述或客观信息传播为 objective。\n"
        "6. 空文本、乱码、明显模板复读、API失败文本或完全不能作为中文社交媒体行为时 is_invalid=true；仍需尽量给出 labels。\n"
        "7. fluency_score 是文本作为中文社交媒体内容的流畅度，范围 1 到 5。\n"
        "8. 必须为每个 text_id 返回一条结果，labels 必须包含全部 8 个维度。\n\n"
        "输出 JSON schema：\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\n待评估数据：\n"
        + json.dumps({"rows": rows}, ensure_ascii=False)
    )


def extract_json(text: str) -> Dict[str, Any] | None:
    text = (text or "").strip().replace("```json", "```")
    if text.startswith("```"):
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {"items": parsed}
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def normalize_label(dim: str, value: Any) -> str:
    value = str(value or "").strip().lower()
    aliases = {
        "comments": "comment",
        "评论": "comment",
        "转发": "share",
        "转发微博": "share",
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "supportive": "support",
        "opposition": "oppose",
        "objective_fact": "objective",
        "中性": "neutral",
        "平静": "calm",
        "冷静": "calm",
        "愤怒": "angry",
        "开心": "happy",
        "高兴": "happy",
        "悲伤": "sad",
        "恐惧": "fear",
        "惊讶": "surprise",
    }
    value = aliases.get(value, value)
    if dim == "rumor":
        rumor_aliases = {
            "rumor": "spread", "misinformation": "spread",
            "debunk": "counter", "debunking": "counter",
            "refute": "counter", "refutation": "counter",
            "skepticism": "counter", "skeptical": "counter",
        }
        value = rumor_aliases.get(value, value)
    if dim == "sentiment_state" and value == "neutral":
        value = "calm"
    return value if value in LABELS[dim] else ""


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def bounded_float(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, lower), upper)


def get_thread_openai_client(state: threading.local) -> Any:
    client = getattr(state, "openai_client", None)
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing")
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            max_retries=0,
        )
        state.openai_client = client
    return client


def load_attempts(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    attempts, strategies = {}, {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Corrupt request evidence at {path}:{number}") from exc
            for uid, count in record.get("attempts", {}).items():
                attempts[uid] = max(attempts.get(uid, 0), int(count))
            if record.get("strategy_after"):
                for uid in record["text_uids"]:
                    strategies[uid] = record["strategy_after"]
    return attempts, strategies


def record_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_text(value: str) -> str:
    secret = os.environ.get("OPENAI_API_KEY", "")
    if secret:
        value = value.replace(secret, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_.-]+", "[redacted]", value)


def build_text_record(text_row: Dict[str, Any], item: Dict[str, Any], model: str) -> Dict[str, Any]:
    labels = item.get("labels") or {}
    rec = {
        "text_uid": text_row["text_uid"],
        "mode": text_row.get("mode", ""),
        "event": text_row.get("event", ""),
        "side": text_row.get("side", ""),
        "time_step": text_row.get("time_step", ""),
        "action_hint": text_row.get("action_hint", ""),
        "text": text_row.get("text", ""),
        "is_invalid": as_bool(item.get("is_invalid")),
        "invalid_reason": str(item.get("invalid_reason", ""))[:180],
        "fluency_score": bounded_float(item.get("fluency_score"), 1.0, 5.0, 1.0),
        "short_reason": str(item.get("short_reason", ""))[:220],
        "llm_model": model,
    }
    for dim in DIMENSIONS:
        rec[dim] = normalize_label(dim, labels.get(dim))
    keywords = item.get("keywords", [])
    if isinstance(keywords, list):
        rec["keywords"] = "|".join(short_text(x, 24) for x in keywords[:5])
    else:
        rec["keywords"] = short_text(keywords, 80)
    return rec


def chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def evaluate_event(
    event: str,
    text_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    meta: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    if Path(event).name != event:
        raise ValueError("Event identity must be a filename stem")
    event_dir = Path(args.output_dir) / "events"
    cache_path = event_dir / f"{event}_text_cache.jsonl"
    evidence_path = event_dir / f"{event}_requests.jsonl"
    text_csv_path = event_dir / f"{event}_text_eval.csv"
    cache = load_cache(cache_path)
    accepted_before = {uid: record_hash(record) for uid, record in cache.items() if is_complete_record(record)}
    for row in text_rows:
        record = cache.get(row["text_uid"], {})
        if is_complete_record(record):
            if record.get("llm_model") != args.model or any(record.get(key) != row.get(key) for key in ("text_uid", "event", "side", "text")):
                raise ValueError("Accepted cache belongs to a different model or text; use a new output directory")
    pending = [row for row in text_rows if not is_complete_record(cache.get(row["text_uid"], {}))]
    attempts, strategies = load_attempts(evidence_path)
    batches = list(chunks(pending, args.batch_size))
    client_state = threading.local()
    audit_lock = threading.Lock()
    stopped = getattr(args, "stop_event", threading.Event())

    def evidence(trace: dict) -> None:
        serialized = safe_text(json.dumps(trace, ensure_ascii=False))
        with audit_lock:
            append_jsonl(evidence_path, json.loads(serialized))

    def accept(row: dict, item: dict) -> None:
        record = build_text_record(row, item, args.model)
        if not is_complete_record(record):
            raise ValueError("Validated response produced incomplete labels")
        with audit_lock:
            uid = row["text_uid"]
            if is_complete_record(cache.get(uid, {})):
                return
            append_jsonl(cache_path, record)
            cache[uid] = record

    def request(payload: dict) -> dict:
        client = get_thread_openai_client(client_state)
        response = client.chat.completions.create(**payload)
        choice = response.choices[0]
        return {"raw_response": response.model_dump(mode="json"), "content": choice.message.content,
                "refusal": getattr(choice.message, "refusal", None), "finish_reason": choice.finish_reason}

    def evaluate_batch(batch: list[dict]) -> None:
        try:
            judge_batch(
                batch, model=args.model, labels=LABELS,
                prompt=lambda rows: build_prompt(event, meta.get(event, {}), rows),
                parse=extract_json, request=request, accept=accept, evidence=evidence,
                attempts=attempts, strategies=strategies, max_attempts=args.max_attempts,
                timeout=args.timeout, max_tokens=args.max_tokens, backoff=args.backoff,
                max_backoff=args.max_backoff, cancelled=stopped.is_set,
            )
        except Exception as exc:
            stopped.set()
            raise RuntimeError(safe_text(f"{type(exc).__name__}: {exc}")) from None

    print(f"{event}: unique_texts={len(text_rows)} pending={len(pending)} max_attempts={args.max_attempts}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.batch_workers_per_event) as executor:
            futures = [executor.submit(evaluate_batch, batch) for batch in batches]
            for future in as_completed(futures):
                future.result()
    finally:
        current = load_cache(cache_path)
        if any(not is_complete_record(current.get(uid, {})) or record_hash(current[uid]) != value for uid, value in accepted_before.items()):
            raise ValueError("An accepted semantic cache record changed")
    ordered = [cache[row["text_uid"]] for row in text_rows if is_complete_record(cache.get(row["text_uid"], {}))]
    text_fields = ["text_uid", "mode", "event", "side", "time_step", "action_hint", "text",
                   "is_invalid", "invalid_reason", "fluency_score", "short_reason", "llm_model"] + DIMENSIONS + EXTRA_DIMS
    write_csv(text_csv_path, ordered, text_fields)
    return {"mode": args.mode, "event": event, "total": len(text_rows),
            "completed": len(ordered), "missing": len(text_rows) - len(ordered)}


def build_pairwise_output(mode: str, pair_rows: List[Dict[str, Any]], text_cache: Dict[str, Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in pair_rows:
        real = text_cache.get(row["real_text_uid"], {})
        gen = text_cache.get(row["gen_text_uid"], {})
        rec = {
            "mode": mode,
            "event": row.get("event", ""),
            "time_step": row.get("time_step", ""),
            "agent_index": row.get("agent_index", ""),
            "is_warmup": row.get("is_warmup", ""),
            "row_uid": row["row_uid"],
            "source_record_id": row.get("source_record_id", ""),
            "source_mid": row.get("source_mid", ""),
            "source_content_type": row.get("source_content_type", ""),
            "generated_action_hint": row.get("generated_action", ""),
            "real_text": row.get("real_comment") or row.get("source_text", ""),
            "generated_text": row.get("generated_comment", ""),
            "is_invalid_generation": gen.get("is_invalid", False),
            "invalid_reason": gen.get("invalid_reason", ""),
            "semantic_similarity": "",
            "fluency_score": gen.get("fluency_score", ""),
            "short_reason": gen.get("short_reason", ""),
            "llm_model": model,
        }
        for dim in DIMENSIONS + EXTRA_DIMS:
            rec[f"real_{dim}"] = real.get(dim, "")
            rec[f"gen_{dim}"] = gen.get(dim, "")
        rows.append(rec)
    return rows


def combine(
    output_root: Path,
    mode: str,
    pair_rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    mode_dir = output_root
    text_cache: Dict[str, Dict[str, Any]] = {}
    for path in sorted((mode_dir / "events").glob("*_text_cache.jsonl")):
        text_cache.update({uid: record for uid, record in load_cache(path).items()
                           if record.get("llm_model") == model})
    out_rows = build_pairwise_output(mode, pair_rows, text_cache, model)
    write_csv(mode_dir / "semantic.csv", out_rows, OUTPUT_FIELDS)
    write_csv(mode_dir / "progress.csv", summaries, ["mode", "event", "total", "completed", "missing"])
    return out_rows


def validate_complete_output(
    mode: str,
    pair_rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    out_rows: List[Dict[str, Any]],
    expected_model: str = "gpt-4o-mini",
) -> None:
    expected_events = {
        str(row.get("event", ""))
        for row in pair_rows
        if row.get("event")
    }
    progress_events = {
        str(row.get("event", ""))
        for row in summaries
        if row.get("event")
    }
    output_events = {
        str(row.get("event", ""))
        for row in out_rows
        if row.get("event")
    }
    if (
        not pair_rows
        or len(out_rows) != len(pair_rows)
        or len(summaries) != len(expected_events)
        or progress_events != expected_events
        or output_events != expected_events
    ):
        raise ValueError(
            "GPT micro output coverage mismatch: "
            f"pairs={len(pair_rows)} outputs={len(out_rows)} "
            f"expected_events={len(expected_events)} "
            f"progress_events={len(progress_events)} "
            f"output_events={len(output_events)}"
        )
    progress_ok = all(
        row.get("mode") == mode
        and int(row.get("total", 0) or 0) > 0
        and int(row.get("completed", 0) or 0)
        == int(row.get("total", 0) or 0)
        and int(row.get("missing", 0) or 0) == 0
        for row in summaries
    )
    dimensions_ok = all(
        row.get(f"{side}_{dimension}", "") in LABELS[dimension]
        for row in out_rows
        for dimension in DIMENSIONS
        for side in ("real", "gen")
    )
    model_ok = all(
        row.get("llm_model") == expected_model for row in out_rows
    )
    expected_uids = [row_uid(mode, row) for row in pair_rows]
    output_uids = [str(row.get("row_uid", "")) for row in out_rows]
    uid_sets_match = (
        all(expected_uids)
        and len(set(expected_uids)) == len(expected_uids)
        and all(output_uids)
        and len(set(output_uids)) == len(output_uids)
        and set(output_uids) == set(expected_uids)
    )
    if (
        not progress_ok
        or not dimensions_ok
        or not model_ok
        or not uid_sets_match
    ):
        raise ValueError(
            "GPT micro output is incomplete despite zero-missing progress: "
            f"progress_ok={progress_ok} dimensions_ok={dimensions_ok} "
            f"model_ok={model_ok} uid_sets_match={uid_sets_match}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Independent semantic evaluation with strict labels and bounded retries.")
    ap.add_argument("--pairwise-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch-workers-per-event", type=int, default=1)
    ap.add_argument("--max-attempts", "--retries", dest="max_attempts", type=int, default=5,
                    help="Total requests per text across resume; --retries is a compatibility alias.")
    ap.add_argument("--backoff", type=float, default=2)
    ap.add_argument("--max-backoff", type=float, default=16)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--index-csv", default="data/index.csv")
    args = ap.parse_args()

    args.mode = "OpenSim"
    if min(args.batch_size, args.workers, args.batch_workers_per_event, args.max_attempts) < 1 or min(args.backoff, args.max_backoff) < 0:
        ap.error("batch sizes, workers and max-attempts must be positive; backoff must be nonnegative")
    return args


def run(args: argparse.Namespace) -> int:
    args.stop_event = threading.Event()
    raw_rows = read_csv_rows(Path(args.pairwise_csv))
    grouped_pairs: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        event = row.get("event", "")
        if event:
            grouped_pairs.setdefault(event, []).append(row)
    all_pair_rows: List[Dict[str, Any]] = []
    grouped_texts: Dict[str, List[Dict[str, Any]]] = {}
    for event, rows in grouped_pairs.items():
        pair_rows, text_rows = build_text_records(args.mode, rows)
        all_pair_rows.extend(pair_rows)
        grouped_texts[event] = text_rows

    meta = load_meta(Path(args.index_csv))
    print(
        f"mode={args.mode} model={args.model} events={len(grouped_texts)} "
        f"pairs={len(all_pair_rows)} unique_texts={sum(len(v) for v in grouped_texts.values())} "
        f"out={args.output_dir}",
        flush=True,
    )

    summaries: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(grouped_texts)))) as executor:
        futures = {
            executor.submit(evaluate_event, event, texts, args, meta): event
            for event, texts in grouped_texts.items()
        }
        for future in as_completed(futures):
            event = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                print(f"EVENT_FAILED {event}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                raise
            summaries.append(summary)
            print(f"SUMMARY {event}: {summary}", flush=True)
    summaries.sort(key=lambda x: x["event"])
    out_rows = combine(
        Path(args.output_dir),
        args.mode,
        all_pair_rows,
        summaries,
        args.model,
    )
    incomplete = [
        row
        for row in summaries
        if int(row.get("missing", 0) or 0) != 0
    ]
    if incomplete:
        details = ", ".join(
            f"{row.get('event', '')}:missing={row.get('missing', 0)}"
            for row in incomplete
        )
        print(
            f"INCOMPLETE {args.mode}: {details}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    validate_complete_output(
        args.mode,
        all_pair_rows,
        summaries,
        out_rows,
        args.model,
    )
    print(f"DONE {args.mode}: {Path(args.output_dir) / 'progress.csv'}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {
            "model": args.model, "labels": LABELS,
            "pairwise_sha256": hashlib.sha256(Path(args.pairwise_csv).read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256(Path(args.index_csv).read_bytes()).hexdigest() if Path(args.index_csv).exists() else None,
            "base_prompt_sha256": hashlib.sha256(inspect.getsource(build_prompt).encode()).hexdigest(),
            "constraints": "strict enums; explicit unsupported-schema API errors allow same-model JSON mode",
        }
        protocol_path = output / "semantic_protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Semantic inputs/model/prompt changed; use a new output directory")
        if not protocol_path.exists():
            protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

