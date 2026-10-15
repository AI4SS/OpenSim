"""Strict semantic responses and a persisted, per-text request budget."""
from __future__ import annotations

import math
import time
from typing import Any, Callable


def response_format(uids: list[str], labels: dict[str, list[str]]) -> dict:
    label_schema = {
        "type": "object",
        "properties": {dimension: {"type": "string", "enum": values} for dimension, values in labels.items()},
        "required": list(labels), "additionalProperties": False,
    }
    properties = {
        "text_id": {"type": "string", "enum": uids}, "labels": label_schema,
        "keywords": {"type": "array", "items": {"type": "string"}},
        "is_invalid": {"type": "boolean"}, "invalid_reason": {"type": "string"},
        "fluency_score": {"type": "number"}, "short_reason": {"type": "string"},
    }
    item = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {
        "name": "semantic_labels", "strict": True,
        "schema": {"type": "object", "properties": {"items": {"type": "array", "items": item}},
                   "required": ["items"], "additionalProperties": False},
    }}


def validate_item(item: Any, uid: str, labels: dict[str, list[str]]) -> None:
    if not isinstance(item, dict) or item.get("text_id") != uid:
        raise ValueError("text_id does not exactly match the requested UID")
    actual = item.get("labels")
    if not isinstance(actual, dict) or set(actual) != set(labels):
        raise ValueError("All eight label fields are required")
    for dimension, values in labels.items():
        if not isinstance(actual[dimension], str) or actual[dimension] not in values:
            raise ValueError(f"Illegal or empty label: {dimension}")
    if not isinstance(item.get("is_invalid"), bool) or not isinstance(item.get("invalid_reason"), str):
        raise ValueError("Invalid generation metadata")
    if not isinstance(item.get("keywords"), list) or any(not isinstance(word, str) for word in item["keywords"]):
        raise ValueError("Invalid keywords")
    value = item.get("fluency_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 5:
        raise ValueError("Invalid fluency score")
    if not isinstance(item.get("short_reason"), str) or not item["short_reason"].strip():
        raise ValueError("Empty judgment reason")


def partition_items(items: Any, expected: set[str], labels: dict[str, list[str]]) -> tuple[dict, dict]:
    if not isinstance(items, list):
        return {}, {uid: "Missing items list" for uid in expected}
    ids = [item.get("text_id") if isinstance(item, dict) else None for item in items]
    if any(not isinstance(uid, str) or uid not in expected for uid in ids) or len(set(ids)) != len(ids):
        return {}, {uid: "Unexpected, duplicated, or nonexact UID" for uid in expected}
    accepted, errors = {}, {uid: "Missing result item" for uid in expected}
    for item in items:
        uid = item["text_id"]
        try:
            validate_item(item, uid, labels)
        except ValueError as exc:
            errors[uid] = str(exc)
        else:
            accepted[uid] = item
            errors.pop(uid)
    return accepted, errors


def schema_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return (getattr(exc, "status_code", None) in (400, 404, 415, 422, 501)
            and any(word in message for word in ("unsupported", "not supported"))
            and any(word in message for word in ("json_schema", "response_format", "structured output", "schema")))


def transient_error(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    return (code in (408, 409, 429) or isinstance(code, int) and code >= 500
            or isinstance(exc, (TimeoutError, ConnectionError))
            or type(exc).__name__ in ("APIConnectionError", "APITimeoutError"))


def judge_batch(
    rows: list[dict], *, model: str, labels: dict[str, list[str]], prompt: Callable,
    parse: Callable, request: Callable, accept: Callable, evidence: Callable,
    attempts: dict[str, int], strategies: dict[str, str], max_attempts: int = 5,
    timeout: float = 120, max_tokens: int = 4096, backoff: float = 2,
    max_backoff: float = 16, sleep: Callable = time.sleep, cancelled: Callable | None = None,
) -> None:
    """Each request consumes one attempt for every included text, even on API failure."""
    pending = {row["text_uid"]: row for row in rows}
    while pending:
        if cancelled is not None and cancelled():
            return
        eligible = [row for uid, row in pending.items() if attempts.get(uid, 0) < max_attempts]
        if not eligible:
            return
        strategy = strategies.get(eligible[0]["text_uid"], "json_schema")
        batch = [row for row in eligible if strategies.get(row["text_uid"], "json_schema") == strategy]
        uids = [row["text_uid"] for row in batch]
        for uid in uids:
            attempts[uid] = attempts.get(uid, 0) + 1
        original = prompt(batch)
        appendix = ("\n\n补充输出约束：只返回本次全部 text_id，每个完整且仅出现一次。"
                    "labels 必须包含八个字段，各自严格从上方允许集合选择一个值，"
                    "不能留空、输出 null 或其他维度的标签。此约束不改变上方判定规则。\n")
        payload = {"model": model, "messages": [{"role": "user", "content": original + appendix}],
                   "temperature": 0, "max_tokens": max_tokens, "timeout": timeout,
                   "response_format": response_format(uids, labels) if strategy == "json_schema" else {"type": "json_object"}}
        trace = {"status": "request_started", "time": time.time(), "text_uids": uids,
                 "attempts": {uid: attempts[uid] for uid in uids}, "strategy": strategy,
                 "payload": payload, "base_prompt": original, "enum_appendix": appendix}
        evidence(trace)
        try:
            response = request(payload)
        except Exception as exc:
            unsupported = strategy == "json_schema" and schema_unsupported(exc)
            if unsupported:
                for uid in uids:
                    strategies[uid] = "json_object"
            trace.update(status="request_failed", error=f"{type(exc).__name__}: {exc}",
                         status_code=getattr(exc, "status_code", None), completed_at=time.time(),
                         strategy_after="json_object" if unsupported else strategy,
                         retryable=unsupported or transient_error(exc))
            evidence(trace)
            if not trace["retryable"]:
                raise
        else:
            trace["response"] = response
            trace["status"] = "response_received"
            evidence(trace)
            if response.get("refusal") or response.get("finish_reason") != "stop":
                accepted, errors = {}, {uid: "Refusal or incomplete response" for uid in uids}
            else:
                parsed = parse(response.get("content") or "")
                accepted, errors = partition_items(parsed.get("items") if isinstance(parsed, dict) else None, set(uids), labels)
            for uid, item in accepted.items():
                accept(pending.pop(uid), item)
            trace.update(status="response_validated", accepted=list(accepted), invalid=errors, completed_at=time.time())
            evidence(trace)
        remaining = [uid for uid in uids if uid in pending and attempts[uid] < max_attempts]
        if remaining:
            sleep(min(backoff * 2 ** (min(attempts[uid] for uid in remaining) - 1), max_backoff))
