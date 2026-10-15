from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


from opensim.policy.prompting import POLICY_FORMAT_ACTION_CONTENT, action_name_zh, normalize_action


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {(key or "").lstrip("\ufeff"): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event",
        "time_step",
        "row_uid",
        "source_record_id",
        "source_content_type",
        "nll",
        "target_tokens",
        "content_nll",
        "content_tokens",
        "action_nll",
        "action_tokens",
        "response_nll",
        "response_tokens",
        "target_format",
        "nll_protocol",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class PolicyBundle:
    def __init__(self, adapter_path: str, device: str | None = None, base_model: str | None = None):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = Path(adapter_path)
        peft_config = PeftConfig.from_pretrained(
            str(adapter),
            local_files_only=True,
        )
        base_path = base_model or peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(adapter),
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        target_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = torch.float16 if target_device.type == 'cuda' else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(
            base,
            str(adapter),
            local_files_only=True,
        )
        self.model.to(target_device).eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def encode_example(
    tokenizer: Any,
    prompt: str,
    action: str,
    content: str,
    max_length: int,
) -> tuple[list[int], list[int], list[bool], list[bool]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    prefix_ids = tokenizer(
        "动作：",
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    action_ids = tokenizer(
        action_name_zh(action),
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    content_prefix_ids = tokenizer(
        "\n内容：",
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    content_ids = tokenizer(
        f"{content}<|im_end|>",
        add_special_tokens=False,
        truncation=False,
    ).input_ids
    target_ids = prefix_ids + action_ids + content_prefix_ids + content_ids
    action_mask = (
        [False] * len(prefix_ids)
        + [True] * len(action_ids)
        + [False] * (len(content_prefix_ids) + len(content_ids))
    )
    content_mask = (
        [False] * (len(prefix_ids) + len(action_ids) + len(content_prefix_ids))
        + [True] * len(content_ids)
    )
    if len(target_ids) >= max_length:
        target_ids = target_ids[:max_length]
        action_mask = action_mask[:max_length]
        content_mask = content_mask[:max_length]
        prompt_ids = []
    else:
        prompt_budget = max_length - len(target_ids)
        if len(prompt_ids) > prompt_budget:
            head_budget = int(prompt_budget * 0.6)
            tail_budget = prompt_budget - head_budget
            prompt_ids = prompt_ids[:head_budget] + prompt_ids[-tail_budget:]
    return (
        prompt_ids + target_ids,
        [-100] * len(prompt_ids) + target_ids,
        [False] * len(prompt_ids) + action_mask,
        [False] * len(prompt_ids) + content_mask,
    )


def score_batch(
    bundle: PolicyBundle,
    examples: list[tuple[str, str, str]],
    max_length: int,
) -> list[dict[str, float | int]]:
    import torch

    encoded = [
        encode_example(
            bundle.tokenizer,
            prompt,
            action,
            content,
            max_length,
        )
        for prompt, action, content in examples
    ]
    width = max(len(input_ids) for input_ids, *_ in encoded)
    pad_id = int(bundle.tokenizer.pad_token_id)
    input_ids = torch.full(
        (len(encoded), width),
        pad_id,
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    content_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_idx, (
        row_ids,
        row_labels,
        row_action_mask,
        row_content_mask,
    ) in enumerate(encoded):
        length = len(row_ids)
        input_ids[row_idx, :length] = torch.tensor(
            row_ids,
            dtype=torch.long,
            device=bundle.device,
        )
        attention_mask[row_idx, :length] = 1
        labels[row_idx, :length] = torch.tensor(
            row_labels,
            dtype=torch.long,
            device=bundle.device,
        )
        action_mask[row_idx, :length] = torch.tensor(
            row_action_mask,
            dtype=torch.bool,
            device=bundle.device,
        )
        content_mask[row_idx, :length] = torch.tensor(
            row_content_mask,
            dtype=torch.bool,
            device=bundle.device,
        )
    with torch.inference_mode():
        logits = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    response_valid = shift_labels.ne(-100)
    action_valid = action_mask[:, 1:] & response_valid
    content_valid = content_mask[:, 1:] & response_valid

    def summarize(
        row_loss: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> tuple[float, int]:
        token_count = int(row_mask.sum().item())
        nll = float(
            (row_loss * row_mask).sum().item() / max(token_count, 1)
        )
        return nll, token_count

    scores = []
    for row_loss, row_response, row_action, row_content in zip(
        losses,
        response_valid,
        action_valid,
        content_valid,
    ):
        response_nll, response_tokens = summarize(row_loss, row_response)
        action_nll, action_tokens = summarize(row_loss, row_action)
        content_nll, content_tokens = summarize(row_loss, row_content)
        scores.append(
            {
                "nll": content_nll,
                "target_tokens": content_tokens,
                "content_nll": content_nll,
                "content_tokens": content_tokens,
                "action_nll": action_nll,
                "action_tokens": action_tokens,
                "response_nll": response_nll,
                "response_tokens": response_tokens,
            }
        )
    return scores


def row_uid(row: dict[str, Any]) -> str:
    existing = clean_text(row.get("row_uid"))
    if existing:
        return existing
    return "|".join(
        [
            clean_text(row.get("event")),
            clean_text(row.get("time_step")),
            clean_text(row.get("source_record_id")),
            clean_text(row.get("agent_index")),
        ]
    )


def target_from_row(row: dict[str, Any]) -> tuple[str, str]:
    action = normalize_action(clean_text(row.get("source_content_type")))
    content = clean_text(
        row.get("source_text") or row.get("real_comment")
    )
    if not content:
        content = "转发微博" if action == "share" else "无"
    return action, content


def validate_scored_rows(
    input_rows: list[dict[str, Any]],
    scored_rows: dict[str, dict[str, Any]],
) -> None:
    expected_uids = [row_uid(row) for row in input_rows]
    if not expected_uids:
        raise ValueError("Teacher NLL input is empty.")
    if len(set(expected_uids)) != len(expected_uids):
        raise ValueError("Teacher NLL row_uid values are not unique.")
    if set(scored_rows) != set(expected_uids):
        raise ValueError(
            "Teacher NLL coverage mismatch: "
            f"expected={len(expected_uids)} scored={len(scored_rows)}"
        )
    invalid = []
    for uid in expected_uids:
        row = scored_rows[uid]
        try:
            nll = float(row.get("content_nll", "nan"))
            tokens = int(float(row.get("content_tokens", 0) or 0))
        except (TypeError, ValueError):
            nll = math.nan
            tokens = 0
        if (
            not math.isfinite(nll)
            or tokens <= 0
            or row.get("target_format") != POLICY_FORMAT_ACTION_CONTENT
            or "content tokens only" not in row.get("nll_protocol", "")
        ):
            invalid.append(uid)
    if invalid:
        raise ValueError(
            f"Teacher NLL has {len(invalid)} invalid scored rows."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score observed content using exact recorded policy prompts.")
    parser.add_argument("--pairwise-csv", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-model", help="Override the adapter's recorded base model without changing its metadata.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 2:
        parser.error("batch-size must be positive and max-length must be at least 2")
    rows = read_rows(Path(args.pairwise_csv))
    if not rows or any(not clean_text(row.get("policy_prompt")) for row in rows):
        raise ValueError("Teacher NLL requires nonempty rows with exact policy_prompt.")
    expected_uids = [row_uid(row) for row in rows]
    if len(set(expected_uids)) != len(rows):
        raise ValueError("Teacher NLL row_uid values are not unique.")
    output_path = Path(args.output_csv)
    existing = (
        {row_uid(row): row for row in read_rows(output_path)}
        if args.resume and output_path.exists() else {}
    )
    existing = {uid: row for uid, row in existing.items() if uid in set(expected_uids)}
    pending = [row for row in rows if row_uid(row) not in existing]
    bundle = PolicyBundle(args.adapter_path, args.device, args.base_model) if pending else None
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        examples = [(clean_text(row["policy_prompt"]), *target_from_row(row)) for row in batch]
        scores = score_batch(bundle, examples, args.max_length)
        for row, score in zip(batch, scores):
            uid = row_uid(row)
            existing[uid] = {
                "event": row.get("event", ""),
                "time_step": row.get("time_step", ""),
                "row_uid": uid,
                "source_record_id": row.get("source_record_id", ""),
                "source_content_type": row.get("source_content_type", ""),
                **score,
                "target_format": POLICY_FORMAT_ACTION_CONTENT,
                "nll_protocol": (
                    "content tokens only, conditioned on the exact causal "
                    "prompt and teacher-forced structured action prefix"
                ),
            }
        if (start // args.batch_size + 1) % 50 == 0 or start + args.batch_size >= len(pending):
            write_rows(output_path, list(existing.values()))
            print(f"Scored {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)
    validate_scored_rows(rows, existing)
    write_rows(output_path, [existing[uid] for uid in expected_uids])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

