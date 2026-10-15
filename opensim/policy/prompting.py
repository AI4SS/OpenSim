from __future__ import annotations

import re
from dataclasses import dataclass


POLICY_FORMAT_ACTION_CONTENT = "action_content"
POLICY_FORMATS = {POLICY_FORMAT_ACTION_CONTENT}

COMMENT_ACTION = "comment"
SHARE_ACTION = "share"
ACTION_CONTENT_PREFILL = "动作："
POLICY_CONTENT_CONTAMINATION_RE = re.compile(
    r"(?:"
    r"<\|im_(?:start|end)\|>"
    r"|(?:^|\n)\s*(?:human|assistant|system)\s*[:：]"
    r"|(?:^|\n)\s*(?:"
    r"全部评论|当前讨论的话题是|当天可见的事件帖子|"
    r"截至当前的历史舆论环境|网友资料|个体状态提示|"
    r"模拟内容输出|回答格式"
    r")\s*[:：]"
    r"|请总结以上内容"
    r")",
    flags=re.I,
)


def head_tail_truncate_prompt_ids(
    token_ids: list[int],
    max_length: int,
    *,
    head_fraction: float = 0.6,
) -> list[int]:
    """Match policy training's prompt-only head/tail truncation contract."""

    if max_length < 0:
        raise ValueError("max_length must be non-negative.")
    if not 0.0 <= head_fraction <= 1.0:
        raise ValueError("head_fraction must be between zero and one.")
    values = list(token_ids)
    if len(values) <= max_length:
        return values
    if max_length == 0:
        return []
    head_length = int(max_length * head_fraction)
    tail_length = max_length - head_length
    if tail_length == 0:
        return values[:head_length]
    return values[:head_length] + values[-tail_length:]


ACTION_CONTENT_INSTRUCTION = "\n".join(
    [
        "请先根据事件语境、历史舆论、个体状态和网友资料，自主判断该网友本次会评论还是转发。",
        "有观点、情绪、判断、疑问、支持、反对、补充或互动意图时选择评论，并写一条自然具体的回应。",
        "主要是扩散或保存信息且没有补充观点时选择转发；中立不等于必须转发。",
        "不要因为不确定就固定选择转发，也不要照抄事件原文或提示词。",
        "只输出两行，不要解释、编号或 JSON：",
        "第一行：动作：评论 或 动作：转发",
        "第二行：内容：该网友实际发布的文字",
    ]
)

ACTION_CONTENT_STATE_AGNOSTIC_INSTRUCTION = "\n".join(
    [
        "请根据事件语境、历史舆论和网友资料，自主判断该网友本次会评论还是转发。",
        "有观点、情绪、判断、疑问、支持、反对、补充或互动意图时选择评论，并写一条自然具体的回应。",
        "主要是扩散或保存信息且没有补充观点时选择转发；中立不等于必须转发。",
        "不要因为不确定就固定选择转发，也不要照抄事件原文或提示词。",
        "只输出两行，不要解释、编号或 JSON：",
        "第一行：动作：评论 或 动作：转发",
        "第二行：内容：该网友实际发布的文字",
    ]
)


def normalize_action(action: str) -> str:
    value = str(action or "").strip().lower()
    if value in {"share", "转发", "传播", "repost"}:
        return SHARE_ACTION
    return COMMENT_ACTION


def action_name_zh(action: str) -> str:
    return "转发" if normalize_action(action) == SHARE_ACTION else "评论"


def build_policy_prompt_content(
    *,
    topic: str,
    root_post: str,
    mean_field: str,
    profile: str,
    latent_descriptor: str = "",
    hot_comment: str = "",
    related_cases_info: str = "",
    policy_format: str = POLICY_FORMAT_ACTION_CONTENT,
    include_state_instruction: bool = True,
) -> str:
    if policy_format not in POLICY_FORMATS:
        raise ValueError(f"Unsupported policy format: {policy_format}")

    parts = [f"当前讨论的话题是：{topic}"]
    if root_post:
        parts.append(f"当天可见的事件帖子：{root_post}")
    if hot_comment and hot_comment != "暂无最新热门评论":
        parts.append(f"历史热门背景参考：{hot_comment}")
    if mean_field:
        parts.append(f"截至当前的历史舆论环境：{mean_field}")
    if related_cases_info:
        parts.append(f"当前时间与干预信息：{related_cases_info}")
    if latent_descriptor:
        parts.append(f"个体状态提示：{latent_descriptor}")
    parts.append(f"网友资料：{profile or '未知画像'}")
    if policy_format == POLICY_FORMAT_ACTION_CONTENT:
        parts.append(
            ACTION_CONTENT_INSTRUCTION
            if include_state_instruction
            else ACTION_CONTENT_STATE_AGNOSTIC_INSTRUCTION
        )
    return "\n".join(parts)


def wrap_qwen_chat(prompt_content: str) -> str:
    return (
        f"<|im_start|>user\n{prompt_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_policy_chat_prompt(**kwargs: str) -> str:
    return wrap_qwen_chat(build_policy_prompt_content(**kwargs))


def policy_generation_prompt(prompt: str, policy_format: str) -> str:
    """Add only the fixed output syntax; the policy still chooses the action."""

    if policy_format != POLICY_FORMAT_ACTION_CONTENT:
        raise ValueError(f"Unsupported policy format: {policy_format}")
    return f"{prompt}{ACTION_CONTENT_PREFILL}"


def complete_policy_generation(text: str, policy_format: str) -> str:
    """Restore the fixed prefix so parsing and critic scoring match SFT."""

    value = str(text or "").strip()
    if policy_format != POLICY_FORMAT_ACTION_CONTENT:
        raise ValueError(f"Unsupported policy format: {policy_format}")
    if re.match(r"^(?:动作|行为|行动|action)\s*[:：]", value, flags=re.I):
        return value
    value = value.lstrip(":：").lstrip()
    return f"{ACTION_CONTENT_PREFILL}{value}"


def policy_content_invalid_reason(content: str) -> str:
    """Return an auditable reason when generated social text leaks prompt syntax."""

    value = str(content or "").strip()
    if not value:
        return "empty"
    if POLICY_CONTENT_CONTAMINATION_RE.search(value):
        return "instruction_contamination"
    if len(value) > 280:
        return "excessive_length"
    if len([line for line in value.splitlines() if line.strip()]) > 6:
        return "excessive_lines"
    return ""


def format_policy_target(action: str, content: str) -> str:
    normalized_action = normalize_action(action)
    cleaned = str(content or "").strip()
    if normalized_action == SHARE_ACTION and not cleaned:
        cleaned = "转发微博"
    return f"动作：{action_name_zh(normalized_action)}\n内容：{cleaned}"


@dataclass(frozen=True)
class ParsedPolicyOutput:
    action: str
    content: str
    valid: bool
    raw_text: str


def _clean_content(value: str) -> str:
    text = str(value or "").strip().strip("`").strip()
    text = re.sub(r"^(?:内容|文本|content)\s*[:：]\s*", "", text, flags=re.I)
    return text.split("<|im_end|>", 1)[0].strip()


def parse_policy_output(text: str) -> ParsedPolicyOutput:
    raw = str(text or "").replace("\r", "\n").strip()
    action_match = re.search(
        r"(?:^|\n)\s*(?:动作|行为|行动|action)\s*[:：]\s*"
        r"(评论|转发|传播|comment|share)\s*(?:\n|$)",
        raw,
        flags=re.I,
    )
    content_match = re.search(
        r"(?:^|\n)\s*(?:内容|文本|content)\s*[:：]\s*(.+)",
        raw,
        flags=re.I | re.S,
    )
    if action_match:
        action = normalize_action(action_match.group(1))
        content = _clean_content(content_match.group(1) if content_match else "")
        if action == SHARE_ACTION and not content:
            content = "转发微博"
        return ParsedPolicyOutput(
            action=action,
            content=content,
            valid=bool(content),
            raw_text=raw,
        )

    cleaned = _clean_content(raw.splitlines()[0] if raw else "")
    compact = re.sub(r"[\s。！!，,：:；;\"'“”‘’]", "", cleaned)
    action = (
        SHARE_ACTION
        if compact in {"转发", "转发微博", "分享", "已转发"}
        else COMMENT_ACTION
    )
    return ParsedPolicyOutput(
        action=action,
        content=cleaned,
        valid=False,
        raw_text=raw,
    )
