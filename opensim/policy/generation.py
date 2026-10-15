from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from opensim.policy.prompting import head_tail_truncate_prompt_ids


def tokenize_policy_generation_prompts(
    tokenizer,
    prompts: list[str],
    *,
    max_length: int = 624,
):
    """Tokenize generation prompts with the same head/tail contract as SFT."""

    encoded_prompts = []
    for prompt in prompts:
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        prompt_ids = head_tail_truncate_prompt_ids(
            prompt_ids,
            max_length,
        )
        encoded_prompts.append(
            {
                "input_ids": prompt_ids,
                "attention_mask": [1] * len(prompt_ids),
            }
        )
    return tokenizer.pad(
        encoded_prompts,
        padding=True,
        return_tensors="pt",
    )


def _content_boundary_token_ids(
    tokenizer,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cached_linebreaks = getattr(
        tokenizer,
        "_mfmdp_linebreak_token_ids",
        None,
    )
    cached_blanks = getattr(
        tokenizer,
        "_mfmdp_blank_token_ids",
        None,
    )
    if cached_linebreaks is not None and cached_blanks is not None:
        return tuple(cached_linebreaks), tuple(cached_blanks)
    linebreak_ids = []
    blank_ids = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\n" in decoded or "\r" in decoded:
            linebreak_ids.append(token_id)
        if not decoded or not decoded.strip():
            blank_ids.append(token_id)
    if not linebreak_ids:
        raise ValueError("Tokenizer has no decodable line-break token.")
    cached_linebreaks = tuple(linebreak_ids)
    cached_blanks = tuple(blank_ids)
    setattr(
        tokenizer,
        "_mfmdp_linebreak_token_ids",
        cached_linebreaks,
    )
    setattr(tokenizer, "_mfmdp_blank_token_ids", cached_blanks)
    return cached_linebreaks, cached_blanks


class ActionContentLogitsProcessor(LogitsProcessor):
    """Constrain only the fixed grammar around the autonomous action token."""

    def __init__(
        self,
        *,
        prompt_width: int,
        action_token_ids: tuple[int, int],
        suffix_token_ids: tuple[int, ...],
        newline_token_ids: tuple[int, ...],
        eos_token_ids: tuple[int, ...],
        initial_blocked_token_ids: tuple[int, ...],
        max_content_tokens: int = 48,
    ) -> None:
        self.prompt_width = int(prompt_width)
        self.action_token_ids = tuple(int(value) for value in action_token_ids)
        self.suffix_token_ids = tuple(int(value) for value in suffix_token_ids)
        self.newline_token_ids = tuple(
            int(value) for value in newline_token_ids
        )
        self.eos_token_ids = tuple(int(value) for value in eos_token_ids)
        self.initial_blocked_token_ids = tuple(
            int(value) for value in initial_blocked_token_ids
        )
        self.max_content_tokens = int(max_content_tokens)
        if not self.newline_token_ids:
            raise ValueError("Structured content decoding needs a newline token.")
        if not self.eos_token_ids:
            raise ValueError("Structured content decoding needs an EOS token.")
        if self.max_content_tokens <= 0:
            raise ValueError("max_content_tokens must be positive.")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_count = int(input_ids.shape[1]) - self.prompt_width
        if generated_count == 0:
            allowed = self.action_token_ids
        elif 1 <= generated_count <= len(self.suffix_token_ids):
            allowed = (self.suffix_token_ids[generated_count - 1],)
        else:
            content_start = (
                self.prompt_width + 1 + len(self.suffix_token_ids)
            )
            content_ids = input_ids[:, content_start:]
            if content_ids.shape[1] == 0:
                constrained = scores.clone()
                constrained[
                    :,
                    list(self.initial_blocked_token_ids),
                ] = float("-inf")
                return constrained
            stop_rows = torch.zeros(
                input_ids.shape[0],
                dtype=torch.bool,
                device=input_ids.device,
            )
            if content_ids.shape[1] >= self.max_content_tokens:
                stop_rows.fill_(True)
            elif content_ids.numel():
                for token_id in self.newline_token_ids:
                    stop_rows |= content_ids.eq(token_id).any(dim=1)
            if not bool(stop_rows.any()):
                return scores
            constrained = scores.clone()
            constrained[stop_rows] = float("-inf")
            for token_id in self.eos_token_ids:
                constrained[stop_rows, token_id] = scores[
                    stop_rows,
                    token_id,
                ]
            return constrained
        constrained = torch.full_like(scores, float("-inf"))
        constrained[:, list(allowed)] = scores[:, list(allowed)]
        return constrained


def action_content_logits_processors(
    tokenizer,
    *,
    prompt_width: int,
    max_content_tokens: int = 48,
) -> LogitsProcessorList:
    comment_ids = tokenizer.encode("评论", add_special_tokens=False)
    share_ids = tokenizer.encode("转发", add_special_tokens=False)
    if len(comment_ids) != 1 or len(share_ids) != 1:
        raise ValueError(
            "Structured action decoding requires one token per action label."
        )
    suffix_ids = tokenizer.encode("\n内容：", add_special_tokens=False)
    if not suffix_ids:
        raise ValueError("Structured action decoding suffix cannot be empty.")
    newline_ids, blank_ids = _content_boundary_token_ids(tokenizer)
    raw_eos_ids = tokenizer.eos_token_id
    eos_ids = (
        tuple(raw_eos_ids)
        if isinstance(raw_eos_ids, (list, tuple))
        else (int(raw_eos_ids),)
        if raw_eos_ids is not None
        else ()
    )
    initial_blocked_ids = tuple(
        sorted(set(newline_ids) | set(blank_ids) | set(eos_ids))
    )
    return LogitsProcessorList(
        [
            ActionContentLogitsProcessor(
                prompt_width=prompt_width,
                action_token_ids=(comment_ids[0], share_ids[0]),
                suffix_token_ids=tuple(suffix_ids),
                newline_token_ids=tuple(newline_ids),
                eos_token_ids=eos_ids,
                initial_blocked_token_ids=initial_blocked_ids,
                max_content_tokens=max_content_tokens,
            )
        ]
    )
