"""Local Qwen3 mean-field summarization with the established Chinese prompt."""
from __future__ import annotations
import re
import torch

def clean_generated_text(text: str) -> str:
    """
    清理模型生成的文本，移除不相关的英文内容和数学公式。

    Args:
        text: 原始生成的文本

    Returns:
        清理后的文本（只保留中文相关内容）
    """
    if not text:
        return text
    lines = text.split('\n')
    cleaned_lines = []
    skip_mode = False
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if not skip_mode:
                cleaned_lines.append(line)
            continue
        english_chars = sum((1 for c in line_stripped if c.isalpha() and ord(c) < 128))
        total_chars = sum((1 for c in line_stripped if c.isalpha()))
        if total_chars > 0 and english_chars / total_chars > 0.7:
            skip_mode = True
            continue
        math_keywords = ['$', '\\frac', '^', '{', '}', '\\\\', 'sqrt', 'sum', 'int']
        if any((keyword in line for keyword in math_keywords)):
            skip_mode = True
            continue
        if line_stripped.startswith(('If ', 'When ', 'Given that', 'Let ')):
            skip_mode = True
            continue
        if re.search('\\b(What|How|Why|Where|When|Find|Calculate|Solve)\\s+\\w+', line_stripped):
            skip_mode = True
            continue
        if not skip_mode:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1].strip():
            cleaned_lines.append('')
    cleaned_text = '\n'.join(cleaned_lines)
    math_patterns = ['If \\$[a-z]+\\$.*?\\n', 'Given that.*?\\n', 'Let the.*?\\n', 'To find the value of.*?\\n', '\\$[^\\$]*\\$.*?\\n', 'If we know.*?\\n']
    for pattern in math_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    summary_end_pattern = '(综上所述[，。].*?)(\\n\\s*综上所述[，。].*?)+'
    cleaned_text = re.sub(summary_end_pattern, '\\1', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub('\\n\\s*\\n\\s*\\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def model_generate(prompt, model, tokenizer, device):
    if tokenizer is None:
        raise RuntimeError('Summarizer tokenizer is required')
    messages = [{'role': 'system', 'content': '你是一个专业的舆论分析助手。请根据提供的信息总结评论分布。'}, {'role': 'user', 'content': prompt}]
    tokenizer_name = str(getattr(tokenizer, 'name_or_path', '')).lower()
    chat_template = str(getattr(tokenizer, 'chat_template', '') or '')
    template_kwargs = {'enable_thinking': False} if 'qwen3' in tokenizer_name or 'enable_thinking' in chat_template else {}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    inputs = tokenizer([text], return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=300, do_sample=True, temperature=0.1, top_p=0.9, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
    gen_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    summary = gen_text.strip()
    if not summary or '<think>' in summary.lower() or '</think>' in summary.lower():
        raise RuntimeError('Qwen3 summary is empty or contains reasoning tags')
    return summary

def build_summary_prompt(topic, states, actions, pre_mean_field, state_distribution):
    """
    使用 LLM 或 GPT 计算指定索引的 mean field。
    """
    state_action_prompt = ''
    for i, (s_prev, a_prev) in enumerate(zip(states, actions)):
        state_action_prompt += f'第{i + 1}个网友评论：{a_prev}\n'
    dist_info_str = ''
    if state_distribution is not None:
        pos, neu, neg = [f'{x * 100:.1f}%' for x in state_distribution]
        dist_info_str = f'【群体情绪统计分布数据】：积极占比 {pos}，中立占比 {neu}，消极占比 {neg}。\n'
        dist_info_str += '请注意：上述统计数据是由社会动力学模型计算得出的群体真实现状，你的总结必须与此数据分布保持一致。\n'
    end_prompt = '请你结合以上信息，总结网友的评论分布情况，方便后续网友快速了解已有讨论内容，对以下 6 个方面，根据重要性的顺序进行回答：1. 立场分布：大部分网友们是持有反对意见，还是支持？2. 观点分布：网友们有哪些观点？请总结主要的观点内容3. 情绪分布：大部分网友是生气、兴奋、质疑、焦虑?是积极倾向、消极，还是中立？4. 行为分布：他们更倾向于转发，还是评论？5. 话题真实性：网友们如何评价该话题的真实性？网友们是相信，还是质疑？6. 发言意图：网友的评论主要是提问、发表个人观点，还是转播信息？任务：请使用 200 字左右的中文总结回答，并确保信息结构清晰、简洁明了，严格按照这6个方面进行回答。'
    prompt = f'当前话题：{topic}\n过往网友评论总结：{pre_mean_field}。\n{dist_info_str}最新网友讨论情况：\n{state_action_prompt}\n' + end_prompt
    return prompt


def load_summarizer(checkpoint, base_model, device):
    from pathlib import Path
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    checkpoint = Path(checkpoint)
    if not (checkpoint / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Summarizer adapter is missing: {checkpoint}')
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        local_files_only=True,
    ).to(device)
    return PeftModel.from_pretrained(model, str(checkpoint), local_files_only=True).to(device).eval()


def summarize(topic, profiles, actions, previous_summary, cumulative_state,
              model, tokenizer, device):
    prompt = build_summary_prompt(topic, profiles, actions, previous_summary, cumulative_state)
    summary = clean_generated_text(model_generate(prompt, model, tokenizer, device))
    if not summary:
        raise RuntimeError('Summarizer output is empty after text cleanup')
    return summary
