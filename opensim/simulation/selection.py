"""Response-sensitive candidate scoring; no factual-label target is used."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from opensim.policy.critic import FutureTrajectoryCritic
from opensim.policy.model import extract_policy_features
from opensim.policy.prompting import (
    POLICY_FORMAT_ACTION_CONTENT, complete_policy_generation,
    head_tail_truncate_prompt_ids, parse_policy_output,
    policy_content_invalid_reason, policy_generation_prompt,
)
from opensim.policy.generation import (
    action_content_logits_processors, tokenize_policy_generation_prompts,
)

def load_policy(adapter_path: str | Path, device: torch.device, *, base_model: str | None=None) -> tuple[Any, Any]:
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapter_path = Path(adapter_path)
    peft_config = PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    base_path = base_model or peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True, fix_mistral_regex=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    model = PeftModel.from_pretrained(base_model, str(adapter_path), local_files_only=True).to(device)
    model.eval()
    model._opensim_policy_adapter_path = str(adapter_path)
    format_path = adapter_path / 'policy_format.json'
    policy_metadata = json.loads(format_path.read_text(encoding='utf-8')) if format_path.exists() else {}
    model._opensim_policy_format = policy_metadata.get('policy_format', 'unsupported')
    if model._opensim_policy_format == 'action_content_v1':
        model._opensim_policy_format = POLICY_FORMAT_ACTION_CONTENT
    model._opensim_latent_clusterer = None
    latent_path = adapter_path / 'latent_kmeans.joblib'
    if latent_path.exists():
        import joblib
        model._opensim_latent_clusterer = joblib.load(latent_path)
    model._opensim_trajectory_critic = None
    model._opensim_text_sensitive = False
    critic_path = adapter_path / 'trajectory_critic.pt'
    state_path = adapter_path / 'trainer_state.pt'
    if critic_path.exists():
        critic_state = torch.load(critic_path, map_location='cpu', weights_only=True)
        fusion_weight = critic_state['fusion_in.weight']
        critic_config_path = adapter_path / 'trajectory_critic_config.json'
        critic_config = json.loads(critic_config_path.read_text(encoding='utf-8')) if critic_config_path.exists() else {}
        critic_hidden_size = int(critic_config.get('critic_hidden_size', fusion_weight.shape[0]))
        llm_hidden_size = int(critic_config.get('llm_hidden_size', base_model.config.hidden_size))
        latent_dim = int(critic_config.get('latent_dim', fusion_weight.shape[1] - 2 * llm_hidden_size - 3))
        critic = FutureTrajectoryCritic(llm_hidden_size=llm_hidden_size, latent_dim=latent_dim, critic_hidden_size=critic_hidden_size, k_steps=int(critic_config.get('k_steps', 10)), dropout=float(critic_config.get('dropout', 0.2)))
        critic.load_state_dict(critic_state)
        model._opensim_trajectory_critic = critic.to(device).eval()
        if state_path.exists():
            trainer_state = torch.load(state_path, map_location='cpu', weights_only=False)
            metrics = trainer_state.get('metrics') or {}
            model._opensim_text_sensitive = bool(metrics.get('text_sensitive'))
    if model._opensim_policy_format != POLICY_FORMAT_ACTION_CONTENT:
        raise RuntimeError('Action/content policy checkpoint is required')
    if model._opensim_trajectory_critic is None or not model._opensim_text_sensitive:
        raise RuntimeError('A trained text-sensitive trajectory critic is required')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (tokenizer, model)

def _candidate_feature_batch(prompts, responses, tokenizer, max_length=704):
    rows = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
        response_ids = tokenizer(f'{response}<|im_end|>', add_special_tokens=False, truncation=False).input_ids
        if len(response_ids) >= max_length:
            response_ids = response_ids[:max_length]
            prompt_ids = []
        else:
            prompt_budget = max_length - len(response_ids)
            prompt_ids = head_tail_truncate_prompt_ids(prompt_ids, prompt_budget)
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        rows.append((input_ids, labels))
    width = max((len(ids) for ids, _ in rows))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for row_idx, (ids, row_labels) in enumerate(rows):
        length = len(ids)
        input_ids[row_idx, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, :length] = 1
        labels[row_idx, :length] = torch.tensor(row_labels, dtype=torch.long)
    return (input_ids, attention_mask, labels)

def _score_policy_candidates(*, prompts, responses, tokenizer, a_model, device, current_states, state_latents, target_trajectory, mc_samples=4, score_batch_size=16):
    critic = getattr(a_model, '_opensim_trajectory_critic', None)
    if critic is None or target_trajectory is None:
        raise RuntimeError('Trained trajectory critic and predicted trajectory are required')
    all_scores = []
    target_array = np.asarray(target_trajectory, dtype=np.float32)
    for start in range(0, len(responses), score_batch_size):
        end = min(start + score_batch_size, len(responses))
        input_ids, attention_mask, labels = _candidate_feature_batch(prompts[start:end], responses[start:end], tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = a_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            prompt_feature, response_feature = extract_policy_features(last_hidden_state, labels)
            current = torch.tensor(current_states[start:end], dtype=torch.float32, device=device)
            latent = torch.tensor(state_latents[start:end], dtype=torch.float32, device=device)
            samples = critic(prompt_feature=prompt_feature, response_feature=response_feature, current_state=current, st_latent=latent, mc_samples=mc_samples, mc_dropout=True)
        k_steps = samples.shape[2]
        valid_steps = min(k_steps, int(target_array.shape[0]))
        target = current.unsqueeze(1).expand(-1, k_steps, -1).clone()
        if valid_steps:
            observed = torch.tensor(target_array[:valid_steps], dtype=torch.float32, device=device)
            target[:, :valid_steps] = observed.unsqueeze(0)
        mask = torch.zeros((len(responses[start:end]), k_steps), device=device)
        mask[:, :valid_steps] = 1.0
        mean_prediction = samples.mean(dim=0).clamp_min(1e-07)
        target = target.clamp_min(1e-07)
        kl = (target * (target.log() - mean_prediction.log())).sum(dim=-1)
        distance = (kl * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        uncertainty = samples.var(dim=0, unbiased=False).mean(dim=(1, 2))
        all_scores.extend((distance + 0.1 * uncertainty).cpu().tolist())
    return all_scores


def generate_policy_batch(prompts, tokenizer, model, device, *, current_states,
                          state_latents, target_trajectory, num_candidates=4,
                          temperature=0.8, top_p=0.9, max_content_tokens=48):
    """Sample candidates and retain the smallest predicted trajectory score."""
    if num_candidates < 1:
        raise ValueError('At least one policy candidate is required')
    if not getattr(model, '_opensim_text_sensitive', False):
        raise RuntimeError('The policy trajectory critic has not passed its text-sensitivity gate')
    policy_format = model._opensim_policy_format
    generation_prompts = [policy_generation_prompt(p, policy_format) for p in prompts]
    inputs = tokenize_policy_generation_prompts(
        tokenizer, generation_prompts, max_length=624,
    ).to(device)
    width = inputs['input_ids'].shape[1]
    with torch.no_grad():
        generated = model.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_new_tokens=80, do_sample=True, temperature=temperature,
            top_p=top_p, repetition_penalty=1.05,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=action_content_logits_processors(
                tokenizer, prompt_width=width, max_content_tokens=max_content_tokens,
            ),
        )
    raw_candidates = [complete_policy_generation(
        tokenizer.decode(sequence[width:], skip_special_tokens=True).strip(),
        policy_format,
    ) for sequence in generated]
    if len(raw_candidates) != len(prompts) * num_candidates:
        raise RuntimeError('Policy generation returned an unexpected number of candidates')
    scores = _score_policy_candidates(
        prompts=[p for p in prompts for _ in range(num_candidates)],
        responses=raw_candidates, tokenizer=tokenizer, a_model=model, device=device,
        current_states=[s for s in current_states for _ in range(num_candidates)],
        state_latents=[s for s in state_latents for _ in range(num_candidates)],
        target_trajectory=target_trajectory,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError('Trajectory candidate scores are not finite')
    selected = []
    for row_idx in range(len(prompts)):
        candidates, seen = [], set()
        for offset in range(num_candidates):
            index = row_idx * num_candidates + offset
            parsed = parse_policy_output(raw_candidates[index])
            duplicate = parsed.content in seen
            seen.add(parsed.content)
            reason = policy_content_invalid_reason(parsed.content)
            penalty = (5.0 if not parsed.content else 0.0)
            penalty += 1.0 if not parsed.valid else 0.0
            penalty += 8.0 if reason and reason != 'empty' else 0.0
            penalty += 0.5 if duplicate else 0.0
            candidates.append({
                'action': parsed.action, 'content': parsed.content,
                'raw_text': parsed.raw_text, 'valid': parsed.valid,
                'invalid_reason': reason, 'trajectory_score': float(scores[index]),
                'sft_log_likelihood': None, 'selector_mode': 'trajectory_critic',
                'quality_penalty': penalty,
                'score': float(scores[index]) + 0.05 * penalty,
                'generation_index': offset,
            })
        winner = dict(min(candidates, key=lambda c: c['score']))
        winner.update(candidate_count=num_candidates, all_candidates=candidates)
        selected.append(winner)
    return selected


def response_nll(model, tokenizer, prompts, actions, device, max_length=600):
    """Teacher-forced response loss, recorded separately from generation."""
    from torch.nn.utils.rnn import pad_sequence

    rows, labels, lengths = [], [], []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    try:
        for prompt, action in zip(prompts, actions):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            action_ids = tokenizer.encode(action if action and action.strip() else '转发微博',
                                          add_special_tokens=False)[:max_length - 1]
            if not action_ids:
                raise RuntimeError('Response NLL has no response tokens')
            budget = max_length - len(action_ids)
            if len(prompt_ids) > budget:
                prompt_ids = ([prompt_ids[0]] if budget == 1
                              else [prompt_ids[0], *prompt_ids[-(budget - 1):]])
            ids = prompt_ids + action_ids
            rows.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor([-100] * len(prompt_ids) + action_ids, dtype=torch.long))
            lengths.append(len(ids))
        ids = pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        targets = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        positions = torch.arange(ids.size(1), device=device).unsqueeze(0)
        mask = (positions < torch.tensor(lengths, device=device).unsqueeze(1)).long()
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, return_dict=True, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[..., 1:].contiguous().view(-1), ignore_index=-100,
            )
        if not torch.isfinite(loss):
            raise RuntimeError('Response NLL is not finite')
        return loss.item()
    finally:
        tokenizer.padding_side = old_side
