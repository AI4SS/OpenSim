"""Run the complete trained method on full test micro and macro data."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from opensim.policy.prompting import build_policy_chat_prompt
from opensim.summarizer import load_summarizer, summarize
from opensim.volume import disturb_type, load_causal_autoregressive_bundle, predict_causal_autoregressive_for_day
from .environment import day_root, load_event, update_history
from .population import content_type, record_text, sample_reference_actors, warmup_output
from .selection import generate_policy_batch, load_policy, response_nll
from .state import build_policy_profile, latent_state_descriptor, load_state_model, normalize_mean_field, predict_state


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 46
    batch_size: int = 16
    warmup: int = 50
    num_candidates: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    max_content_tokens: int = 48
    scale_factor: int = 1
    state_history_volume: str = 'simulated'

    def validate(self):
        if self.state_history_volume not in {'observed', 'simulated'}:
            raise ValueError('State history volume must be observed or simulated')
        if min(self.batch_size, self.num_candidates, self.max_content_tokens, self.scale_factor) < 1:
            raise ValueError('Batch size, candidates, token budget and scale factor must be positive')
        if self.warmup < 0 or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError('Invalid warmup or sampling parameters')


def _validate_states(outputs, count):
    required = {'state_dist', 'state_latent', 'daily_pred', 'future_daily_pred',
                'future_cumulative_pred', 'future_delta_pred', 'future_impact_pred'}
    if not required.issubset(outputs):
        raise RuntimeError('State inference did not return all required outputs')
    if np.asarray(outputs['state_dist']).shape != (count, 3):
        raise RuntimeError('State inference returned an invalid [positive, neutral, negative] shape')
    for name in required:
        if not np.isfinite(np.asarray(outputs[name])).all():
            raise RuntimeError(f'State inference returned non-finite {name}')
    distributions = np.asarray(outputs['state_dist'])
    if (distributions < 0).any() or not np.allclose(distributions.sum(axis=-1), 1.0, atol=1e-4):
        raise RuntimeError('Individual state outputs are not probability distributions')


def simulate_event(micro_path, macro_path, *, state_bundle, volume_bundle,
                   policy_model, summarizer_model, tokenizer, device,
                   config=SimulationConfig()):
    """Use prior completed-day simulated volume by default; volume recurses."""
    config.validate()
    if volume_bundle.get('forecaster_type') != 'causal_autoregressive' or volume_bundle.get('target_key') != 'sample_num_interactors':
        raise RuntimeError('The train-only sampled-volume causal forecaster is required')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    root, rows, pool, by_day = load_event(micro_path, macro_path)
    history = deepcopy(rows)
    topic = str(rows[0].get('topic') or Path(micro_path).stem)
    summary, all_frames = '', []
    state_counts = torch.zeros(3, dtype=torch.float32, device=device)
    agent_count, step_count = 0, 0
    for day_idx, row in enumerate(rows):
        day_number = day_idx + 1
        day_key = f't{day_number}'
        day_records = by_day.get(day_number, [])
        posts = [r for r in day_records if content_type(r) == 'post']
        root_text = str(day_root(root, posts).get('text') or '')
        forecast = predict_causal_autoregressive_for_day(
            history, day_idx, volume_bundle, event_file=str(micro_path),
        )
        raw = float(forecast['predicted'])
        if not np.isfinite(raw):
            raise RuntimeError('Volume forecast is not finite')
        predicted_raw = max(1, round(raw))
        count = max(1, round(predicted_raw / config.scale_factor))
        actors = sample_reference_actors(pool, count, config.seed + day_number * 1009)
        day_results = []
        before_day = agent_count
        for start in range(0, count, config.batch_size):
            batch = actors[start:start + config.batch_size]
            profiles = [build_policy_profile(r) for r in batch]
            states = predict_state(state_bundle, history, day_idx, batch, device)
            _validate_states(states, len(batch))
            individual = np.asarray(states['state_dist'])
            latents = np.asarray(states['state_latent'])
            descriptors = [latent_state_descriptor(
                individual[i], states['future_delta_pred'], states['future_impact_pred'],
                state_latent=latents[i],
                latent_clusterer=getattr(policy_model, '_opensim_latent_clusterer', None),
            ) for i in range(len(batch))]
            # Counts accumulate predicted individual states, including warmup actors.
            state_counts += torch.tensor(individual.tolist(), dtype=torch.float32, device=device).sum(dim=0)
            agent_count += len(batch)
            cumulative = (state_counts / agent_count).cpu().tolist()
            inp = history[day_idx].get('input') or {}
            disturb_names = '、'.join(str(v) for v in inp.get('disturb_names') or [] if str(v).strip())
            related = (f'day={day_idx}; has_disturb={bool(inp.get("has_disturb"))}; '
                       f'disturb={disturb_names or disturb_type(row) or "无"}; days_since_disturb={inp.get("days_since_disturb")}')
            prompts = [build_policy_chat_prompt(
                topic=topic, root_post=root_text, mean_field=normalize_mean_field(summary), profile=profiles[i],
                latent_descriptor=descriptors[i], hot_comment='暂无最新热门评论',
                related_cases_info=related, policy_format=policy_model._opensim_policy_format,
                include_state_instruction=True,
            ) for i in range(len(batch))]
            warmup_flags = [before_day + start + i < config.warmup for i in range(len(batch))]
            outputs = [warmup_output(p) for p in profiles]
            if not all(warmup_flags):
                outputs = generate_policy_batch(
                    prompts, tokenizer, policy_model, device, current_states=individual.tolist(),
                    state_latents=latents.tolist(), target_trajectory=states['future_cumulative_pred'],
                    num_candidates=config.num_candidates, temperature=config.temperature,
                    top_p=config.top_p, max_content_tokens=config.max_content_tokens,
                )
                for i, warm in enumerate(warmup_flags):
                    if warm:
                        outputs[i] = warmup_output(profiles[i])
            true_actions = ['转发微博' if content_type(r) == 'share' else record_text(r) for r in batch]
            nll = response_nll(policy_model, tokenizer, prompts, true_actions, device)
            step_count += 1
            for i, (actor, output) in enumerate(zip(batch, outputs)):
                source_type = content_type(actor)
                generated_action = output['action']
                record = {
                    'step': step_count, 'agent_index': before_day + start + i + 1,
                    'is_warmup': warmup_flags[i], 'topic': topic,
                    'policy_prompt': prompts[i], 'policy_profile': profiles[i],
                    'policy_root_post': root_text, 'policy_format': policy_model._opensim_policy_format,
                    'actual_model_generation': not warmup_flags[i],
                    'state': ('Positive', 'Neutral', 'Negative')[int(np.argmax(individual[i]))],
                    'profile_state': profiles[i], 'generated_comment': output['content'],
                    'generated_action': generated_action, 'generated_raw_output': output['raw_text'],
                    'generated_format_valid': output['valid'],
                    'generated_content_valid': not bool(output.get('invalid_reason', '')),
                    'generated_invalid_reason': output.get('invalid_reason', ''),
                    'trajectory_candidate_score': output['trajectory_score'],
                    'sft_candidate_log_likelihood': None,
                    'trajectory_candidate_count': output['candidate_count'],
                    'trajectory_candidates': json.dumps(output['all_candidates'], ensure_ascii=False),
                    'real_comment': str(actor.get('text') or record_text(actor)),
                    'source_text': actor.get('source_text') or record_text(actor),
                    'source_content_type': source_type,
                    'action_match': source_type == generated_action,
                    'actor_sampling_mode': 'reference',
                    'log_prob': nll, 'mf_loss': 0.0, 'mean_field': summary,
                    'mu_dist': np.asarray(states['daily_pred']).tolist(),
                    'individual_mu_dist': individual[i].tolist(),
                    'state_latent': latents[i].tolist(), 'latent_state_descriptor': descriptors[i],
                    'future_daily_pred': np.asarray(states['future_daily_pred']).tolist(),
                    'future_cumulative_pred': np.asarray(states['future_cumulative_pred']).tolist(),
                    'future_delta_pred': np.asarray(states['future_delta_pred']).tolist(),
                    'future_impact_pred': np.asarray(states['future_impact_pred']).tolist(),
                    'event': Path(micro_path).stem,
                    'day_key': day_key, 'day_index': day_number,
                    'time_step': day_key, 'time_step_index': day_number,
                    'post_mode': 'environment', 'source_sampling_seed': config.seed,
                    'actor_sampling_seed': config.seed,
                    'day_full_pred': predicted_raw, 'day_run_pred': count,
                    'day_predicted': count, 'day_predicted_model': count,
                    'day_prediction_source': 'model', 'day_predicted_raw': predicted_raw,
                    'day_scale': 1.0, 'day_scale_factor': config.scale_factor,
                    'day_simulated_interactions': count,
                    'state_history_volume': config.state_history_volume,
                    'day_state_history_volume_writeback': (
                        count * config.scale_factor if config.state_history_volume == 'simulated'
                        else (row.get('output') or {}).get('num_interactions', 0)
                    ),
                    'day_available_hint': sum(content_type(r) != 'post' for r in day_records),
                    'day_post_count': len(posts), 'file': str(micro_path),
                }
                for key in ('source_record_id', 'source_mid', 'source_t', 'source_original_index',
                            'source_reuse_round', 'source_is_reused', 'source_sampling_mode',
                            'source_random_draw_index', 'source_random_pool_size',
                            'source_random_action_pool_size', 'source_random_share_ratio',
                            'source_random_seed', 'source_is_original', 'source_original_match_method'):
                    record[key] = actor.get(key, '')
                day_results.append(record)
            summary = summarize(topic, profiles, [o['content'] for o in outputs],
                                summary, cumulative, summarizer_model, tokenizer, device)
        frame = pd.DataFrame(day_results)
        summary = normalize_mean_field(summary)
        update_history(history, day_idx, frame, cumulative, summary, config.scale_factor,
                       state_history_volume=config.state_history_volume)
        print(json.dumps({
            'type': 'day_feedback', 'arm': config.state_history_volume,
            'event': Path(micro_path).stem, 'day': day_key,
            'observed_previous_volume': ((rows[day_idx - 1].get('output') or {}).get('num_interactions', 0)
                                         if day_idx else 0),
            'state_previous_volume_used': ((history[day_idx - 1].get('output') or {}).get('num_interactions', 0)
                                           if day_idx else 0),
            'generated_interactions': len(frame),
            'full_sample_count_proxy': len(frame) * config.scale_factor,
            'state_volume_written': (history[day_idx].get('output') or {}).get('num_interactions', 0),
            'volume_diagnostic_written': (history[day_idx].get('diagnostics') or {}).get('sample_num_interactors', 0),
        }, ensure_ascii=False), flush=True)
        all_frames.append(frame)
        print(f'{Path(micro_path).stem}: {day_key}, predicted={count}, generated={len(frame)}', flush=True)
    return pd.concat(all_frames, ignore_index=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _input_digest(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if '__pycache__' in file.parts or file.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')):
                continue
            result[str(file)] = _sha256(file)
    return result


def _json_write(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)


def _prepare_output(output, fingerprint, resume):
    output = Path(output).resolve()
    manifest_path = output / 'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError('Output directory must be empty unless resuming its own recorded run')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('fingerprint') != fingerprint:
            raise RuntimeError('Resume input/model/configuration fingerprint does not match this run')
        for event, item in manifest.get('completed_events', {}).items():
            file = output / f'{event}.csv'
            if not file.is_file() or _sha256(file) != item['sha256']:
                raise RuntimeError(f'Resume output is missing or changed: {event}')
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'fingerprint': fingerprint, 'completed_events': {}, 'status': 'running'}
    _json_write(manifest_path, manifest)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('micro-dir', 'macro-dir', 'output-dir', 'state-checkpoint',
                   'volume-checkpoint', 'policy-checkpoint', 'summarizer-checkpoint'):
        p.add_argument('--' + option, required=True)
    p.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--events', default='', help='Comma-separated event IDs; default: all micro JSON files')
    p.add_argument('--seed', type=int, default=46)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--num-candidates', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-content-tokens', type=int, default=48)
    p.add_argument('--scale-factor', type=int, default=1)
    p.add_argument('--state-history-volume', choices=('observed', 'simulated'), default='simulated')
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = SimulationConfig(**{name: getattr(args, name) for name in SimulationConfig.__dataclass_fields__})
    config.validate()
    micro, macro, output = [Path(getattr(args, name)).resolve() for name in ('micro_dir', 'macro_dir', 'output_dir')]
    events = args.events.split(',') if args.events else sorted(p.stem for p in micro.glob('*.json'))
    if not events or len(set(events)) != len(events):
        raise ValueError('At least one unique event ID is required')
    for event in events:
        if event != Path(event).name or event in {'.', '..'}:
            raise ValueError('Events must be simple file stems')
    event_paths = [(micro / f'{event}.json', macro / f'{event}.jsonl') for event in events]
    checkpoint_paths = [Path(getattr(args, name)).resolve() for name in
                        ('state_checkpoint', 'volume_checkpoint', 'policy_checkpoint', 'summarizer_checkpoint')]
    protected = [micro, macro, *checkpoint_paths]
    if Path(args.base_model).is_dir():
        protected.append(Path(args.base_model).resolve())
    for path in protected:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('Output must be separate from data, checkpoints and the base model')
    if Path(args.base_model).is_dir():
        base_path = Path(args.base_model).resolve()
    else:
        from huggingface_hub import snapshot_download
        base_path = Path(snapshot_download(repo_id=args.base_model, local_files_only=True)).resolve()
    if output == base_path or output in base_path.parents or base_path in output.parents:
        raise ValueError('Output must be separate from the base model cache')
    package_sources = sorted(Path(__file__).resolve().parents[1].rglob('*.py'))
    fingerprint = {'config': asdict(config), 'events': events, 'base_model': args.base_model,
                   'device': args.device,
                   'inputs': _input_digest([p for pair in event_paths for p in pair] + checkpoint_paths
                                           + [base_path] + package_sources)}
    manifest = _prepare_output(output, fingerprint, args.resume)
    if manifest['status'] == 'completed':
        print(f'Complete matching run already exists: {output}', flush=True)
        return 0
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    state_bundle = load_state_model(checkpoint_paths[0], device, Path.cwd(),
                                   base_model=str(base_path), cache_path=output / 'state_text_cache.sqlite')
    volume_bundle = load_causal_autoregressive_bundle(checkpoint_paths[1])
    tokenizer, policy = load_policy(checkpoint_paths[2], device, base_model=str(base_path))
    summarizer = load_summarizer(checkpoint_paths[3], str(base_path), device)
    try:
        for event, (micro_path, macro_path) in zip(events, event_paths):
            if event in manifest['completed_events']:
                continue
            frame = simulate_event(micro_path, macro_path, state_bundle=state_bundle,
                                   volume_bundle=volume_bundle, policy_model=policy,
                                   summarizer_model=summarizer, tokenizer=tokenizer,
                                   device=device, config=config)
            temporary = output / f'{event}.csv.tmp'
            destination = output / f'{event}.csv'
            frame.to_csv(temporary, index=False)
            temporary.replace(destination)
            manifest['completed_events'][event] = {'sha256': _sha256(destination), 'rows': len(frame)}
            _json_write(output / 'run_manifest.json', manifest)
            entries = [dict(event=e, file_class=e, test_files=str(micro / f'{e}.json'),
                            output_file=str(output / f'{e}.csv'), micro_dir=str(micro),
                            macro_dir=str(macro), scale_factor=config.scale_factor)
                       for e in events if e in manifest['completed_events']]
            _json_write(output / 'saved_file_paths.json', entries)
        manifest['status'] = 'completed'
        _json_write(output / 'run_manifest.json', manifest)
    except Exception:
        manifest['status'] = 'failed'
        _json_write(output / 'run_manifest.json', manifest)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
