"""Command-line orchestration for OpenSim training, simulation and evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN_STAGES = ('state', 'volume', 'policy_data', 'summarizer_data',
                'summarizer', 'sft', 'future', 'gate')
EVAL_STAGES = ('deterministic', 'likelihood', 'semantic', 'metrics', 'metrics_text')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_config(directory: Path, name: str) -> dict:
    import yaml
    with (directory / f'{name}.yaml').open(encoding='utf-8') as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a mapping in {name}.yaml')
    return value


def command(module: str, values: dict, *, underscores: bool = False) -> list[str]:
    result = [sys.executable, '-m', module]
    for key, value in values.items():
        option = '--' + (key if underscores else key.replace('_', '-'))
        if isinstance(value, bool):
            if value:
                result.append(option)
        elif isinstance(value, list):
            for item in value:
                result.extend((option, str(item)))
        elif value is not None:
            result.extend((option, str(value)))
    return result


def build_stages(args: argparse.Namespace, configs: dict) -> list[dict]:
    data, out = args.data_dir, args.output_dir
    artifacts = args.checkpoint_root or out / 'checkpoints'
    state = artifacts / 'state_transition'
    volume = artifacts / 'volume'
    policy_data = out / 'training_data/policy'
    summary_data = out / 'training_data/summarizer'
    summary = artifacts / 'summarizer'
    sft, future = artifacts / 'policy_sft', artifacts / 'policy_future'
    raw, evaluation = out / 'simulation', out / 'evaluation'
    t = configs['train']
    stages = []

    def add(name, module, params, destination, required, underscores=False):
        stages.append({'name': name, 'command': command(module, params, underscores=underscores),
                       'destination': destination, 'required': [str(p) for p in required]})

    add('state', 'opensim.training.train_state_transition', {
        **t['state'], 'macro_dir': data / 'train/macro', 'micro_dir': data / 'train/micro',
        'output_dir': state, 'text_encoder_path': args.base_model,
        'text_cache_db': state / 'embeddings.sqlite', 'device': args.device,
        'text_encoder_device': args.device,
    }, state, [state / 'future_aware_latent_st_best.pt', state / 'train_config.json'])
    add('volume', 'opensim.volume', {
        **t['volume'], 'train_dir': data / 'train/macro', 'output_dir': volume,
    }, volume, [volume / 'causal_autoregressive.joblib'])
    add('policy_data', 'opensim.training.prepare_policy', {
        **t['policy_data'], 'macro_dir': data / 'train/macro', 'micro_dir': data / 'train/micro',
        'st_checkpoint': state / 'future_aware_latent_st_best.pt', 'output_dir': policy_data,
        'device': args.device, 'text_encoder_device': args.device,
    }, policy_data, [policy_data / 'policy_train.csv', policy_data / 'policy_valid.csv',
                     policy_data / 'latent_kmeans.joblib'], True)
    add('summarizer_data', 'opensim.training.prepare_summarizer', {
        **t['summarizer_data'], 'macro_dir': data / 'train/macro',
        'micro_dir': data / 'train/micro', 'output_dir': summary_data,
        'split_config': state / 'train_config.json',
    }, summary_data, [summary_data / 'summarizer_train.jsonl', summary_data / 'summarizer_valid.jsonl'])
    add('summarizer', 'opensim.training.train_summarizer', {
        **t['summarizer'], 'pretrain': args.base_model, 'device': args.device,
        'train_data': summary_data / 'summarizer_train.jsonl',
        'valid_data': summary_data / 'summarizer_valid.jsonl', 'output_dir': summary,
    }, summary, [summary / 'best_checkpoint/adapter_model.safetensors', summary / 'train_summary.json'])
    common = {**t['policy'], 'pretrain': args.base_model,
              'device': args.device,
              'dataset': policy_data / 'policy_train.csv', 'eval_dataset': policy_data / 'policy_valid.csv'}
    add('sft', 'opensim.training.train_policy_sft', {
        **common, **t['sft'], 'output_dir': sft,
    }, sft, [sft / 'best_checkpoint/adapter_model.safetensors', sft / 'train_summary.json'], True)
    add('future', 'opensim.training.train_policy_future', {
        **common, **t['future'], 'init_adapter': sft / 'best_checkpoint', 'output_dir': future,
    }, future, [future / 'train_summary.json'], True)
    add('gate', 'opensim.training.future_gate', {
        **t['gate'], 'checkpoint_root': future, 'pretrain': args.base_model, 'device': args.device,
        'valid_csv': policy_data / 'policy_valid.csv', 'output': out / 'validation/future_gate.json',
    }, out / 'validation', [out / 'validation/future_gate.json',
                           future / 'best_checkpoint/trajectory_critic.pt',
                           future / 'best_checkpoint/adapter_model.safetensors',
                           future / 'best_checkpoint/latent_kmeans.joblib'])
    add('simulation', 'opensim.simulation.rollout', {
        **configs['simulate'], 'micro_dir': args.simulation_micro_dir or data / 'test/micro',
        'macro_dir': data / 'test/macro',
        'output_dir': raw, 'base_model': args.base_model, 'device': args.device,
        'state_checkpoint': state / 'future_aware_latent_st_best.pt',
        'volume_checkpoint': volume / 'causal_autoregressive.joblib',
        'policy_checkpoint': future / 'best_checkpoint',
        'summarizer_checkpoint': summary / 'best_checkpoint',
        'resume': args.resume,
    }, raw, [raw / 'saved_file_paths.json', raw / 'run_manifest.json'])
    add('deterministic', 'opensim.evaluation.deterministic', {
        'saved_paths_json': raw / 'saved_file_paths.json',
        'macro_dir': args.evaluation_macro_dir or data / 'test/macro',
        'output_dir': evaluation / 'deterministic',
    }, evaluation / 'deterministic', [evaluation / 'deterministic/pairwise.csv',
                                      evaluation / 'deterministic/daily.csv'])
    add('likelihood', 'opensim.evaluation.likelihood', {
        'device': args.device,
        'base_model': args.base_model,
        **configs['evaluate']['likelihood'], 'pairwise_csv': evaluation / 'deterministic/pairwise.csv',
        'adapter_path': future / 'best_checkpoint', 'output_csv': evaluation / 'teacher_nll.csv',
        'resume': args.resume,
    }, evaluation / 'teacher_nll.csv', [evaluation / 'teacher_nll.csv'])
    add('semantic', 'opensim.evaluation.semantic', {
        **configs['evaluate']['semantic'], 'pairwise_csv': evaluation / 'deterministic/pairwise.csv',
        'index_csv': data / 'index.csv', 'output_dir': evaluation / 'semantic',
    }, evaluation / 'semantic', [evaluation / 'semantic/semantic.csv', evaluation / 'semantic/progress.csv'])
    add('metrics', 'opensim.evaluation.metrics', {
        **configs['evaluate']['metrics'], 'det_dir': evaluation / 'deterministic',
        'semantic_csv': evaluation / 'semantic/semantic.csv',
        'teacher_nll_csv': evaluation / 'teacher_nll.csv',
        'macro_dir': args.evaluation_macro_dir or data / 'test/macro',
        'output_dir': evaluation / 'metrics',
    }, evaluation / 'metrics', [evaluation / 'metrics/micro.csv', evaluation / 'metrics/macro.csv',
                                evaluation / 'metrics/micro_by_event.csv',
                                evaluation / 'metrics/protocol.json'])
    add('metrics_text', 'opensim.evaluation.metrics', {
        **configs['evaluate']['metrics'], 'cad_source': 'text_classification',
        'det_dir': evaluation / 'deterministic',
        'semantic_csv': evaluation / 'semantic/semantic.csv',
        'teacher_nll_csv': evaluation / 'teacher_nll.csv',
        'macro_dir': args.evaluation_macro_dir or data / 'test/macro',
        'output_dir': evaluation / 'metrics/text_classification',
    }, evaluation / 'metrics/text_classification', [
        evaluation / 'metrics/text_classification/micro.csv',
        evaluation / 'metrics/text_classification/macro.csv',
        evaluation / 'metrics/text_classification/micro_by_event.csv',
        evaluation / 'metrics/text_classification/protocol.json'])
    chosen = {'all': TRAIN_STAGES + ('simulation',) + EVAL_STAGES,
              'train': TRAIN_STAGES, 'simulate': ('simulation',), 'evaluate': EVAL_STAGES}[args.task]
    if args.stage:
        if args.stage not in chosen:
            raise ValueError(f'{args.stage} is not part of {args.task}')
        chosen = (args.stage,)
    return [stage for stage in stages if stage['name'] in chosen]


def require_artifacts(stage: dict) -> None:
    for item in stage['required']:
        path = Path(item)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Incomplete {stage['name']}: {path}")
    if stage['name'] == 'gate':
        if json.loads(Path(stage['required'][0]).read_text())['passed'] is not True:
            raise RuntimeError('Future-policy validation did not pass')


def run(args: argparse.Namespace) -> None:
    if args.checkpoint_root and args.task in ('train', 'all'):
        raise ValueError('--checkpoint-root is for inference/evaluation only; use a new output tree for training')
    if (args.output_dir.is_relative_to(args.data_dir)
            or args.data_dir.is_relative_to(args.output_dir)):
        raise ValueError('Dataset and output directories must not overlap')
    configs = {name: read_config(args.config_dir, name) for name in ('train', 'simulate', 'evaluate')}
    stages = build_stages(args, configs)
    if args.dry_run:
        for stage in stages:
            print(f"[{stage['name']}] {shlex.join(stage['command'])}")
        return
    for split in ('train', 'test'):
        for kind in ('micro', 'macro'):
            if not (args.data_dir / split / kind).is_dir():
                raise FileNotFoundError(args.data_dir / split / kind)
    if not (args.data_dir / 'index.csv').is_file():
        raise FileNotFoundError(args.data_dir / 'index.csv')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Lock only this output tree; concurrent work in another tree is independent.
    import fcntl
    with (args.output_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [args.data_dir / 'index.csv']
        files += sorted(p for split in ('train', 'test')
                        for p in (args.data_dir / split).rglob('*') if p.is_file())
        inputs = {str(p.relative_to(args.data_dir)): digest(p) for p in files}
        for label, directory in (('simulation_micro', args.simulation_micro_dir),
                                 ('evaluation_macro', args.evaluation_macro_dir)):
            if directory:
                if not directory.is_dir():
                    raise FileNotFoundError(directory)
                inputs.update({f'{label}/{p.relative_to(directory)}': digest(p)
                               for p in sorted(directory.rglob('*')) if p.is_file()})
        settings = {'configs': configs, 'base_model': args.base_model, 'device': args.device,
                    'data_dir': str(args.data_dir), 'inputs': inputs,
                    'checkpoint_root': str(args.checkpoint_root) if args.checkpoint_root else None,
                    'validation_gate': str(args.validation_gate) if args.validation_gate else None,
                    'simulation_micro_dir': str(args.simulation_micro_dir) if args.simulation_micro_dir else None,
                    'evaluation_macro_dir': str(args.evaluation_macro_dir) if args.evaluation_macro_dir else None}
        if args.checkpoint_root:
            if not args.checkpoint_root.is_dir():
                raise FileNotFoundError(args.checkpoint_root)
            settings['checkpoint_inputs'] = {
                str(p.relative_to(args.checkpoint_root)): digest(p)
                for p in sorted(args.checkpoint_root.rglob('*')) if p.is_file()}
        if args.validation_gate:
            settings['validation_gate_sha256'] = digest(args.validation_gate)
        state_path = args.output_dir / 'pipeline.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if not args.resume:
                raise RuntimeError('Output already exists; use --resume or a new --output-dir')
            if state['settings'] != settings:
                raise RuntimeError('Configuration or data changed; choose a new --output-dir')
        else:
            state = {'settings': settings, 'completed': [], 'current_stage': None}
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
        for stage in stages:
            name = stage['name']
            if name in state['completed']:
                require_artifacts(stage)
                print(f'[{name}] complete', flush=True)
                continue
            destination = stage['destination']
            occupied = destination.exists() and (destination.is_file() or any(destination.iterdir()))
            if occupied and not (args.resume and name in ('simulation', *EVAL_STAGES)):
                raise RuntimeError(f'Incomplete output exists at {destination}; use a new --output-dir')
            if name == 'simulation':
                gate = args.validation_gate or args.output_dir / 'validation/future_gate.json'
                if not gate.exists() or not json.loads(gate.read_text()).get('passed'):
                    raise RuntimeError('Train and validate the future policy before simulation')
            state['current_stage'] = name
            write_json(state_path, state)
            print(f'[{name}] {shlex.join(stage["command"])}', flush=True)
            subprocess.run(stage['command'], cwd=ROOT, env=env, check=True)
            require_artifacts(stage)
            state['completed'].append(name)
            state['current_stage'] = None
            write_json(state_path, state)


def main(default_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(description='OpenSim training, simulation and evaluation')
    if default_task is None:
        parser.add_argument('task', choices=('all', 'train', 'simulate', 'evaluate'))
    else:
        parser.set_defaults(task=default_task)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    parser.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--checkpoint-root', type=Path,
                        help='Existing validated checkpoints; inference/evaluation only')
    parser.add_argument('--validation-gate', type=Path,
                        help='Existing passed future-policy gate for pretrained simulation')
    parser.add_argument('--simulation-micro-dir', type=Path,
                        help='Optional scaled actor/reference pool; macro features remain unscaled')
    parser.add_argument('--evaluation-macro-dir', type=Path,
                        help='Optional evaluation targets matching the simulation scale')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=TRAIN_STAGES + ('simulation',) + EVAL_STAGES)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for key in ('data_dir', 'output_dir', 'config_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for key in ('checkpoint_root', 'validation_gate', 'simulation_micro_dir', 'evaluation_macro_dir'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if Path(args.base_model).expanduser().exists():
        args.base_model = str(Path(args.base_model).expanduser().resolve())
    run(args)


if __name__ == '__main__':
    main()
