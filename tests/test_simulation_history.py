"""CPU regressions for history feedback, resume isolation and archive loaders."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from opensim import volume
from opensim.simulation import environment, rollout, selection, state


class SimulationHistoryTests(unittest.TestCase):
    def test_defaults_and_explicit_observed_mode(self):
        required = []
        for name in ('micro-dir', 'macro-dir', 'output-dir', 'state-checkpoint',
                     'volume-checkpoint', 'policy-checkpoint', 'summarizer-checkpoint'):
            required += ['--' + name, 'fixture']
        self.assertEqual(rollout.SimulationConfig().state_history_volume, 'simulated')
        self.assertEqual(rollout.parser().parse_args(required).state_history_volume, 'simulated')
        self.assertEqual(rollout.parser().parse_args(
            required + ['--state-history-volume', 'observed']).state_history_volume, 'observed')
        with self.assertRaises(ValueError):
            rollout.SimulationConfig(state_history_volume='invalid').validate()

    def test_day_feedback_uses_actual_rows_and_preserves_volume_diagnostics(self):
        frame = pd.DataFrame({
            'generated_action': ['comment', 'share', 'comment'],
            'individual_mu_dist': [[.2, .5, .3]] * 3,
        })
        original = [{'output': {'num_interactions': 999}},
                    {'output': {'num_interactions': 1234}}]
        for scale in (1, 10):
            for mode in ('simulated', 'observed'):
                with self.subTest(scale=scale, mode=mode):
                    rows = deepcopy(original)
                    environment.update_history(rows, 0, frame, [.2, .5, .3], '模拟摘要', scale,
                                               state_history_volume=mode)
                    expected = 3 * scale if mode == 'simulated' else 999
                    self.assertEqual(rows[0]['output']['num_interactions'], expected)
                    self.assertEqual(rows[1]['output'], original[1]['output'])
                    self.assertEqual(rows[0]['diagnostics']['sample_num_interactors'], 3 * scale)
                    self.assertEqual(rows[0]['diagnostics']['num_comments'], 2 * scale)
                    self.assertEqual(rows[0]['diagnostics']['num_shares'], scale)
                    self.assertEqual(rows[1]['input']['previous_mf_summary'], '模拟摘要')

    def test_two_day_rollout_reads_only_completed_day_history(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            micro, macro = path / 'event_debunk_fixture.json', path / 'event_debunk_fixture.jsonl'
            records = [{'text': '根帖'},
                       {'record_id': 'a', 'content_type': 'comment', 'text': '参考评论',
                        'profile_text': '普通用户', 't': 1},
                       {'record_id': 'b', 'content_type': 'share', 'text': '转发微博',
                        'profile_text': '官方媒体', 't': 2}]
            rows = [dict(t=f't{i + 1}', topic='主题', day_index=i,
                         input=dict(event='主题', disturb_names=[], has_disturb=False,
                                    days_since_disturb=30, previous_mf_summary=''),
                         output={'num_interactions': 999 + i},
                         diagnostics={'sample_num_interactors': 90 + i}) for i in range(2)]
            micro.write_text(json.dumps(records), encoding='utf-8')
            macro.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            frames = {}
            for mode in ('observed', 'simulated'):
                captures = []

                def forecast(history, day_idx, bundle, **kwargs):
                    if day_idx:
                        self.assertEqual(history[0]['diagnostics']['sample_num_interactors'], 30)
                    return {'predicted': 31.0}

                def predict(bundle, history, day_idx, actors, device):
                    captures.append((day_idx, deepcopy(history)))
                    # No current or future observed output is rewritten before generation.
                    self.assertEqual(history[day_idx]['output']['num_interactions'], 999 + day_idx)
                    if day_idx:
                        expected = 30 if mode == 'simulated' else 999
                        self.assertEqual(history[0]['output']['num_interactions'], expected)
                        self.assertAlmostEqual(state.day_numeric_features(history, day_idx)[4],
                                               np.log1p(expected))
                    size = len(actors)
                    return dict(state_dist=np.tile([.2, .5, .3], (size, 1)),
                                state_latent=np.tile([1., 2.], (size, 1)),
                                daily_pred=np.array([.2, .5, .3]),
                                future_daily_pred=np.tile([.2, .5, .3], (3, 1)),
                                future_cumulative_pred=np.tile([.2, .5, .3], (3, 1)),
                                future_delta_pred=np.zeros((3, 3)), future_impact_pred=np.zeros(3))

                def generate(prompts, *args, **kwargs):
                    return [dict(action='comment', content='生成评论', raw_text='动作：评论\n内容：生成评论',
                                 valid=True, trajectory_score=.1, candidate_count=4,
                                 all_candidates=[]) for _ in prompts]

                with ExitStack() as stack:
                    stack.enter_context(patch.object(rollout, 'predict_causal_autoregressive_for_day', forecast))
                    stack.enter_context(patch.object(rollout, 'predict_state', predict))
                    stack.enter_context(patch.object(rollout, 'generate_policy_batch', generate))
                    stack.enter_context(patch.object(rollout, 'response_nll', return_value=.5))
                    stack.enter_context(patch.object(rollout, 'summarize', return_value='模拟摘要'))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    frames[mode] = rollout.simulate_event(
                        micro, macro, state_bundle={}, volume_bundle={
                            'forecaster_type': 'causal_autoregressive', 'target_key': 'sample_num_interactors'},
                        policy_model=SimpleNamespace(_opensim_policy_format='action_content',
                                                     _opensim_latent_clusterer=None),
                        summarizer_model=None, tokenizer=None, device=torch.device('cpu'),
                        config=rollout.SimulationConfig(batch_size=2, warmup=2, scale_factor=10,
                                                       state_history_volume=mode))
                self.assertEqual(len(captures), 4)
                self.assertEqual(len(frames[mode]), 6)
                self.assertEqual(frames[mode].is_warmup.tolist(), [True, True, False, False, False, False])
            excluded = ['state_history_volume', 'day_state_history_volume_writeback']
            first = {mode:frame[frame.time_step == 't1'].drop(columns=excluded)
                     for mode, frame in frames.items()}
            pd.testing.assert_frame_equal(first['observed'], first['simulated'])
            self.assertEqual(json.loads(micro.read_text()), records)
            self.assertEqual([json.loads(line) for line in macro.read_text().splitlines()], rows)

    def test_resume_rejects_a_different_history_mode(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            simulated = {'config': asdict(rollout.SimulationConfig())}
            observed = {'config': asdict(rollout.SimulationConfig(state_history_volume='observed'))}
            rollout._prepare_output(output, simulated, False)
            rollout._prepare_output(output, simulated, True)
            with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
                rollout._prepare_output(output, observed, True)

    def test_archived_volume_alias_keeps_all_contract_checks(self):
        archived = {'forecaster_type': volume.FORECASTER_TYPE,
                    'variant': 'causal_ar_train_only_groupcv_v1',
                    'feature_schema': volume.feature_schema(),
                    'test_data_used_for_fit_or_selection': False}
        with patch('joblib.load', return_value=archived):
            loaded = volume.load_causal_autoregressive_bundle('fixture')
        self.assertEqual(loaded['variant'], volume.VARIANT)
        self.assertEqual(archived['variant'], 'causal_ar_train_only_groupcv_v1')
        for field, value in (('test_data_used_for_fit_or_selection', True),
                             ('feature_schema', []), ('forecaster_type', 'unsupported'),
                             ('variant', 'unknown')):
            with self.subTest(field=field), patch('joblib.load', return_value=dict(archived, **{field:value})):
                with self.assertRaises(ValueError):
                    volume.load_causal_autoregressive_bundle('fixture')

    def test_archived_policy_alias_is_only_in_memory(self):
        class Model:
            config = SimpleNamespace(hidden_size=4)

            def to(self, *args):
                return self

            def eval(self):
                return self

            def load_state_dict(self, value):
                pass

        with TemporaryDirectory() as directory:
            adapter = Path(directory)
            metadata = adapter / 'policy_format.json'
            metadata.write_text(json.dumps({'policy_format': 'action_content_v1'}), encoding='utf-8')
            (adapter / 'trajectory_critic.pt').touch()
            (adapter / 'trainer_state.pt').touch()
            saved = metadata.read_bytes()

            def fake_load(path, **kwargs):
                return ({'fusion_in.weight': torch.zeros((2, 16))}
                        if Path(path).name == 'trajectory_critic.pt'
                        else {'metrics': {'text_sensitive': True}})

            with ExitStack() as stack:
                stack.enter_context(patch('peft.PeftConfig.from_pretrained', return_value=SimpleNamespace(base_model_name_or_path='fixture')))
                stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(pad_token='pad')))
                stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=Model()))
                stack.enter_context(patch('peft.PeftModel.from_pretrained', return_value=Model()))
                stack.enter_context(patch.object(selection, 'FutureTrajectoryCritic', return_value=Model()))
                stack.enter_context(patch('torch.load', side_effect=fake_load))
                _, model = selection.load_policy(adapter, torch.device('cpu'), base_model='fixture')
            self.assertEqual(model._opensim_policy_format, 'action_content')
            self.assertEqual(metadata.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
