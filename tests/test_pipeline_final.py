import argparse
import unittest
from pathlib import Path

from opensim.pipeline import build_stages, run


class PipelineFinalTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            data_dir=Path('/dataset'), output_dir=Path('/new-output'),
            checkpoint_root=Path('/existing-checkpoints'),
            validation_gate=Path('/validation/future_gate.json'),
            simulation_micro_dir=Path('/scaled/micro'),
            evaluation_macro_dir=Path('/scaled/macro'),
            base_model='/models/Qwen3-1.7B', device='cpu', resume=True,
            task='evaluate', stage=None,
        )
        self.configs = {
            'train': {key: {} for key in ('state', 'volume', 'policy_data',
                'summarizer_data', 'summarizer', 'policy', 'sft', 'future', 'gate')},
            'simulate': {'state_history_volume': 'simulated'},
            'evaluate': {'likelihood': {}, 'semantic': {'max_attempts': 5},
                'metrics': {'cad_source': 'native_state'}},
        }

    def test_pretrained_simulation_preserves_feature_macro(self):
        self.args.task = 'simulate'
        stage, = build_stages(self.args, self.configs)
        command = stage['command']
        self.assertEqual(command[command.index('--micro-dir') + 1], '/scaled/micro')
        self.assertEqual(command[command.index('--macro-dir') + 1], '/dataset/test/macro')
        self.assertEqual(command[command.index('--state-history-volume') + 1], 'simulated')
        self.assertIn('/existing-checkpoints/policy_future/best_checkpoint', command)

    def test_evaluation_includes_native_and_text_cad_without_sensitivity(self):
        stages = {stage['name']: stage for stage in build_stages(self.args, self.configs)}
        self.assertEqual(set(stages), {'deterministic', 'likelihood', 'semantic', 'metrics', 'metrics_text'})
        command = stages['likelihood']['command']
        self.assertEqual(command[command.index('--base-model') + 1], self.args.base_model)
        for key, source in (('metrics', 'native_state'), ('metrics_text', 'text_classification')):
            command = stages[key]['command']
            self.assertEqual(command[command.index('--cad-source') + 1], source)
            self.assertNotIn('--sensitivity-support', command)
            self.assertEqual(command[command.index('--macro-dir') + 1], '/scaled/macro')
            self.assertFalse(any('/sensitivity/' in path for path in stages[key]['required']))

    def test_existing_checkpoints_cannot_be_training_destination(self):
        self.args.task = 'train'
        with self.assertRaisesRegex(ValueError, 'inference/evaluation only'):
            run(self.args)


if __name__ == '__main__':
    unittest.main()
