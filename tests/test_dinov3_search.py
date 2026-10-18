"""Search orchestration checks using synthetic pipeline outputs; no GPU or weights needed."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from variants.dinov3 import search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'upstream'
        self.repo.mkdir()
        (self.repo / 'hubconf.py').write_text('# official source placeholder for orchestration test')
        self.weights = self.root / 'weights.pth'
        self.argv = ['search.py', '--experiment', 'dinov3_test', '--datasets', 'RAVDESS',
                     '--dinov3-repo', str(self.repo), '--dinov3-weights', str(self.weights),
                     '--learning-rates', '0.00001', '0.0001', '--weight-decays', '0.0001']

    def run_search(self, runner, evaluator, resume=False):
        with patch.object(search, 'ROOT', self.root), patch.object(search, 'require_baseline_complete'), \
             patch.object(search.sys, 'argv', self.argv + (['--resume'] if resume else [])), \
             patch.object(search.subprocess, 'run', side_effect=runner) as run, \
             patch.object(search, 'evaluate_winner', side_effect=evaluator) as evaluate:
            code = search.main()
        return code, run, evaluate

    def test_missing_weights_persists_state_and_never_launches(self):
        code, run, evaluate = self.run_search(None, None)
        state = json.loads((self.root / 'artifacts/dinov3_test/search_status.json').read_text())
        self.assertEqual(code, 2)
        self.assertEqual(state['state'], 'awaiting_weights')
        run.assert_not_called()
        evaluate.assert_not_called()

    def test_validation_selects_winner_and_resume_does_not_repeat(self):
        self.weights.write_bytes(b'synthetic fixture; never loaded as a model')
        def runner(command, **kwargs):
            self.assertIn('--validation-only', command)
            self.assertEqual(command[command.index('--label-percent')+1], '10')
            experiment = command[command.index('--experiment')+1]
            lr = float(command[command.index('--learning-rate')+1])
            folder = self.root / 'experiments_logs' / experiment / 'RAVDESS/ft_SupCon_10p_seed_0'
            folder.mkdir(parents=True)
            # Lower loss must win even when its validation accuracy is lower.
            loss, accuracy = (0.5, 0.6) if lr == 1e-5 else (0.8, 0.9)
            (folder / 'selection.json').write_text(json.dumps(dict(epoch=2, validation_loss=loss)))
            (folder / 'history.json').write_text(json.dumps([dict(epoch=2, validation_accuracy=accuracy)]))
            return SimpleNamespace(returncode=0)
        code, run, evaluate = self.run_search(runner, lambda *args: {'accuracy': 0.7})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(evaluate.call_count, 1)
        best = json.loads((self.root / 'artifacts/dinov3_test/best_parameters_RAVDESS.json').read_text())
        self.assertEqual(best['learning_rate'], 1e-5)
        self.assertEqual(best['validation_accuracy'], 0.6)
        self.assertTrue((self.root / 'artifacts/dinov3_test/trials.csv').is_file())
        code, run, evaluate = self.run_search(runner, None, resume=True)
        self.assertEqual(code, 0)
        run.assert_not_called()
        evaluate.assert_not_called()

    def test_failed_trial_is_recorded_and_other_trial_can_finish(self):
        self.weights.write_bytes(b'fixture')
        def runner(command, **kwargs):
            experiment = command[command.index('--experiment')+1]
            if experiment.endswith('t00'):
                return SimpleNamespace(returncode=1)
            folder = self.root / 'experiments_logs' / experiment / 'RAVDESS/ft_SupCon_10p_seed_0'
            folder.mkdir(parents=True)
            (folder / 'selection.json').write_text(json.dumps(dict(epoch=1, validation_loss=1.0)))
            (folder / 'history.json').write_text(json.dumps([dict(epoch=1, validation_accuracy=0.5)]))
            return SimpleNamespace(returncode=0)
        code, _, _ = self.run_search(runner, lambda *args: {'accuracy': 0.5})
        self.assertEqual(code, 0)
        state = json.loads((self.root / 'artifacts/dinov3_test/search_status.json').read_text())
        self.assertEqual(state['trials'][0]['state'], 'failed')
        self.assertEqual(state['trials'][0]['exit_code'], 1)
        self.assertEqual(state['trials'][1]['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
