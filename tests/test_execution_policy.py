"""Safety checks for backend selection and remote execution boundaries."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('manager', Path(__file__).parents[1] / 'scripts/colab_manager.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = manager.config()
        self.hw = {'cuda_available': True, 'free_vram_bytes': 3 * 1024**3}

    def decision(self, backend='auto', **kwargs):
        with patch.object(manager, 'local_hardware', return_value=self.hw):
            return manager.choose_backend(self.cfg, backend, **kwargs)

    def test_unknown_work_stays_local(self):
        self.assertEqual(self.decision()['backend'], 'local')

    def test_vram_pressure_recommends_colab_and_preserves_reason(self):
        decision = self.decision(required_vram_gb=8)
        self.assertEqual(decision['backend'], 'colab')
        self.assertTrue(decision['costly'])
        self.assertIn('VRAM', decision['reason'])

    def test_overrides_are_respected(self):
        self.assertEqual(self.decision('local', estimated_minutes=60)['backend'], 'local')
        self.assertEqual(self.decision('colab')['backend'], 'colab')

    def test_missing_cuda_alone_does_not_allocate_remote(self):
        self.hw = {'cuda_available': False}
        self.assertEqual(self.decision()['backend'], 'local')

    def test_costly_job_requires_approval_before_sync(self):
        args = SimpleNamespace(backend='auto', estimated_minutes=60, required_vram_gb=None,
                               task='verify', workers=None, batch_size=None, dry_run=False,
                               approve_expensive=False, budget_seconds=None)
        with patch.object(manager, 'local_hardware', return_value=self.hw), patch.object(manager, 'sync') as sync:
            with self.assertRaisesRegex(RuntimeError, 'approval'):
                manager.run(self.cfg, args, [])
            sync.assert_not_called()

    def test_private_and_generated_directories_are_not_synced(self):
        files = [path.relative_to(manager.ROOT) for path in manager.get_file_list()]
        self.assertIn(Path('scripts/experiment.py'), files)
        for path in files:
            self.assertFalse(set(path.parts) & {'.aws', '.agents', '.codex', '.git', '.venv', 'data', 'checkpoints', 'runs'})

    def test_unknown_or_busy_session_fails_before_exec(self):
        for status in ('Status: BUSY (stdin)', "Session 'lab3' not found.", 'Status: DISCONNECTED'):
            with self.subTest(status=status):
                with patch.object(manager.subprocess, 'run') as call:
                    call.return_value.returncode = 0
                    call.return_value.stdout = status
                    with self.assertRaises(RuntimeError):
                        manager.cli(dict(self.cfg), 'exec', '-s', 'lab3', input_code='print(1)')
                    self.assertEqual(call.call_count, 1)  # status only; no exec or allocation


if __name__ == '__main__':
    unittest.main()
