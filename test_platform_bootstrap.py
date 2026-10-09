import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parent


class BootstrapTests(unittest.TestCase):
    def fixture_run(self, root, mode, source, arguments=(), output=None):
        app = root / 'app'
        app.mkdir()
        shutil.copyfile(SOURCE / 'platform_bootstrap.py', app / 'platform_bootstrap.py')
        (app / f'platform_{mode}.py').write_text(source, encoding='utf-8')
        out = root / 'output' if output is None else output
        result = subprocess.run(
            [sys.executable, '-u', str(app / 'platform_bootstrap.py'), mode,
             '--output_dir', str(out), *arguments],
            capture_output=True, text=True, encoding='utf-8', timeout=30)
        return result, out

    def test_import_failures_are_logged_before_adapter_initialization(self):
        for mode in ('train', 'infer'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                result, out = self.fixture_run(Path(tmp), mode, 'import nonexistent_dependency_probe\n')
                self.assertEqual(result.returncode, 1)
                debug = (out / 'bootstrap.log').read_text(encoding='utf-8')
                self.assertIn('ModuleNotFoundError', result.stderr)
                self.assertIn('ModuleNotFoundError', debug)
                self.assertIn('Traceback', debug)
                log = (out / ('state.txt' if mode == 'train' else 'reasoning.log')).read_text(encoding='utf-8')
                self.assertTrue(log.startswith('Start Training\n' if mode == 'train' else 'reasoning start\n'))
                self.assertIn('error', log)
                self.assertNotIn('finish', log)
                self.assertNotIn('reasoning close success', log)

    def test_syntax_error_and_stderr_are_persisted(self):
        for source, expected in [('def main(:\n', 'SyntaxError'),
                                 ('import sys\nprint("loader diagnostic", file=sys.stderr)\n'
                                  'raise RuntimeError("probe")\n', 'loader diagnostic')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                result, out = self.fixture_run(Path(tmp), 'infer', source)
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, (out / 'bootstrap.log').read_text(encoding='utf-8'))
                self.assertIn(expected, result.stderr)

    def test_success_appends_protocol_without_duplicate_start_or_fake_finish(self):
        for mode, marker, terminal in [('infer', 'reasoning start', 'reasoning close success'),
                                       ('train', 'Start Training', 'finish omniad training')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                protocol = root / 'output' / ('state.txt' if mode == 'train' else 'reasoning.log')
                source = ('from platform_bootstrap import ProtocolLog\n'
                          'def main():\n'
                          f'    log = ProtocolLog({str(protocol)!r})\n'
                          f'    log.write({marker!r})\n'
                          '    log.write("progress")\n'
                          f'    log.write({terminal!r})\n'
                          '    log.close()\n')
                result, out = self.fixture_run(root, mode, source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(protocol.read_text(encoding='utf-8').splitlines(),
                                 [marker, 'progress', terminal])
                self.assertEqual(result.stdout.count(marker), 1)
                self.assertIn('exit code=0', (out / 'bootstrap.log').read_text(encoding='utf-8'))

    def test_nonzero_exit_is_preserved_and_never_adds_training_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.fixture_run(Path(tmp), 'train', 'def main():\n    raise SystemExit(7)\n')
            self.assertEqual(result.returncode, 7)
            self.assertNotIn('finish', (out / 'state.txt').read_text(encoding='utf-8'))

    def test_exception_text_cannot_inject_training_terminal_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.fixture_run(Path(tmp), 'train',
                                          'raise RuntimeError("finish accidentally in error")\n')
            self.assertEqual(result.returncode, 1)
            self.assertNotIn('finish', (out / 'state.txt').read_text(encoding='utf-8'))
            self.assertIn('finish accidentally', (out / 'bootstrap.log').read_text(encoding='utf-8'))

    def test_zero_exit_does_not_manufacture_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.fixture_run(Path(tmp), 'infer', 'def main():\n    raise SystemExit(0)\n')
            self.assertEqual(result.returncode, 0)
            self.assertEqual((out / 'reasoning.log').read_text(encoding='utf-8'), 'reasoning start\n')

    def test_unwritable_output_reports_to_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = root / 'blocked'
            blocked.write_text('keep', encoding='utf-8')
            result, _ = self.fixture_run(root, 'infer', 'def main():\n    pass\n', output=blocked)
            self.assertEqual(result.returncode, 73)
            self.assertIn('reasoning error, code=0x80100000', result.stderr)
            self.assertIn('Traceback', result.stderr)
            self.assertEqual(blocked.read_text(encoding='utf-8'), 'keep')

    def test_argument_errors_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.fixture_run(Path(tmp), 'infer',
                'import argparse\ndef main():\n    argparse.ArgumentParser().parse_args()\n',
                arguments=('--invalid',))
            self.assertEqual(result.returncode, 2)
            self.assertIn('unrecognized arguments', (out / 'bootstrap.log').read_text(encoding='utf-8'))
            self.assertIn('reasoning error', (out / 'reasoning.log').read_text(encoding='utf-8'))

    def test_real_inference_missing_parameters_logs_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'output'
            result = subprocess.run([sys.executable, str(SOURCE / 'platform_bootstrap.py'), 'infer',
                '--input_dir', str(Path(tmp) / 'missing'), '--output_dir', str(out)],
                capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertNotEqual(result.returncode, 0)
            log = (out / 'reasoning.log').read_text(encoding='utf-8')
            self.assertEqual(log.count('reasoning start'), 1)
            self.assertIn('param.json', (out / 'bootstrap.log').read_text(encoding='utf-8'))
            self.assertNotIn('reasoning close success', log)

    def test_help_does_not_require_numpy_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'output'
            result = subprocess.run([sys.executable, '-S', str(SOURCE / 'root/train.py'),
                '--help', '--output_dir', str(out)], capture_output=True, text=True,
                encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertNotIn('finish', (out / 'state.txt').read_text(encoding='utf-8'))

    def shell_run(self, root, missing_python=False):
        bash = shutil.which('bash')
        if not bash and os.name == 'nt':
            candidate = Path('C:/Program Files/Git/bin/bash.exe')
            if candidate.is_file():
                bash = str(candidate)
        if not bash:
            self.skipTest('Bash is not installed')
        app = root / 'app'
        app.mkdir()
        for name in ('platform_bootstrap.py', 'platform_launch.sh'):
            shutil.copyfile(SOURCE / name, app / name)
        (app / 'platform_infer.py').write_text('def main():\n    raise SystemExit(9)\n', encoding='utf-8')
        out = root / 'output'
        env = dict(os.environ)
        if missing_python:
            # Git Bash ships mkdir/tee/dirname but no Python. Linux gets a minimal PATH.
            if os.name == 'nt':
                env['PATH'] = str(Path(bash).parent.parent / 'usr/bin')
            else:
                bin_dir = root / 'bin'
                bin_dir.mkdir()
                for tool in ('mkdir', 'tee', 'dirname'):
                    (bin_dir / tool).symlink_to(shutil.which(tool))
                env['PATH'] = str(bin_dir)
        result = subprocess.run([bash, (app / 'platform_launch.sh').as_posix(), 'infer',
                                 '--output_dir', out.as_posix()], env=env,
                                capture_output=True, text=True, encoding='utf-8', timeout=30)
        return result, out

    def test_shell_tee_preserves_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.shell_run(Path(tmp))
            self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
            self.assertIn('SystemExit: 9', (out / 'entrypoint.log').read_text(encoding='utf-8'))
            self.assertIn('reasoning error', (out / 'reasoning.log').read_text(encoding='utf-8'))

    def test_shell_logs_missing_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out = self.shell_run(Path(tmp), missing_python=True)
            self.assertEqual(result.returncode, 127, result.stdout + result.stderr)
            self.assertIn('command not found', (out / 'entrypoint.log').read_text(encoding='utf-8'))
            self.assertIn('reasoning error', (out / 'reasoning.log').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
