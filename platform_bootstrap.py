"""Standard-library-only startup logging, before loading the model adapters."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
from pathlib import Path
import signal
import sys
import traceback


_active_protocol = None
_active_start = None


class ProtocolLog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrapped = self.path == _active_protocol
        self.handle = self.path.open('a' if self.bootstrapped else 'w',
                                     encoding='utf-8', buffering=1)

    def write(self, message):
        if self.bootstrapped and message == _active_start:
            return
        print(message, flush=True)
        self.handle.write(message + '\n')

    def close(self):
        self.handle.close()


class TeeStream:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.console.flush()
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def error_record(mode, exc):
    if mode == 'train':
        # Never put exception text containing the terminal keyword in state.txt.
        return 'error Startup or execution failed; see bootstrap.log and train_debug.log'
    detail = f'{type(exc).__name__}: {exc}'.replace('\r', ' ').replace('\n', ' ')
    return f'reasoning error, code=0x80100000, message={detail}'


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def launch(mode, arguments=None):
    global _active_protocol, _active_start
    if mode not in {'train', 'infer'}:
        raise ValueError('mode must be train or infer')
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    print(f'omniad bootstrap start mode={mode} pid={os.getpid()}', flush=True)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument('--output_dir', default='/output')
    parse_error = None
    try:
        known, _ = parser.parse_known_args(arguments)
        output = Path(known.output_dir).resolve()
    except (argparse.ArgumentError, ValueError) as exc:
        output, parse_error = Path('/output'), exc
    protocol = output / ('state.txt' if mode == 'train' else 'reasoning.log')
    start = 'Start Training' if mode == 'train' else 'reasoning start'
    try:
        output.mkdir(parents=True, exist_ok=True)
        debug = (output / 'bootstrap.log').open('w', encoding='utf-8', buffering=1)
    except OSError as exc:
        print(error_record(mode, exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 73

    old_argv = sys.argv
    old_protocol, old_start = _active_protocol, _active_start
    old_handler = None
    code = 0
    with debug, redirect_stdout(TeeStream(sys.stdout, debug)), redirect_stderr(TeeStream(sys.stderr, debug)):
        try:
            print(f'omniad bootstrap version=1.0.3 mode={mode} cwd={Path.cwd()} python={sys.executable}', flush=True)
            print(f'output={output}', flush=True)
            protocol.write_text(start + '\n', encoding='utf-8')
            print(start, flush=True)
            _active_protocol, _active_start = protocol.resolve(), start
            if parse_error is not None:
                raise parse_error
            old_handler = signal.signal(signal.SIGTERM, _terminate)
            sys.argv = [f'platform_{mode}.py', *arguments]
            print(f'loading platform_{mode}', flush=True)
            adapter = importlib.import_module(f'platform_{mode}')
            adapter.main()
        except BaseException as exc:
            code = (exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)) \
                if isinstance(exc, SystemExit) else (130 if isinstance(exc, KeyboardInterrupt) else 1)
            if code:
                traceback.print_exc()
                record = error_record(mode, exc)
                print(record, file=sys.stderr, flush=True)
                try:
                    with protocol.open('a', encoding='utf-8') as handle:
                        handle.write(record + '\n')
                except OSError:
                    traceback.print_exc()
        finally:
            sys.argv = old_argv
            _active_protocol, _active_start = old_protocol, old_start
            if old_handler is not None:
                signal.signal(signal.SIGTERM, old_handler)
            print(f'omniad bootstrap exit code={code}', flush=True)
    return code


if __name__ == '__main__':
    # Adapters import this same module to append to the already-started protocol log.
    sys.modules['platform_bootstrap'] = sys.modules[__name__]
    if len(sys.argv) < 2 or sys.argv[1] not in {'train', 'infer'}:
        raise SystemExit('usage: platform_bootstrap.py {train|infer} [arguments...]')
    raise SystemExit(launch(sys.argv[1], sys.argv[2:]))
