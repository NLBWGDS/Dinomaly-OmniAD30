"""Run inside the Linux image: verify PDF entry paths and intentional failures."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    empty_input = output / 'empty-input'
    empty_input.mkdir()
    results = []
    commands = [
        ('pdf-infer', 'infer', 'cd /home/apps/ats-reasoning-tool/ && sh start.sh {input} {output}'),
        ('legacy-infer', 'infer', 'cd /opt/omniad && sh start.sh {input} {output}'),
        ('pdf-train', 'train', 'cd ./root && python train.py --input_dir {input} --output_dir {output}'),
        ('absolute-train', 'train', 'cd /root && python train.py --input_dir {input} --output_dir {output}'),
        ('shell-train', 'train', 'cd /opt/omniad && sh train.sh {input} {output}'),
    ]
    for name, mode, template in commands:
        job_output = output / name
        command = template.format(input=shlex.quote(str(empty_input)), output=shlex.quote(str(job_output)))
        result = subprocess.run(['/bin/bash', '-c', command], cwd='/opt/omniad',
                                capture_output=True, text=True, encoding='utf-8', timeout=90)
        (output / f'{name}.console.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode == 0:
            raise RuntimeError(f'{name}: missing parameters must fail')
        protocol = (job_output / ('state.txt' if mode == 'train' else 'reasoning.log')).read_text(encoding='utf-8')
        debug = (job_output / 'bootstrap.log').read_text(encoding='utf-8')
        marker = 'Start Training' if mode == 'train' else 'reasoning start'
        if protocol.splitlines()[0] != marker or protocol.count(marker) != 1:
            raise RuntimeError(f'{name}: protocol start missing or duplicated')
        if 'error' not in protocol or 'param.json' not in debug or 'Traceback' not in debug:
            raise RuntimeError(f'{name}: failure not diagnosed')
        if 'finish' in protocol or 'reasoning close success' in protocol:
            raise RuntimeError(f'{name}: false success marker')
        results.append(dict(case=name, expected_failure=True, exit_code=result.returncode))

    # A missing interpreter must be visible even before Python can create bootstrap.log.
    bin_dir = output / 'no-python-bin'
    bin_dir.mkdir()
    for command in ('mkdir', 'dirname', 'tee'):
        (bin_dir / command).symlink_to(shutil.which(command))
    env = dict(os.environ, PATH=str(bin_dir))
    job_output = output / 'missing-python'
    result = subprocess.run(['/bin/bash', '/opt/omniad/platform_launch.sh', 'infer',
                             '--input_dir', str(empty_input), '--output_dir', str(job_output)],
                            env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
    if result.returncode != 127:
        raise RuntimeError(f'Missing Python returned {result.returncode}, expected 127')
    protocol = (job_output / 'reasoning.log').read_text(encoding='utf-8')
    console = (job_output / 'entrypoint.log').read_text(encoding='utf-8')
    if 'reasoning error, code=' not in protocol or 'command not found' not in console:
        raise RuntimeError('Missing Python was not logged')
    results.append(dict(case='missing-python', expected_failure=True, exit_code=127))

    broken = output / 'broken-dependency'
    broken.mkdir()
    (broken / 'numpy.py').write_text('raise ImportError("dependency failure probe")\n', encoding='utf-8')
    job_output = output / 'import-error'
    result = subprocess.run(['/bin/bash', '/opt/omniad/platform_launch.sh', 'infer',
                             '--input_dir', str(empty_input), '--output_dir', str(job_output)],
                            env=dict(os.environ, PYTHONPATH=str(broken)), capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    debug = (job_output / 'bootstrap.log').read_text(encoding='utf-8')
    protocol = (job_output / 'reasoning.log').read_text(encoding='utf-8')
    if result.returncode == 0 or 'dependency failure probe' not in debug or 'reasoning error' not in protocol:
        raise RuntimeError('Import failure was not logged before loading the adapter')
    results.append(dict(case='import-error', expected_failure=True, exit_code=result.returncode))
    (output / 'startup_verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Startup checks passed: {len(results)} intentional failures produced diagnostics', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    verify(Path(parser.parse_args().output).resolve())
