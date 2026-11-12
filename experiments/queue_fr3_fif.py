"""Wait for a successful FR3 6-D FIF run, then train the 5-D counterpart.

This controller uses no GPU itself. It checks both the original process exit
code and complete training artifacts before launching the next job. Status and
console output live in the explicitly selected queue directory.
"""

import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cfmpe_checkpoint_io import atomic_save, retry_file_operation


def read_json(path):
    return retry_file_operation(lambda: Path(path).read_text(encoding='utf-8'))


def last_epoch(folder):
    try:
        with (folder / 'log.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        return int(rows[-1]['epoch']) if rows else 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def verify_complete(folder, problem):
    cfg = json.loads(read_json(folder / 'config.json'))
    summary = json.loads(read_json(folder / 'summary.json'))
    if (cfg.get('method') != 'FIF' or cfg.get('problem') != problem
            or summary.get('problem') != problem or summary.get('run') != cfg.get('run')
            or summary.get('n_params') != cfg.get('n_params')):
        raise ValueError(f'Training configuration and summary do not agree at {folder}')
    with (folder / 'log.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row['epoch']) for row in rows]
    if epochs != list(range(1, int(cfg['epochs']) + 1)):
        raise ValueError(f'Training did not complete all {cfg["epochs"]} epochs at {folder}')
    for filename in ('model.pt', 'last.pt'):
        if (folder / filename).stat().st_size == 0:
            raise ValueError(f'Empty training artifact: {folder / filename}')
    return cfg


def target_command(cfg, target_name):
    command = [sys.executable, '-u', str(ROOT / 'experiments' / 'train_fif.py'),
               '--problem', 'fr3_5d', '--run', target_name]
    for flag, key in (
        ('--arch', 'arch'), ('--seed', 'seed'), ('--device', 'device'),
        ('--threads', 'threads'), ('--n-train', 'n_train'), ('--n-val', 'n_val'),
        ('--epochs', 'epochs'), ('--batch-size', 'batch_size'),
        ('--param-budget', 'param_budget'), ('--beta', 'beta'), ('--lr', 'lr'),
        ('--layers', 'layers'), ('--blocks', 'blocks'), ('--hidden', 'hidden'),
        ('--sigma', 'sigma'), ('--kde-points', 'kde_points'),
    ):
        if cfg.get(key) is not None:
            command.extend([flag, str(cfg[key])])
    return command


class WindowsProcess:
    """Hold a handle to the original process, preventing PID-reuse confusion."""

    def __init__(self, pid):
        if os.name != 'nt':
            raise RuntimeError('This controller monitors an existing Windows process')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x101000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:  # WAIT_TIMEOUT: original process is still active
            return None
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Status:
    def __init__(self, directory, source_name, target_name, source_pid):
        self.directory = directory
        self.value = {'manager_pid': os.getpid(), 'source_pid': source_pid,
                      'source_run': source_name, 'target_run': target_name}

    def update(self, phase, **details):
        new = {**self.value, 'phase': phase, **details}
        if new == self.value:
            return
        new['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            atomic_save(new, self.directory / 'status.json',
                        serializer=lambda value, stream: stream.write(json.dumps(value, indent=2).encode('utf-8')))
        except PermissionError as error:
            # A sync lock on status metadata must not interrupt the supervised job.
            print(f'Status file is temporarily unavailable: {error}', file=sys.stderr, flush=True)
        self.value = new
        print(json.dumps(new), flush=True)


def run_queue(source, destination, source_process, status, poll_seconds):
    if destination.exists():
        raise FileExistsError(f'Target run already exists; refusing to overwrite: {destination}')
    cfg = json.loads(read_json(source / 'config.json'))
    if cfg.get('method') != 'FIF' or cfg.get('problem') != 'fr3_6d':
        raise ValueError('The source must be a FR3 6-D FIF run')
    command = target_command(cfg, destination.name)
    while True:
        status.update('waiting_for_6d', source_epoch=last_epoch(source), source_epochs=cfg['epochs'],
                      target_command=command)
        exit_code = source_process.poll()
        if exit_code is not None:
            break
        time.sleep(poll_seconds)
    if exit_code != 0:
        raise RuntimeError(f'6-D process exited with code {exit_code}; 5-D was not started')
    if verify_complete(source, 'fr3_6d') != cfg:
        raise ValueError('Source configuration changed while waiting; 5-D was not started')
    if destination.exists():
        raise FileExistsError(f'Target run appeared while waiting; refusing to overwrite: {destination}')
    training_log = status.directory / 'train_5d.log'
    status.update('starting_5d', source_epoch=cfg['epochs'], source_exit_code=exit_code,
                  target_log=str(training_log))
    with training_log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        while True:
            status.update('training_5d', target_pid=process.pid, target_epoch=last_epoch(destination),
                          target_epochs=cfg['epochs'])
            try:
                exit_code = process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
    if exit_code != 0:
        raise RuntimeError(f'5-D training exited with code {exit_code}; inspect {training_log}')
    verify_complete(destination, 'fr3_5d')
    status.update('completed', target_epoch=cfg['epochs'], target_exit_code=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--target-run', required=True)
    p.add_argument('--source-pid', type=int, required=True)
    p.add_argument('--queue-dir', type=Path, required=True)
    p.add_argument('--poll-seconds', type=float, default=15)
    args = p.parse_args(argv)
    for name in (args.source_run, args.target_run):
        if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name:
            p.error('Run names must be filename components')
    if not 0 < args.poll_seconds <= 60:
        p.error('--poll-seconds must be in (0, 60]')
    directory = args.queue_dir.resolve()
    if not directory.is_relative_to(ROOT):
        p.error('--queue-dir must be inside this project')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'controller.lock').open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    status = Status(directory, args.source_run, args.target_run, args.source_pid)
    source_process = None
    try:
        source_process = WindowsProcess(args.source_pid)
        run_queue(ROOT / 'experiments' / 'runs' / args.source_run,
                  ROOT / 'experiments' / 'runs' / args.target_run, source_process, status, args.poll_seconds)
    except Exception as error:
        status.update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if source_process is not None:
            source_process.close()


if __name__ == '__main__':
    main()
