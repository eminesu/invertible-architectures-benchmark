"""CFMPE-local checkpoint I/O, including transient Windows sharing failures."""

import csv
import io
import os
from pathlib import Path
import pickle
import shutil
import time
import uuid
import warnings


def retry_file_operation(operation, attempts=20, delay=0.1):
    """Bounded retry for Windows access/sharing/lock violations only."""
    if attempts < 1 or delay < 0:
        raise ValueError('Need positive attempts and a nonnegative delay')
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == attempts - 1:
                raise
            time.sleep(min(delay * 2 ** attempt, 1.0))


def atomic_save(state, path, serializer=None):
    """Close and flush the temporary file before retrying atomic replacement.

    A failed replacement leaves both the old destination and the complete
    temporary file available. No delete-and-copy fallback is used.
    """
    if serializer is None:
        import torch
        serializer = torch.save
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')

    def write():
        with temporary.open('wb') as stream:
            serializer(state, stream)
            stream.flush()
            os.fsync(stream.fileno())

    retry_file_operation(write)
    try:
        retry_file_operation(lambda: os.replace(temporary, path))
    except PermissionError as error:
        if hasattr(error, 'add_note'):
            error.add_note(f'Complete temporary save retained at {temporary}. '
                           'Release the file lock (e.g. pause folder sync) and rerun the same command.')
        raise


def _checkpoint_arguments(state):
    # Compatibility with the original toy CFMPE checkpoints.
    return {'nominal_fraction': 0.0, 'threads': 4, **state['args']}


def load_latest_checkpoint(path, loader=None):
    """Read-only selection of the newest complete main or temporary checkpoint."""
    if loader is None:
        import torch
        loader = lambda source: torch.load(source, map_location='cpu', weights_only=False)
    path = Path(path)
    candidates, failures = [], []
    required = {'args', 'epoch', 'step', 'seconds', 'train_loss', 'val_loss',
                'model', 'opt', 'sched', 'rng_cpu', 'rng_device'}
    for source in (path, path.with_name(path.name + '.tmp')):
        if not source.is_file():
            continue
        try:
            state = retry_file_operation(lambda: loader(source))
            if (not isinstance(state, dict) or not required.issubset(state)
                    or not isinstance(state['args'], dict) or int(state['epoch']) < 0
                    or int(state['step']) < 1):
                raise ValueError('Incomplete checkpoint structure')
            candidates.append((state, source))
        except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
            failures.append(f'{source.name}: {error}')
    if not candidates:
        if failures:
            raise RuntimeError('No readable complete checkpoint; existing files were preserved. '
                               + '; '.join(failures))
        return None, None
    if any(_checkpoint_arguments(state) != _checkpoint_arguments(candidates[0][0])
           for state, _ in candidates[1:]):
        raise ValueError('Main and temporary checkpoints have different training arguments; files were preserved')
    state, source = max(candidates, key=lambda item: (int(item[0]['epoch']), int(item[0]['step'])))
    if failures:
        warnings.warn('Ignoring unreadable checkpoint candidate: ' + '; '.join(failures), RuntimeWarning)
    if source != path:
        warnings.warn(f'Resuming from complete temporary checkpoint {source} at epoch {state["epoch"]}',
                      RuntimeWarning)
    return state, source


def align_resume_log(path, state):
    """Keep committed epochs once; back up the old log before changing it.

    The trainer logs before saving. A failed/incomplete save can therefore
    leave an extra row, which must not be duplicated when its epoch is rerun.
    """
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if fields != ['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'seconds']:
        raise ValueError(f'Unexpected CFMPE log header at {path}; log was preserved')
    committed = {}
    for row in rows:
        epoch = int(row['epoch'])
        if epoch <= int(state['epoch']):
            committed[epoch] = row
    latest = committed.get(int(state['epoch']))
    if (latest is None or int(latest['step']) != int(state['step'])
            or float(latest['train_loss']) != float(state['train_loss'])
            or float(latest['val_loss']) != float(state['val_loss'])):
        raise ValueError(f'Log and checkpoint disagree at epoch {state["epoch"]}; log was preserved')
    kept = [committed[epoch] for epoch in sorted(committed)]
    if kept == rows:
        return
    backup = path.with_name(f'{path.name}.before_resume_{uuid.uuid4().hex[:8]}')
    retry_file_operation(lambda: shutil.copyfile(path, backup))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(kept)
    atomic_save(buffer.getvalue().encode('utf-8'), path, serializer=lambda value, stream: stream.write(value))
    print(f'Aligned log to checkpoint epoch {state["epoch"]}; original log saved at {backup}', flush=True)
