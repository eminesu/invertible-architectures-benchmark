"""Checkpoint filesystem failure tests; no model, training or GPU operations."""

import csv
import errno
from pathlib import Path
import pickle

import pytest

from experiments import cfmpe_checkpoint_io as checkpoint_io


def lock_error(code=5):
    error = PermissionError(errno.EACCES, 'simulated Windows file lock')
    error.winerror = code
    return error


def payload(epoch, **arguments):
    return {'args': arguments, 'epoch': epoch, 'step': (epoch + 1) * 1000,
            'seconds': float(epoch + 1), 'train_loss': 4 / (epoch + 1),
            'val_loss': 3 / (epoch + 1), 'model': {}, 'opt': {}, 'sched': {},
            'rng_cpu': b'cpu-state', 'rng_device': None}


def write_state(path, epoch, **arguments):
    with path.open('wb') as stream:
        pickle.dump(payload(epoch, **arguments), stream)


def read_state(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


@pytest.mark.parametrize('error_code', [5, 32, 33])
def test_replace_retries_transient_windows_locks(tmp_path, monkeypatch, error_code):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old checkpoint')
    actual_replace = checkpoint_io.os.replace
    attempts, sleeps = [], []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert target.read_bytes() == b'old checkpoint'
            raise lock_error(error_code)
        actual_replace(source, destination)

    monkeypatch.setattr(checkpoint_io.os, 'replace', replace)
    monkeypatch.setattr(checkpoint_io.time, 'sleep', sleeps.append)
    checkpoint_io.atomic_save({'epoch': 2}, target, serializer=pickle.dump)
    assert read_state(target) == {'epoch': 2}
    assert len(attempts) == 3 and sleeps == [0.1, 0.2]
    assert not target.with_name('checkpoint.pt.tmp').exists()


def test_persistent_lock_retains_old_and_complete_temporary_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old checkpoint')
    attempts = []

    def locked(*args):
        attempts.append(1)
        raise lock_error()

    monkeypatch.setattr(checkpoint_io.os, 'replace', locked)
    monkeypatch.setattr(checkpoint_io.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError):
        checkpoint_io.atomic_save({'epoch': 32}, target, serializer=pickle.dump)
    assert len(attempts) == 20
    assert target.read_bytes() == b'old checkpoint'
    assert read_state(target.with_name('checkpoint.pt.tmp')) == {'epoch': 32}


def test_unrelated_permission_error_is_not_retried(monkeypatch):
    attempts = []

    def denied():
        attempts.append(1)
        raise PermissionError(errno.EACCES, 'not a Windows sharing violation')

    monkeypatch.setattr(checkpoint_io.time, 'sleep', lambda _: pytest.fail('must fail immediately'))
    with pytest.raises(PermissionError):
        checkpoint_io.retry_file_operation(denied)
    assert len(attempts) == 1


def test_serialization_failure_does_not_replace_previous_checkpoint(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old checkpoint')

    def interrupted(value, stream):
        stream.write(b'incomplete')
        raise RuntimeError('simulated interrupted serialization')

    with pytest.raises(RuntimeError, match='interrupted serialization'):
        checkpoint_io.atomic_save({}, target, serializer=interrupted)
    assert target.read_bytes() == b'old checkpoint'


def test_newer_complete_temporary_checkpoint_is_selected_without_mutation(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    temporary = tmp_path / 'checkpoint.pt.tmp'
    write_state(target, 31, seed=0)
    write_state(temporary, 32, seed=0)
    before = [path.read_bytes() for path in (target, temporary)]
    with pytest.warns(RuntimeWarning, match='complete temporary'):
        state, source = checkpoint_io.load_latest_checkpoint(target, loader=read_state)
    assert state['epoch'] == 32 and source == temporary
    assert [path.read_bytes() for path in (target, temporary)] == before


def test_partial_temporary_checkpoint_falls_back_to_committed_checkpoint(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    write_state(target, 31, seed=0)
    (tmp_path / 'checkpoint.pt.tmp').write_bytes(b'partial write')
    with pytest.warns(RuntimeWarning, match='Ignoring unreadable'):
        state, source = checkpoint_io.load_latest_checkpoint(target, loader=read_state)
    assert state['epoch'] == 31 and source == target


def test_complete_temporary_checkpoint_can_resume_when_main_is_missing(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    write_state(tmp_path / 'checkpoint.pt.tmp', 0)
    with pytest.warns(RuntimeWarning, match='complete temporary'):
        state, _ = checkpoint_io.load_latest_checkpoint(target, loader=read_state)
    assert state['epoch'] == 0


def test_bad_checkpoints_do_not_start_a_new_run(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'corrupt file')
    with pytest.raises(RuntimeError, match='No readable complete checkpoint'):
        checkpoint_io.load_latest_checkpoint(target, loader=read_state)
    assert target.read_bytes() == b'corrupt file'


def test_mismatching_checkpoint_arguments_are_rejected(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    write_state(target, 31, seed=0)
    write_state(tmp_path / 'checkpoint.pt.tmp', 32, seed=1)
    with pytest.raises(ValueError, match='different training arguments'):
        checkpoint_io.load_latest_checkpoint(target, loader=read_state)


def write_log(path, epochs):
    fields = ['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'seconds']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in epochs:
            state = payload(epoch)
            writer.writerow({key: state[key] if key != 'lr' else 0.001 for key in fields})


def test_resume_log_backs_up_and_removes_uncommitted_or_duplicate_rows(tmp_path):
    path = tmp_path / 'log.csv'
    write_log(path, [0, 1, 1, 2])
    before = path.read_bytes()
    checkpoint_io.align_resume_log(path, payload(1))
    with path.open(newline='', encoding='utf-8') as stream:
        assert [row['epoch'] for row in csv.DictReader(stream)] == ['0', '1']
    backups = list(tmp_path.glob('log.csv.before_resume_*'))
    assert len(backups) == 1 and backups[0].read_bytes() == before


def test_consistent_resume_log_is_left_untouched(tmp_path):
    path = tmp_path / 'log.csv'
    write_log(path, [0, 1, 2])
    before = path.read_bytes()
    checkpoint_io.align_resume_log(path, payload(2))
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('log.csv.before_resume_*'))


def test_disagreeing_log_is_preserved(tmp_path):
    path = tmp_path / 'log.csv'
    write_log(path, [0, 1])
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Log and checkpoint disagree'):
        checkpoint_io.align_resume_log(path, payload(2))
    assert path.read_bytes() == before
