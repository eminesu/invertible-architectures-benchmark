"""Queue gates and setting transfer; no training, GPU work or real processes."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import queue_fr3_fif as queue


def complete_run(path, problem, epochs=2):
    path.mkdir()
    cfg = {'problem': problem, 'method': 'FIF', 'run': path.name, 'epochs': epochs,
           'n_params': 2000, 'seed': 0, 'param_budget': 3000000, 'arch': 'mlp',
           'device': 'cuda', 'hidden': None, 'layers': 4, 'blocks': None}
    (path / 'config.json').write_text(json.dumps(cfg))
    (path / 'summary.json').write_text(json.dumps({**cfg}))
    with (path / 'log.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch'])
        writer.writerows([[i] for i in range(1, epochs + 1)])
    (path / 'model.pt').write_bytes(b'fake test artifact')
    (path / 'last.pt').write_bytes(b'fake test artifact')
    return cfg


def test_transfers_settings_with_distinct_5d_name_and_automatic_width():
    cfg = {'arch': 'mlp', 'seed': 0, 'epochs': 50, 'n_train': 1000000,
           'n_val': 20000, 'batch_size': 1000, 'hidden': None, 'layers': 4,
           'blocks': None, 'param_budget': 3000000, 'beta': 100, 'lr': 0.001,
           'device': 'cuda', 'threads': 4, 'kde_points': 1024}
    command = queue.target_command(cfg, 'fr3_5d_fif_s0')
    for flag, value in [('--problem', 'fr3_5d'), ('--run', 'fr3_5d_fif_s0'),
                        ('--seed', '0'), ('--epochs', '50'), ('--param-budget', '3000000')]:
        assert command[command.index(flag) + 1] == value
    assert '--hidden' not in command and '--blocks' not in command


@pytest.mark.parametrize('failure', ['missing_summary', 'incomplete_log', 'missing_last', 'wrong_problem'])
def test_completion_requires_real_completion_artifacts(tmp_path, failure):
    path = tmp_path / 'source'
    complete_run(path, 'fr3_6d')
    if failure == 'missing_summary':
        (path / 'summary.json').unlink()
    elif failure == 'incomplete_log':
        (path / 'log.csv').write_text('epoch\n1\n')
    elif failure == 'missing_last':
        (path / 'last.pt').unlink()
    else:
        cfg = json.loads((path / 'summary.json').read_text())
        cfg['problem'] = 'fr3_5d'
        (path / 'summary.json').write_text(json.dumps(cfg))
    with pytest.raises((OSError, ValueError)):
        queue.verify_complete(path, 'fr3_6d')


def test_no_training_when_source_fails_even_with_completion_marker(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    complete_run(source, 'fr3_6d')
    status = queue.Status(tmp_path, source.name, destination.name, 123)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='5-D was not started'):
        queue.run_queue(source, destination, SimpleNamespace(poll=lambda: 1), status, 0.01)
    assert not destination.exists()


def test_existing_target_is_preserved(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    complete_run(source, 'fr3_6d')
    complete_run(destination, 'fr3_5d')
    before = (destination / 'model.pt').read_bytes()
    status = queue.Status(tmp_path, source.name, destination.name, 123)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(FileExistsError):
        queue.run_queue(source, destination, SimpleNamespace(poll=lambda: 0), status, 0.01)
    assert (destination / 'model.pt').read_bytes() == before


def test_launch_waits_for_source_exit_and_supervises_target_completion(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    complete_run(source, 'fr3_6d')
    status = queue.Status(tmp_path, source.name, destination.name, 123)
    source_results, events = iter([None, 0]), []

    def poll():
        code = next(source_results)
        events.append(('source', code))
        return code

    def launch(command, **kwargs):
        events.append(('launch', command))
        assert events[-2] == ('source', 0)
        complete_run(destination, 'fr3_5d')
        return SimpleNamespace(pid=456, wait=lambda timeout: 0)

    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    queue.run_queue(source, destination, SimpleNamespace(poll=poll), status, 0.01)
    assert status.value['phase'] == 'completed'
    assert status.value['target_epoch'] == 2
    assert len([event for event in events if event[0] == 'launch']) == 1
    assert json.loads((tmp_path / 'status.json').read_text())['phase'] == 'completed'
