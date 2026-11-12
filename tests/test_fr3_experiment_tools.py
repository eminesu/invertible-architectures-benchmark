"""Planning and scaling analysis use synthetic records, never experiment runs."""

from copy import deepcopy
import json

import pytest

from experiments.analyze_fr3_scaling import collect, fit_group
from experiments.run_fr3_fif_cfmpe import main as plan


def test_plan_never_executes_by_default(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('A default plan must not launch training or evaluation')

    monkeypatch.setattr('experiments.run_fr3_fif_cfmpe.subprocess.run', forbidden)
    commands = plan(['--stage', 'train'])
    assert len(commands) == 2
    for command in commands:
        assert command[command.index('--n-train') + 1] == '1000000'
        assert command[command.index('--epochs') + 1] == '50'
        assert command[command.index('--param-budget') + 1] == '3000000'
    assert 'Plan only' in capsys.readouterr().out
    assert len(plan(['--stage', 'train', '--budgets', '300000', '1000000', '3000000'])) == 6
    assert len(plan(['--stage', 'gt'])) == 2


def test_test_stage_requires_a_single_selected_cfmpe_solver():
    with pytest.raises(SystemExit):
        plan(['--stage', 'test'])
    with pytest.raises(SystemExit):
        plan(['--stage', 'test', '--solver', 'euler:32', 'euler:64'])
    commands = plan(['--stage', 'test', '--methods', 'cfmpe', '--solver', 'euler:64'])
    assert len(commands) == 1 and 'euler:64' in commands[0]
    assert len(plan(['--stage', 'test', '--methods', 'fif'])) == 1


def record(size, seed=0):
    return {'benchmark': 'fr3_6d', 'family': 'FIF', 'arch': 'mlp', 'split': 'val',
            'method': 'conditional FIF', 'n_params': size, 'n_conditions': 200, 'n_samples': 1000,
            'pos_err_mm': {'median': 4 * (size / 1e6) ** -0.5},
            'training': {'seed': seed, 'n_train': 1000000, 'n_val': 20000,
                         'epochs': 50, 'batch_size': 1000, 'param_budget': size}}


def test_scaling_fit_uses_actual_counts_and_known_exponent():
    rows = [(f'{n}_{s}.json', record(n, s)) for n in (100000, 500000, 3000000) for s in (0, 1)]
    result = fit_group(rows, 'pos_err_mm.median')
    assert result['alpha'] == pytest.approx(0.5)
    assert result['A'] == pytest.approx(4)
    assert result['log_space_r2'] == pytest.approx(1)
    assert not result['single_seed']


def test_scaling_rejects_insufficient_sizes_duplicate_runs_and_mixed_seeds():
    rows = [(f'{n}.json', record(n)) for n in (100000, 500000, 3000000)]
    with pytest.raises(ValueError, match='three distinct'):
        fit_group(rows[:2], 'pos_err_mm.median')
    with pytest.raises(ValueError, match='Duplicate'):
        fit_group(rows + [rows[0]], 'pos_err_mm.median')
    with pytest.raises(ValueError, match='same set'):
        fit_group(rows + [('extra.json', record(100000, 1))], 'pos_err_mm.median')
    rows[0][1]['pos_err_mm']['median'] = -0.01
    with pytest.raises(ValueError, match='positive'):
        fit_group(rows, 'pos_err_mm.median')


def test_scaling_separates_protocols_and_ignores_duplicate_summary(tmp_path):
    first = record(100000)
    second = deepcopy(first)
    second['n_samples'] = 100
    for name, value in [('a.json', first), ('b.json', second), ('summary.json', first)]:
        (tmp_path / name).write_text(json.dumps(value))
    groups = collect([tmp_path], 'val')
    assert len(groups) == 2 and sum(map(len, groups.values())) == 2
