"""FR3 integration contracts; no full training or exact-cache generation."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip('pytorch_kinematics')

from data.toy_data import MODELS, PRIOR_PARAMS
from experiments.fr3_fif_cfmpe_adapters import (
    Fr3ConditionalFMPE, conditional_model_class, evaluation_prior_params,
    make_conditional_dataset, sample_prior_given,
)
from metrics.ground_truth import ground_truth_posterior
from models.conditional_fmpe import ConditionalFMPE
from models.fif import FIF, parameter_count
from robot_ik.kinematics import joint_limits


@pytest.mark.parametrize('problem,y_dims,z_dims', [('fr3_6d', 9, 1), ('fr3_5d', 6, 2)])
@pytest.mark.parametrize('arch', ['mlp', 'paper'])
def test_fif_uses_intrinsic_task_dimension(tmp_path, monkeypatch, problem, y_dims, z_dims, arch):
    import experiments.train_fif as training

    captured = {}
    constructor = training.FIF

    def capture(**kwargs):
        captured.update(kwargs)
        return constructor(**kwargs)

    class BeforeData(Exception):
        pass

    def stop(*args):
        raise BeforeData

    monkeypatch.setattr(training, 'FIF', capture)
    monkeypatch.setattr(training, 'dataset', stop)
    with pytest.raises(BeforeData):
        training.main(['--problem', problem, '--arch', arch, '--out', str(tmp_path / 'run'),
                       '--device', 'cpu', '--param-budget', '20000'])
    assert (captured['x_dims'], captured['y_dims'], captured['z_dims']) == (7, y_dims, z_dims)
    assert captured['sigma'] == 0  # exact posterior: no observation-noise bridge
    model = constructor(**captured)
    layers = 2 if arch == 'paper' else 4
    count = parameter_count(7, y_dims, z_dims, captured['hidden'], layers, arch)
    assert sum(p.numel() for p in model.parameters()) == count <= 20000
    assert parameter_count(7, y_dims, z_dims, captured['hidden'] + 1, layers, arch) > 20000


@pytest.mark.parametrize('problem', ['fr3_6d', 'fr3_5d'])
def test_fr3_prior_family_recovers_benchmark_and_respects_limits(problem):
    nominal = MODELS[problem]().sample_prior(500, np.random.RandomState(4))
    np.testing.assert_array_equal(nominal, sample_prior_given(problem, 500, np.random.RandomState(4), [1] * 7))
    lo, hi = joint_limits()
    widths = np.linspace(0.5, 1, 7)
    x = sample_prior_given(problem, 10000, np.random.RandomState(3), widths)
    standardized = (x - (hi + lo) / 2) / ((hi - lo) / 2 * widths)
    assert np.all(np.abs(standardized) <= 1)
    np.testing.assert_allclose(standardized.mean(0), 0, atol=0.03)
    np.testing.assert_allclose(standardized.std(0), np.sqrt(1 / 3), atol=0.02)
    with pytest.raises(ValueError, match='widths'):
        sample_prior_given(problem, 2, np.random.RandomState(0), [1.1] * 7)


@pytest.mark.parametrize('problem,y_dims', [('fr3_6d', 9), ('fr3_5d', 6)])
def test_conditional_data_anchor_and_reproducibility(problem, y_dims):
    a = make_conditional_dataset(problem, 256, seed=3, nominal_fraction=0.25)
    b = make_conditional_dataset(problem, 256, seed=3, nominal_fraction=0.25)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.float32 and np.isfinite(first).all()
    x, y, c = a
    assert x.shape == c.shape == (256, 7) and y.shape == (256, y_dims)
    assert 20 < np.all(c == 1, axis=1).sum() < 110
    assert np.any(c < 0.9)
    np.testing.assert_allclose(MODELS[problem]().forward_process(x), y, atol=1e-6)


def test_nominal_prior_reuses_exact_reference(monkeypatch):
    import robot_ik.ground_truth as reference

    sentinel, calls = object(), []

    def cached(task, y_star, n_gt, seed, verbose):
        calls.append((task, n_gt, seed))
        return sentinel

    monkeypatch.setattr(reference, 'posterior_samples', cached)
    y = np.zeros((2, 9))
    params = evaluation_prior_params('fr3_6d', [1] * 7)
    assert params is None
    assert ground_truth_posterior('fr3_6d', y, prior_params=params) is sentinel
    assert ground_truth_posterior('fr3_6d', y) is sentinel
    assert calls == [('6d', 1000, 777)] * 2
    with pytest.raises(ValueError, match='nominal'):
        evaluation_prior_params('fr3_6d', [0.8] * 7)
    with pytest.raises(ValueError, match='has no prior parameters'):
        ground_truth_posterior('fr3_6d', y, prior_params=[1] * 7)
    assert len(calls) == 2  # never use the nominal cache to score a narrower prior


def test_adapters_leave_shared_registries_and_classes_unchanged():
    assert 'fr3_6d' not in PRIOR_PARAMS and 'fr3_5d' not in PRIOR_PARAMS
    assert not hasattr(MODELS['fr3_6d'], 'n_constraints')
    assert not hasattr(MODELS['fr3_5d'], 'n_constraints')
    assert conditional_model_class('kinematics') is ConditionalFMPE
    assert conditional_model_class('fr3_6d') is Fr3ConditionalFMPE
    assert Fr3ConditionalFMPE.set_normalization is not ConditionalFMPE.set_normalization


def test_constant_nominal_prior_guard_is_experiment_local():
    model = Fr3ConditionalFMPE(7, 9, 8, 1, prior_dims=7)
    x, y, c = torch.randn(16, 7), torch.randn(16, 9), torch.ones(16, 7)
    model.set_normalization(x, y, c)
    assert torch.isfinite(model.loss(x, y, c))


@pytest.mark.parametrize('problem', ['kinematics', 'ballistics'])
def test_local_data_adapter_preserves_default_toy_draws(problem):
    from data.toy_data import make_conditional_dataset as original

    a = original(problem, 32, seed=4)
    b = make_conditional_dataset(problem, 32, seed=4)
    for expected, actual in zip(a, b):
        np.testing.assert_array_equal(expected, actual)


def test_cfmpe_checkpoint_and_nominal_validation(tmp_path, monkeypatch):
    import experiments.evaluate_conditional_fmpe as scoring

    model = Fr3ConditionalFMPE(7, 9, 8, 1, alpha=1, prior_dims=7)
    model.set_normalization(torch.randn(32, 7), torch.randn(32, 9), torch.rand(32, 7) * 0.5 + 0.5)
    cfg = dict(problem='fr3_6d', hidden_resolved=8, blocks=1, activation='silu',
               sigma_min=1e-4, alpha=1, prior_range=[0.5, 1.0], nominal_fraction=0.25)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    restored, _ = scoring.load_run(tmp_path)
    draws = restored.sample(np.zeros((2, 9), dtype=np.float32), [1] * 7, 5, steps=2, chunk=3)
    assert draws.shape == (2, 5, 7) and torch.isfinite(draws).all()

    seen = []

    def evaluate(sample, problem, **options):
        seen.append(options)
        return {'n_conditions': 2, 'n_samples': 8}

    monkeypatch.setattr(scoring, 'evaluate', evaluate)
    results = scoring.main([str(tmp_path), '--split', 'val', '--conditions', '2',
                            '--samples', '8', '--out', str(tmp_path), '--solver', 'euler:2'])
    assert seen[0]['prior_params'] is None and seen[0]['split'] == 'val'
    assert seen[0]['name'].endswith('_val') and results[0]['family'] == 'CFMPE'
    assert results[0]['prior_params_evaluated'] == [1] * 7
    with pytest.raises(SystemExit):
        scoring.main([str(tmp_path), '--prior', '0.8,0.8,0.8,0.8,0.8,0.8,0.8'])
    assert len(seen) == 1


@pytest.mark.parametrize('y_dims,z_dims', [(9, 1), (6, 2)])
@torch.enable_grad()
def test_fif_fr3_gradients_and_raw_sample_shape(y_dims, z_dims):
    model = FIF(x_dims=7, y_dims=y_dims, z_dims=z_dims, hidden=8, layers=2, sigma=0, kde_points=8)
    x, y = torch.randn(24, 7), torch.randn(24, y_dims)
    model.fit(x, y)
    model.loss_terms(x, y)['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    draws = model.sample(y[:2], 5)
    assert draws.shape == (2, 5, 7) and torch.isfinite(draws).all()


def test_fif_validation_no_longer_reshapes_to_four_variables():
    from experiments.train_fif import validation
    from data.toy_data import make_dataset

    x, y = (torch.from_numpy(a) for a in make_dataset('fr3_6d', 16, 0))
    model = FIF(x_dims=7, y_dims=9, z_dims=1, hidden=8, layers=2, sigma=0, kde_points=8)
    model.fit(x, y)
    result = validation(model, x, y, 8, 'fr3_6d')
    assert all(np.isfinite(v) for v in result.values())
