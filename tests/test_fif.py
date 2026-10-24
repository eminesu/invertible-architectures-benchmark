"""Checks of the conditional FIF mathematics and its benchmark integration."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from models.fif import FIF, parameter_count, width_for_budget


@pytest.fixture(scope='module', autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Affine(nn.Module):
    def __init__(self, matrix, context):
        super().__init__()
        self.matrix = nn.Parameter(torch.as_tensor(matrix, dtype=torch.float64))
        self.context = nn.Parameter(torch.as_tensor(context, dtype=torch.float64))

    def forward(self, x, y):
        return x @ self.matrix.T + y @ self.context.T


def test_exact_trace_and_gradient_routing():
    model = FIF(x_dims=3, y_dims=1, z_dims=2, hidden=4, layers=1, sigma=0).double()
    model.encoder = Affine([[1, 2, 3], [4, 5, 7]], [[2], [3]])
    model.decoder = Affine([[0.2, 0.4], [0.3, 0.7], [0.5, 0.8]], [[1], [2], [3]])
    x, y = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 1, dtype=torch.float64)
    terms = model.loss_terms(x, y)
    expected = torch.trace(model.encoder.matrix @ model.decoder.matrix)
    assert torch.allclose(terms['trace'], expected)
    (-terms['trace']).backward()
    assert torch.allclose(model.encoder.matrix.grad, -model.decoder.matrix.T)
    assert model.decoder.matrix.grad is None  # stop-gradient belongs on the decoder
    assert model.encoder.context.grad is None  # differentiate x, holding y fixed


def test_sampling_uses_condition_and_returns_raw_units():
    model = FIF(x_dims=2, y_dims=1, z_dims=1, hidden=4, layers=1, sigma=0).double()
    model.decoder = Affine([[1], [0]], [[0], [1]])
    model.x_mean.copy_(torch.tensor([10, 20]))
    model.x_std.copy_(torch.tensor([2, 3]))
    model.y_mean.fill_(7)
    model.y_std.fill_(2)
    y = np.array([[9.0], [5.0]])
    a = model.sample(y, 100, generator=torch.Generator().manual_seed(6), chunk_size=37)
    b = model.sample(y, 100, generator=torch.Generator().manual_seed(6), chunk_size=37)
    assert a.shape == (2, 100, 2) and torch.equal(a, b)
    assert torch.equal(a[:, :, 1], torch.tensor([23., 17.])[:, None].expand(2, 100))
    assert 1.5 < a[0, :, 0].std() < 2.5


def test_noise_bridge_matches_gaussian_bayes_update():
    model = FIF(x_dims=2, y_dims=1, z_dims=1, sigma=0.5, kde_points=1).double()
    model.centres.fill_(2)
    model.kde_cov.fill_(4)
    model.fitted.fill_(True)
    draws = model.clean_context(torch.zeros(1, 1, dtype=torch.float64), 20000,
                                torch.Generator().manual_seed(7))
    assert float(draws.mean()) == pytest.approx(2 * 0.25 / 4.25, abs=0.015)
    assert float(draws.var()) == pytest.approx(4 * 0.25 / 4.25, abs=0.01)


def test_noise_bridge_weights_output_prior_instead_of_only_jittering_y():
    model = FIF(x_dims=2, y_dims=1, z_dims=1, sigma=1, kde_points=2).double()
    model.centres.copy_(torch.tensor([[-3.], [3.]]))
    model.kde_cov.fill_(1)
    model.fitted.fill_(True)
    draws = model.clean_context(torch.tensor([[2.]], dtype=torch.float64), 20000,
                                torch.Generator().manual_seed(8))
    weight_right = 1 / (1 + np.exp(-6))
    expected_mean = weight_right * 2.5 + (1 - weight_right) * -0.5
    assert float(draws.mean()) == pytest.approx(expected_mean, abs=0.025)


def test_noise_bridge_two_dimensional_scaling_and_covariance():
    model = FIF(sigma=0.4, kde_points=1).double()
    model.y_mean.copy_(torch.tensor([5., -2.]))
    model.y_std.copy_(torch.tensor([2., 4.]))
    model.centres.copy_(torch.tensor([[1., -1.]]))
    model.kde_cov.copy_(torch.tensor([[2., 0.6], [0.6, 1.]]))
    model.fitted.fill_(True)
    observed = torch.tensor([[5., -2.]], dtype=torch.float64)
    draws = model.clean_context(observed, 30000, torch.Generator().manual_seed(9))
    noise = torch.diag((0.4 / model.y_std).square())
    covariance = torch.linalg.inv(torch.linalg.inv(model.kde_cov) + torch.linalg.inv(noise))
    mean = covariance @ torch.linalg.solve(model.kde_cov, model.centres[0])
    assert torch.allclose(draws.mean(0), mean, atol=0.004)
    assert torch.allclose(torch.cov(draws.T), covariance, atol=0.001)


@pytest.mark.parametrize('problem,y_dims,z_dims', [('kinematics', 2, 2), ('ballistics', 1, 3)])
def test_training_gradients_and_checkpoint_roundtrip(tmp_path, problem, y_dims, z_dims):
    torch.manual_seed(4)
    model = FIF(y_dims=y_dims, z_dims=z_dims, hidden=12, layers=2, kde_points=16,
                discrete_speed=problem == 'ballistics').double()
    x = torch.randn(32, 4, dtype=torch.float64)
    x[:, 3] = torch.arange(32) % 20
    y = torch.randn(32, y_dims, dtype=torch.float64)
    model.fit(x, y)
    terms = model.loss_terms(x, y)
    terms['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = FIF(y_dims=y_dims, z_dims=z_dims, hidden=12, layers=2, kde_points=16,
                   discrete_speed=problem == 'ballistics').double()
    restored.load_state_dict(torch.load(path, weights_only=True))
    a = model.sample(y[:3], 17, generator=torch.Generator().manual_seed(2))
    b = restored.sample(y[:3], 17, generator=torch.Generator().manual_seed(2))
    assert a.shape == (3, 17, 4) and torch.isfinite(a).all() and torch.equal(a, b)
    if problem == 'ballistics':
        assert (a[..., 3] >= 0).all() and torch.equal(a[..., 3], a[..., 3].round())


def test_parameter_budget_and_invalid_inputs():
    width = width_for_budget(2, 2, 3_000_000)
    model = FIF(hidden=width)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(4, 2, 2, width, 4)
    assert parameter_count(4, 2, 2, width, 4) <= 3_000_000
    assert parameter_count(4, 2, 2, width + 1, 4) > 3_000_000
    with pytest.raises(ValueError):
        FIF(z_dims=4)
    with pytest.raises(RuntimeError, match='fit'):
        model.sample(np.zeros((1, 2)), 3)
    with pytest.raises(ValueError):
        model.sample(np.zeros((1, 3)), 3)
    with pytest.raises(ValueError):
        model.sample(np.zeros((1, 2)), 0)


@pytest.mark.parametrize('problem', ['kinematics', 'ballistics'])
def test_train_and_shared_evaluation(tmp_path, monkeypatch, problem):
    from experiments.train_fif import dataset, main as train
    from experiments.evaluate_fif import main as score
    import metrics.evaluate as shared

    folder = tmp_path / ('kin_fif_s0' if problem == 'kinematics' else 'bal_fif_s0')
    train(['--problem', problem, '--out', str(folder), '--device', 'cpu', '--threads', '1',
           '--n-train', '64', '--n-val', '16', '--epochs', '1', '--batch-size', '16',
           '--hidden', '8', '--layers', '2', '--kde-points', '16'])
    x, y, _ = dataset(problem, 16, 30000)
    # Small synthetic references exercise the real evaluator without a costly
    # rejection run. This test makes no posterior-accuracy claim.
    monkeypatch.setattr(shared, 'load_test_conditions', lambda _: (y[:2].numpy(), x[:2].numpy()))
    monkeypatch.setattr(shared, 'ground_truth_posterior', lambda *a, **kw: [x[:8].numpy()] * 2)
    result = score([str(folder), '--conditions', '2', '--samples', '8', '--gt-samples', '8',
                    '--threads', '1'])
    assert result['n_conditions'] == 2 and result['n_samples'] == 8
    assert result['prior_mode'] == 'fixed' and np.isfinite(result['err_post']['mean'])
    assert (folder / 'evaluation' / 'summary.json').is_file()
    assert json.loads((folder / 'config.json').read_text())['model']['z_dims'] == (2 if problem == 'kinematics' else 3)
    with pytest.raises(SystemExit):
        train(['--problem', problem, '--out', str(folder)])  # never overwrite a run
