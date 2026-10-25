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
@pytest.mark.parametrize('arch', ['mlp', 'paper'])
def test_training_gradients_and_checkpoint_roundtrip(tmp_path, problem, y_dims, z_dims, arch):
    torch.manual_seed(4)
    model = FIF(y_dims=y_dims, z_dims=z_dims, hidden=12, layers=2, kde_points=16,
                discrete_speed=problem == 'ballistics', arch=arch).double()
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
                   discrete_speed=problem == 'ballistics', arch=arch).double()
    restored.load_state_dict(torch.load(path, weights_only=True))
    a = model.sample(y[:3], 17, generator=torch.Generator().manual_seed(2))
    b = restored.sample(y[:3], 17, generator=torch.Generator().manual_seed(2))
    assert a.shape == (3, 17, 4) and torch.isfinite(a).all() and torch.equal(a, b)
    if problem == 'ballistics':
        assert (a[..., 3] >= 0).all() and torch.equal(a[..., 3], a[..., 3].round())


@pytest.mark.parametrize('arch,layers', [('mlp', 4), ('paper', 2)])
def test_parameter_budget_and_invalid_inputs(arch, layers):
    width = width_for_budget(2, 2, 3_000_000, arch=arch)
    model = FIF(hidden=width, arch=arch)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(4, 2, 2, width, layers, arch)
    assert parameter_count(4, 2, 2, width, layers, arch) <= 3_000_000
    assert parameter_count(4, 2, 2, width + 1, layers, arch) > 3_000_000
    with pytest.raises(ValueError):
        FIF(z_dims=4)
    with pytest.raises(RuntimeError, match='fit'):
        model.sample(np.zeros((1, 2)), 3)
    with pytest.raises(ValueError):
        model.sample(np.zeros((1, 3)), 3)
    with pytest.raises(ValueError):
        model.sample(np.zeros((1, 2)), 0)
    with pytest.raises(ValueError):
        FIF(arch='unknown')
    with pytest.raises(ValueError):
        FIF(arch='paper', blocks=-1)


def test_paper_architecture_order_and_residual_identity():
    model = FIF(arch='paper', sigma=0)
    assert len(model.encoder.mlp.hidden) == 2
    assert model.encoder.mlp.hidden[0].out_features == 256
    assert len(model.encoder.residuals) == len(model.decoder.residuals) == 2
    assert all(isinstance(net.activation, nn.ReLU)
               for net in model.modules() if hasattr(net, 'activation'))
    assert not any(isinstance(net, (nn.BatchNorm1d, nn.LayerNorm)) for net in model.modules())
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    assert not encoder_ids.intersection(id(p) for p in model.decoder.parameters())

    events, handles = [], []
    for name, net in [('encode', model.encoder), ('decode', model.decoder)]:
        for part in ('mlp', 'residuals'):
            modules = [net.mlp] if part == 'mlp' else list(net.residuals)
            for module in modules:
                label = f'{name}:{part}'
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output, label=label: events.append(label)))
    x, y, z = torch.randn(5, 4), torch.randn(5, 2), torch.randn(5, 2)
    model.encoder(x, y)
    model.decoder(z, y)
    assert events == ['encode:mlp', 'encode:residuals', 'encode:residuals',
                      'decode:residuals', 'decode:residuals', 'decode:mlp']
    for handle in handles:
        handle.remove()
    for net in (model.encoder, model.decoder):
        for block in net.residuals:
            for parameter in block.parameters():
                nn.init.zeros_(parameter)
    assert torch.equal(model.encoder(x, y), model.encoder.mlp(x, y))
    assert torch.equal(model.decoder(z, y), model.decoder.mlp(z, y))
    assert not torch.allclose(model.encoder(x, y), model.encoder(x, y + 1))


@pytest.mark.parametrize('y_dims,z_dims', [(2, 2), (1, 3)])
def test_paper_trace_gradient_stays_with_encoder(y_dims, z_dims):
    torch.manual_seed(12)
    model = FIF(y_dims=y_dims, z_dims=z_dims, hidden=12, arch='paper', sigma=0)
    terms = model.loss_terms(torch.randn(6, 4), torch.randn(6, y_dims))
    (-terms['trace']).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())


@pytest.mark.parametrize('arch', ['mlp', 'paper'])
def test_evaluator_loads_architecture_from_config(tmp_path, arch):
    from experiments.evaluate_fif import load_run

    settings = dict(hidden=8, layers=2, sigma=0, kde_points=4)
    if arch == 'paper':
        settings.update(arch=arch, blocks=2)
    # A legacy MLP config has no architecture field.
    model = FIF(**settings)
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'config.json').write_text(json.dumps(
        {'method': 'FIF', 'prior_mode': 'fixed', 'model': settings}), encoding='utf-8')
    restored, _, _ = load_run(tmp_path)
    assert restored.arch == arch
    y = torch.randn(2, 2)
    assert torch.equal(model.sample(y, 4, generator=torch.Generator().manual_seed(9)),
                       restored.sample(y, 4, generator=torch.Generator().manual_seed(9)))


@pytest.mark.parametrize('problem,y_dims', [('kinematics', 2), ('ballistics', 1)])
@pytest.mark.parametrize('arch', ['mlp', 'paper'])
def test_cli_architecture_defaults_without_training(tmp_path, monkeypatch, capsys, problem, y_dims, arch):
    import experiments.train_fif as training

    settings = {}
    constructor = training.FIF

    def capture(**kwargs):
        settings.update(kwargs)
        model = constructor(**kwargs)
        settings['n_params'] = sum(p.numel() for p in model.parameters())
        return model

    class StopBeforeData(Exception):
        pass

    def stop(*args):
        raise StopBeforeData

    monkeypatch.setattr(training, 'FIF', capture)
    monkeypatch.setattr(training, 'dataset', stop)
    arguments = ['--problem', problem, '--out', str(tmp_path / 'run'), '--device', 'cpu']
    if arch == 'paper':
        arguments += ['--arch', 'paper']
    with pytest.raises(StopBeforeData):
        training.main(arguments)
    assert settings['arch'] == arch
    assert settings['layers'] == (2 if arch == 'paper' else 4)
    expected_width = {('mlp', 2): 701, ('mlp', 1): 702,
                      ('paper', 2): 697, ('paper', 1): 698}[arch, y_dims]
    assert settings['hidden'] == expected_width
    assert 2_990_000 <= settings['n_params'] <= 3_000_000
    output = capsys.readouterr().out
    assert f'arch={arch}' in output
    prefix = 'kin' if problem == 'kinematics' else 'bal'
    suffix = '_p' if arch == 'paper' else ''
    assert f'{prefix}_fif{suffix}_s0' in output


@pytest.mark.parametrize('problem', ['kinematics', 'ballistics'])
def test_training_rejects_existing_folder_without_touching_checkpoint(tmp_path, monkeypatch, problem):
    import experiments.train_fif as training

    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'previous best checkpoint')

    def fail_if_called(*args):
        pytest.fail('Existing runs must be rejected before preparing training data')

    monkeypatch.setattr(training, 'dataset', fail_if_called)
    with pytest.raises(SystemExit) as error:
        training.main(['--problem', problem, '--arch', 'paper', '--out', str(tmp_path)])
    assert error.value.code == 2
    assert checkpoint.read_bytes() == b'previous best checkpoint'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['model.pt']


@pytest.mark.parametrize('problem', ['kinematics', 'ballistics'])
@pytest.mark.parametrize('previous_evaluations', [0, 2])
def test_evaluation_preserves_existing_outputs(tmp_path, monkeypatch, problem, previous_evaluations):
    from types import SimpleNamespace
    import experiments.evaluate_fif as scoring

    prefix = 'kin' if problem == 'kinematics' else 'bal'
    run = f'{prefix}_fif_p_s0'
    folder = tmp_path / run
    folder.mkdir()
    (folder / 'model.pt').write_bytes(b'previous best checkpoint')
    for index in range(1, previous_evaluations + 1):
        out = folder / ('evaluation' if index == 1 else f'evaluation_{index}')
        out.mkdir()
        (out / 'summary.json').write_text(json.dumps({'previous': index}), encoding='utf-8')
    previous_files = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    model = SimpleNamespace(sample=object(), arch='paper', sigma=0.01,
                            discrete_speed=problem == 'ballistics')
    config = {'problem': problem, 'run': run}
    monkeypatch.setattr(scoring, 'load_run', lambda *args: (model, config, folder))
    outputs = []

    def fake_evaluate(sample, benchmark, name, **kwargs):
        # Verify output routing without sampling or computing benchmark metrics.
        assert sample is model.sample and benchmark == problem and name == run
        outputs.append(kwargs['out_dir'])
        return {'test_output': len(outputs)}

    monkeypatch.setattr(scoring, 'evaluate', fake_evaluate)
    scoring.main([run, '--threads', '1'])
    scoring.main([run, '--threads', '1'])
    expected = [folder / ('evaluation' if i == 1 else f'evaluation_{i}')
                for i in range(previous_evaluations + 1, previous_evaluations + 3)]
    assert outputs == expected
    for path, contents in previous_files.items():
        assert path.read_bytes() == contents
    for number, out in enumerate(outputs, start=1):
        result = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        assert result['test_output'] == number
        assert result['arch'] == 'paper'
        assert (out / f'{problem}_{run}.json').is_file()


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
