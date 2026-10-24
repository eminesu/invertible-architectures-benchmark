import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from models.autoencoder_family import build_model, parameter_count
from models.invertible_autoencoder import InvertibleAutoencoder
from metrics.evaluate import summarize


@pytest.mark.parametrize('family', ['standard', 'cvae', 'invertible'])
@pytest.mark.parametrize('problem', ['kinematics', 'ballistics'])
def test_models_train_and_sample(family, problem):
    model = build_model(dict(family=family, problem=problem, hidden=16, layers=2)).double()
    x = torch.randn(16, 4, dtype=torch.float64)
    y = torch.randn(16, model.y_dims, dtype=torch.float64)
    model.set_normalization(x, y)
    sum(model.losses(x, y).values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    s = model.sample(y[:3], 7, generator=torch.Generator().manual_seed(0))
    assert s.shape == (3, 7, 4) and s.dtype == torch.float64 and torch.isfinite(s).all()
    assert sum(p.numel() for p in model.parameters()) == parameter_count(family, 4, model.y_dims, model.z_dims, 16, 2)


def test_cvae_kl_matches_torch_distribution():
    model = build_model(dict(family='cvae', problem='kinematics', hidden=16, layers=2)).double()
    x, y = torch.randn(32, 4, dtype=torch.float64), torch.randn(32, 2, dtype=torch.float64)
    mu, lv = model.posterior(x, y)
    expected = kl_divergence(Normal(mu, (lv/2).exp()), Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(1).mean()
    assert torch.allclose(model.losses(x, y)['kl'], expected)


def test_invauto_is_exact_with_orthogonal_square_weights():
    model = InvertibleAutoencoder(4, 2, hidden=4, n_layers=2).double()
    with torch.no_grad():
        for layer in model.encoder:
            q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
            layer.weight.copy_(q)
            layer.bias.normal_()
    x = torch.randn(32, 4, dtype=torch.float64)
    y, z = model.encode(x)
    assert torch.allclose(model.decode(y, z), x, atol=1e-10)
    assert model.orthogonality_terms().max() < 1e-20


@pytest.mark.parametrize('family', ['standard', 'cvae', 'invertible'])
def test_budget_and_ablation(family):
    cfg = dict(family=family, problem='kinematics', layers=4, param_budget=3_000_000)
    m = build_model(cfg)
    count = sum(p.numel() for p in m.parameters())
    assert 2_950_000 < count <= 3_000_000
    assert parameter_count(family, 4, 2, 2, m.encoder[0].out_features+1, 4) > 3_000_000
    if family != 'invertible':
        assert build_model({**cfg, 'hidden': 16, 'z_dims': 4}).z_dims == 4
    else:
        with pytest.raises(ValueError):
            build_model({**cfg, 'z_dims': 4})


def test_all_failed_summary_is_serializable():
    assert summarize(np.array([np.nan, np.inf]))['mean'] is None
