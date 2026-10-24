import numpy as np
import pytest
import torch

from data.toy_data import MODELS, PRIOR_PARAMS, make_conditional_dataset, prior_tag, sample_prior_given
from metrics.ground_truth import _cache_path, rejection_sample
from models.conditional_fmpe import ConditionalFMPE
from models.fmpe import count_parameters, n_params


@pytest.mark.parametrize('problem', list(PRIOR_PARAMS))
def test_default_prior_draws_unchanged(problem):
    """The benchmark prior parameters reproduce the benchmark prior draw for draw,
    so their ground truth can share the benchmark cache file."""
    a = MODELS[problem]().sample_prior(5000, np.random.RandomState(0))
    b = sample_prior_given(problem, 5000, np.random.RandomState(0), PRIOR_PARAMS[problem]['default'])
    assert np.array_equal(a, b)


def test_ballistics_dataset():
    x, y, lam = make_conditional_dataset('ballistics', 20_000, seed=0)
    assert x.shape == (20_000, 4) and y.shape == lam.shape == (20_000, 1)
    assert 10 <= lam.min() and lam.max() <= 20
    assert np.isfinite(y).all()
    assert np.allclose(x[:, 3], np.round(x[:, 3]))  # v0 stays integer (no dequantization)
    # v0 | lambda ~ Poisson(lambda): E[v0 - lambda] = 0, Var = E[lambda].
    d = x[:, 3] - lam[:, 0]
    assert abs(d.mean()) < 0.1 and abs(d.var() / lam.mean() - 1) < 0.05


def test_kinematics_dataset():
    x, y, sig = make_conditional_dataset('kinematics', 50_000, seed=0)
    assert x.shape == (50_000, 4) and y.shape == (50_000, 2) and sig.shape == (50_000, 3)
    assert 0.25 <= sig.min() and sig.max() <= 0.75
    # x_0 ~ N(0, 0.25^2) fixed; x_j / sigma_j ~ N(0, 1) for the joints.
    assert abs(x[:, 0].std() - 0.25) < 0.005
    z = x[:, 1:] / sig
    assert np.allclose(z.mean(0), 0, atol=0.02) and np.allclose(z.std(0), 1, atol=0.02)


def test_rejection_sample_follows_prior():
    """The prior-dependent ground truth moves with the prior."""
    y_star = np.array([[8.0]])
    v0 = {lam: rejection_sample('ballistics', y_star, n_gt=300, chunk=50_000, max_proposals=2_000_000,
                                verbose=False, prior_params=[lam])[0][0][:, 3].mean() for lam in (10.0, 20.0)}
    assert v0[10.0] + 1 < v0[20.0]
    # Kinematics: wider joint sigmas spread the joint angles of the posterior.
    y_star = np.array([[1.5, 0.3]])
    spread = {s: rejection_sample('kinematics', y_star, n_gt=500, chunk=500_000, max_proposals=50_000_000,
                                  verbose=False, prior_params=[s] * 3)[0][0][:, 1:].std(0).mean() for s in (0.3, 0.7)}
    assert spread[0.3] * 1.2 < spread[0.7]


def test_cache_key():
    y = np.zeros((3, 1))
    args = ('ballistics', y, 1000, 0.02, 777, 50_000_000)
    assert _cache_path(*args) == _cache_path(*args, None)
    assert _cache_path(*args, [12.0]) != _cache_path(*args) != _cache_path(*args, [18.0])
    assert _cache_path(*args, [12.0]).name.startswith('gt_ballistics_lam12_')
    kin = ('kinematics', np.zeros((3, 2)), 1000, 0.01, 777, 2_000_000_000)
    assert _cache_path(*kin, [0.35, 0.35, 0.35]) != _cache_path(*kin, [0.35, 0.35, 0.65])
    assert prior_tag('kinematics', [0.35, 0.65, 0.5]) == 'sig0.35-0.65-0.5'


@pytest.mark.parametrize('y_dims, prior_dims', [(1, 1), (2, 3)])
def test_parameter_count_and_sample_shapes(y_dims, prior_dims):
    model = ConditionalFMPE(4, y_dims, 16, 2, prior_dims=prior_dims)
    assert count_parameters(model) == n_params(4, y_dims + prior_dims, 16, 2)
    model.set_normalization(torch.randn(100, 4), torch.randn(100, y_dims), torch.rand(100, prior_dims) + 0.5)
    y_star = np.random.RandomState(0).randn(5, y_dims).astype(np.float32)
    one_prior = [0.7] * prior_dims
    assert model.sample(y_star, one_prior, 7, steps=2).shape == (5, 7, 4)
    assert model.sample(y_star, np.full((5, prior_dims), 0.6), 3, steps=2).shape == (5, 3, 4)


# Importing bayesflow (tests/test_npse.py) turns torch autograd off process-wide.
@torch.enable_grad()
def test_learns_prior_dependent_conditional():
    """x | y, c ~ N(y + c_1, (0.3 c_2)^2): a small conditional FMPE trained for a
    few seconds moves its samples with c_1 and scales them with c_2 at fixed y."""
    torch.manual_seed(0)

    def draw(n):
        y, c = torch.randn(n, 1), torch.rand(n, 2) * torch.tensor([4.0, 1.0]) + torch.tensor([-2.0, 0.5])
        return y + c[:, :1] + 0.3 * c[:, 1:] * torch.randn(n, 1), y, c

    model = ConditionalFMPE(1, 1, 64, 2, prior_dims=2)
    model.set_normalization(*draw(10_000))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(2000):
        loss = model.loss(*draw(512))
        opt.zero_grad(); loss.backward(); opt.step()
    y = np.zeros((1, 1), np.float32)
    for shift, scale in [(-1.5, 0.6), (0.0, 1.0), (1.5, 1.4)]:
        x = model.sample(y, [shift, scale], 4000, steps=16, method='midpoint')[0, :, 0]
        assert abs(x.mean().item() - shift) < 0.1
        assert abs(x.std().item() - 0.3 * scale) < 0.08
