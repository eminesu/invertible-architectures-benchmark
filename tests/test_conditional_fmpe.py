import numpy as np
import torch

from data.toy_data import LAM_RANGE, InverseBallisticsModel, make_conditional_dataset
from metrics.ground_truth import _cache_path, rejection_sample
from models.conditional_fmpe import ConditionalFMPE
from models.fmpe import count_parameters, n_params


def test_default_lambda_draws_unchanged():
    """lam = 15 reproduces the benchmark prior draw for draw, so the lambda = 15
    ground truth can share the benchmark cache file."""
    m = InverseBallisticsModel()
    a = m.sample_prior(5000, np.random.RandomState(0))
    b = m.sample_prior(5000, np.random.RandomState(0), lam=15.0)
    assert np.array_equal(a, b)


def test_conditional_dataset():
    x, y, lam = make_conditional_dataset(20_000, seed=0)
    assert x.shape == (20_000, 4) and y.shape == lam.shape == (20_000, 1)
    assert LAM_RANGE[0] <= lam.min() and lam.max() <= LAM_RANGE[1]
    assert np.isfinite(y).all()
    assert np.allclose(x[:, 3], np.round(x[:, 3]))  # v0 stays integer (no dequantization)
    # v0 | lambda ~ Poisson(lambda): E[v0 - lambda] = 0, Var = E[lambda].
    d = x[:, 3] - lam[:, 0]
    assert abs(d.mean()) < 0.1 and abs(d.var() / lam.mean() - 1) < 0.05


def test_rejection_sample_follows_lambda():
    """The lambda-dependent ground truth moves v0 with the prior."""
    y_star = np.array([[8.0]])
    v0 = {lam: rejection_sample('ballistics', y_star, n_gt=300, chunk=50_000, max_proposals=2_000_000,
                                verbose=False, lam=lam)[0][0][:, 3].mean() for lam in (10.0, 20.0)}
    assert v0[10.0] + 1 < v0[20.0]


def test_cache_key():
    y = np.zeros((3, 1))
    args = ('ballistics', y, 1000, 0.02, 777, 50_000_000)
    assert _cache_path(*args) == _cache_path(*args, None)
    assert _cache_path(*args, 12.0) != _cache_path(*args) != _cache_path(*args, 18.0)


def test_parameter_count_and_sample_shapes():
    model = ConditionalFMPE(4, 1, 16, 2)
    assert count_parameters(model) == n_params(4, 2, 16, 2)
    model.set_normalization(torch.randn(100, 4), torch.randn(100, 1), torch.rand(100, 1) * 10 + 10)
    y_star = np.random.RandomState(0).randn(5, 1).astype(np.float32)
    assert model.sample(y_star, 12.0, 7, steps=2).shape == (5, 7, 4)
    assert model.sample(y_star, np.full(5, 18.0), 3, steps=2).shape == (5, 3, 4)


# Importing bayesflow (tests/test_npse.py) turns torch autograd off process-wide.
@torch.enable_grad()
def test_learns_prior_dependent_conditional():
    """x | y, lam ~ N(y + lam, 0.3^2): a small conditional FMPE trained for a few
    seconds moves its samples with lambda at fixed y."""
    torch.manual_seed(0)

    def draw(n):
        y, lam = torch.randn(n, 1), torch.rand(n, 1) * 4 - 2
        return y + lam + 0.3 * torch.randn(n, 1), y, lam

    model = ConditionalFMPE(1, 1, 64, 2)
    model.set_normalization(*draw(10_000))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(1500):
        loss = model.loss(*draw(512))
        opt.zero_grad(); loss.backward(); opt.step()
    y = np.zeros((1, 1), np.float32)
    for lam in (-1.5, 0.0, 1.5):
        x = model.sample(y, lam, 4000, steps=16, method='midpoint')[0, :, 0]
        assert abs(x.mean().item() - lam) < 0.1
        assert abs(x.std().item() - 0.3) < 0.08
