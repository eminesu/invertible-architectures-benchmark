import math

import numpy as np
import pytest
import torch

from models.fmpe import FMPE, count_parameters, n_params, nfe, width_for_budget


def test_parameter_count_formula():
    for x_dims, y_dims, h, blocks in [(4, 2, 37, 3), (4, 1, 512, 6)]:
        assert count_parameters(FMPE(x_dims, y_dims, h, blocks)) == n_params(x_dims, y_dims, h, blocks)


def test_width_for_budget_fits_3M():
    h = width_for_budget(4, 2, 6, 3_000_000)
    assert n_params(4, 2, h, 6) <= 3_000_000 < n_params(4, 2, h + 1, 6)


@pytest.mark.parametrize('alpha', [-0.5, 0.0, 1.0, 4.0])
def test_time_prior_matches_p_alpha(alpha):
    """t = u^(1/(1+alpha)) has density ∝ t^alpha, so E[t] = (alpha+1)/(alpha+2):
    uniform at alpha = 0, more mass near t = 1 for alpha > 0 (FMPE Sec. 3.3, dingo)."""
    model = FMPE(4, 2, 8, 1, alpha=alpha)
    t = model.sample_time(400_000, 'cpu', torch.Generator().manual_seed(0))
    assert abs(t.mean().item() - (alpha + 1) / (alpha + 2)) < 2e-3
    assert 0 <= t.min() and t.max() <= 1


@pytest.mark.parametrize('method, steps, order', [('euler', 64, 1), ('midpoint', 32, 2), ('rk4', 8, 4)])
def test_solvers_converge_at_their_order(method, steps, order):
    """dx/dt = -x + t has the closed form x(t) = t - 1 + (x0 + 1) e^{-t}; halving the step
    size shrinks the error by about 2^order."""
    model = FMPE(2, 1, 8, 1)
    model.velocity = lambda t, x, y: -x + t
    x0 = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    exact = (x0 + 1) * math.exp(-1)
    errs = [(model.integrate(x0, None, s, method) - exact).abs().max().item() for s in (steps, 2 * steps)]
    assert 2 ** order * 0.7 < errs[0] / errs[1] < 2 ** order * 1.4
    assert nfe(method, steps) == {'euler': 1, 'midpoint': 2, 'rk4': 4}[method] * steps


def test_sample_interface_and_shapes():
    model = FMPE(4, 2, 16, 2)
    model.set_normalization(torch.randn(100, 4) * 3 + 1, torch.randn(100, 2))
    y_star = np.random.RandomState(0).randn(5, 2).astype(np.float32)
    x = model.sample(y_star, 7, steps=3, chunk=4)
    assert x.shape == (5, 7, 4)
    # Untrained field is v = 0 (zero output layer), so samples are the base
    # Gaussian mapped back to raw units.
    x = model.sample(y_star, 20_000, steps=2)
    assert torch.allclose(x.reshape(-1, 4).mean(0), model.x_mean, atol=0.1)
    assert torch.allclose(x.reshape(-1, 4).std(0), model.x_std, rtol=0.05)


def test_learns_conditional_gaussian():
    """x | y ~ N((y, -y), diag(0.5^2, 0.2^2)): a small FMPE trained for a few
    seconds recovers the conditional mean and spread."""
    torch.manual_seed(0)
    def draw(n):
        y = torch.randn(n, 1)
        return torch.cat([y + 0.5 * torch.randn(n, 1), -y + 0.2 * torch.randn(n, 1)], 1), y
    model = FMPE(2, 1, 64, 2)
    model.set_normalization(*draw(10_000))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(1500):
        loss = model.loss(*draw(512))
        opt.zero_grad(); loss.backward(); opt.step()
    x = model.sample(torch.tensor([[1.0], [-0.5]]), 4000, steps=16)
    assert torch.allclose(x.mean(1), torch.tensor([[1.0, -1.0], [-0.5, 0.5]]), atol=0.06)
    assert torch.allclose(x.std(1), torch.tensor([[0.5, 0.2], [0.5, 0.2]]).expand(2, 2), rtol=0.15)
