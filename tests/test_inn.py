import math

import numpy as np
import pytest
import torch

from metrics.mmd import mmd, mmd_torch
from models.inn import (CINN, INN, Flow, count_parameters, flow_params, width_for_budget)

N, DY, B = 4, 2, 12


def randomize(model, seed=0, scale=0.3):
    """The zero-initialised last layers make a fresh flow the identity; give every
    parameter random values so the tests exercise the real transformation."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def make_flow(cond_dim, dtype=torch.float64):
    flow = Flow(N, cond_dim, n_blocks=3, hidden=16, n_layers=2, perm_seed=1).to(dtype)
    return randomize(flow)


def rand(*shape, seed=0, dtype=torch.float64):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


@pytest.mark.parametrize('cond_dim', [0, 2])
def test_fresh_flow_is_identity(cond_dim):
    flow = Flow(N, cond_dim, n_blocks=3, hidden=16, n_layers=2)
    u = rand(B, N, dtype=torch.float32)
    c = rand(B, cond_dim, seed=1, dtype=torch.float32) if cond_dim else None
    v, log_det = flow(u, c)
    # permutations reorder the coordinates, but values and the log-det are untouched
    assert torch.equal(v.sort(1).values, u.sort(1).values)
    assert torch.equal(log_det, torch.zeros(B))


@pytest.mark.parametrize('cond_dim', [0, 2])
def test_inverse_is_exact(cond_dim):
    flow = make_flow(cond_dim)
    u = rand(B, N)
    c = rand(B, cond_dim, seed=1) if cond_dim else None
    v, _ = flow(u, c)
    assert torch.allclose(flow.inverse(v, c), u, atol=1e-10)
    # and the other way round
    assert torch.allclose(flow(flow.inverse(u, c), c)[0], u, atol=1e-10)


@pytest.mark.parametrize('cond_dim', [0, 2])
def test_log_det_matches_autograd_jacobian(cond_dim):
    flow = make_flow(cond_dim)
    u = rand(5, N)
    c = rand(5, cond_dim, seed=1) if cond_dim else None
    _, log_det = flow(u, c)
    for i in range(len(u)):
        ci = None if c is None else c[i:i + 1]
        J = torch.autograd.functional.jacobian(lambda w: flow(w[None], ci)[0][0], u[i])
        assert torch.allclose(log_det[i], torch.linalg.slogdet(J)[1], atol=1e-9)


def test_condition_changes_the_latent_code():
    flow = make_flow(2)
    u = rand(B, N)
    z_a, _ = flow(u, rand(B, 2, seed=1))
    z_b, _ = flow(u, rand(B, 2, seed=2))
    assert not torch.allclose(z_a, z_b)


def test_soft_clamp_bounds_the_log_scale():
    flow = make_flow(0)
    flow.blocks[0].clamp = 0.5
    with torch.no_grad():
        for p in flow.blocks[0].parameters():
            p.mul_(50)
    _, ld = flow.blocks[0](rand(1000, N) * 10)
    assert ld.abs().max() <= 0.5 * 0.636 * (math.pi / 2) * N + 1e-9  # N scale entries per block


def test_flow_params_formula_and_budget():
    for cond_dim in (0, 2):
        flow = Flow(N, cond_dim, n_blocks=4, hidden=37, n_layers=3)
        assert count_parameters(flow) == flow_params(N, cond_dim, 4, 37, 3)
    h = width_for_budget(N, 2, 8, 3, 3_000_000)
    assert flow_params(N, 2, 8, h, 3) <= 3_000_000 < flow_params(N, 2, 8, h + 1, 3)


def small_inn(loss, **kw):
    return randomize(INN(N, DY, n_blocks=3, hidden=16, n_layers=2, loss=loss, **kw))


def test_inn_ml_loss_matches_eq4():
    m = small_inn('ml', sigma_y=0.1).double()
    x, y = rand(B, N), rand(B, DY, seed=1)
    out, ld = m.flow(x)
    y_hat, z = out[:, :DY], out[:, DY:]
    ref = (0.5 * (((y_hat - y) ** 2).sum(1) / 0.1 ** 2 + (z ** 2).sum(1)) - ld).mean()
    assert torch.allclose(m.losses(x, y)['nll'], ref)


def test_inn_dim_z_matches_paper_table():
    assert INN(4, 2, n_blocks=2, hidden=8, n_layers=2).z_dims == 2   # kinematics
    assert INN(4, 1, n_blocks=2, hidden=8, n_layers=2).z_dims == 3   # ballistics


@pytest.mark.parametrize('latent_mmd', ['joint', 'marginal'])
def test_inn_l2mmd_terms_are_finite_and_differentiable(latent_mmd):
    m = small_inn('l2mmd', latent_mmd=latent_mmd)
    t = m.losses(rand(64, N, dtype=torch.float32), rand(64, DY, seed=1, dtype=torch.float32))
    (t['y'] + t['z']).backward()
    assert all(torch.isfinite(v) for v in t.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_cinn_nll_matches_eq5_and_normalizes():
    m = randomize(CINN(N, DY, n_blocks=3, hidden=16, n_layers=2)).double()
    x, y = rand(B, N), rand(B, DY, seed=1)
    z, ld = m(x, y)
    assert torch.allclose(m.losses(x, y)['nll'], (0.5 * (z ** 2).sum(1) - ld).mean())

    # nll_raw is a proper density on raw x: with x_std != 1 it differs from the
    # standardized loss by exactly the constant and the log x_std change of variables.
    m.x_std.copy_(torch.tensor([0.5, 2.0, 1.0, 3.0]))
    ref = 0.5 * (z ** 2).sum(1) - ld
    raw = m.nll_raw(x, y)
    z2, ld2 = m(x, y)
    assert torch.allclose(raw, 0.5 * (z2 ** 2).sum(1) - ld2 + 0.5 * N * math.log(2 * math.pi)
                          + m.x_std.log().sum())
    assert torch.isfinite(ref).all()


def test_cinn_1d_gaussian_density_integrates_to_one():
    """For x_dims = 2 the density exp(-nll_raw) over a grid should integrate to ~1
    for a fixed condition. Catches a wrong log-det sign or missing constant."""
    m = randomize(CINN(2, 1, n_blocks=3, hidden=16, n_layers=2), scale=0.2).double()
    m.x_std.copy_(torch.tensor([0.7, 1.3]))
    ax = torch.linspace(-12, 12, 600, dtype=torch.float64)
    xx = torch.stack(torch.meshgrid(ax, ax, indexing='ij'), -1).reshape(-1, 2)
    y = torch.full((len(xx), 1), 0.4, dtype=torch.float64)
    dens = (-m.nll_raw(xx, y)).exp()
    cell = (ax[1] - ax[0]) ** 2
    assert abs(dens.sum().item() * cell.item() - 1.0) < 2e-2


@pytest.mark.parametrize('kind', ['inn', 'cinn'])
def test_sample_shape_and_roundtrip(kind):
    m = (small_inn('ml') if kind == 'inn' else randomize(CINN(N, DY, n_blocks=3, hidden=16, n_layers=2))).double()
    m.x_mean.copy_(torch.tensor([0.1, -0.2, 0.3, 5.0])); m.x_std.copy_(torch.tensor([0.5, 1.0, 2.0, 4.0]))
    m.y_mean.copy_(torch.tensor([1.0, -1.0])); m.y_std.copy_(torch.tensor([2.0, 0.5]))
    y = rand(3, DY, seed=5)
    g = torch.Generator().manual_seed(0)
    x = m.sample(y, n_samples=7, generator=g, chunk=10)   # chunk < B*n exercises the loop
    assert x.shape == (3, 7, N)
    flat, y_rep = x.reshape(-1, N), y.repeat_interleave(7, 0)
    if kind == 'inn':
        y_hat_n, _, _ = m(flat)
        assert torch.allclose(y_hat_n * m.y_std + m.y_mean, y_rep, atol=1e-8)   # y part is pinned to y*
    else:
        z, _ = m(flat, y_rep)
        assert abs(z.mean().item()) < 0.5 and 0.5 < z.std().item() < 1.6      # z ~ N(0, I) coarsely (21 draws)


def test_sampling_is_reproducible_with_generator():
    m = small_inn('ml')
    y = rand(2, DY, seed=3, dtype=torch.float32)
    a = m.sample(y, 5, generator=torch.Generator().manual_seed(7))
    b = m.sample(y, 5, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_mmd_torch_matches_numpy_mmd():
    a, b = rand(200, 3).numpy(), (rand(150, 3, seed=1) * 1.5 + 0.2).numpy()
    for unbiased in (True, False):
        assert np.isclose(mmd_torch(torch.from_numpy(a), torch.from_numpy(b), unbiased=unbiased).item(),
                          mmd(a, b, unbiased=unbiased))
