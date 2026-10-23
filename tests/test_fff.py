import math

import numpy as np
import pytest
import torch

from models.fff import ANGLE_DIMS, ConditionalFFF, count_parameters, n_params


def perturbed(model, std=0.05):
    """Move the zero-initialized output layers off the identity."""
    for net in (model.encoder, model.decoder):
        torch.nn.init.normal_(net.out.weight, std=std)
    return model


@pytest.mark.parametrize('angle_dims', [(), ANGLE_DIMS['kinematics']])
def test_parameter_count_and_sample_shapes(angle_dims):
    model = ConditionalFFF(4, 2, 16, 2, angle_dims=angle_dims)
    assert model.ambient_dims == 4 + len(angle_dims)
    assert count_parameters(model) == n_params(model.ambient_dims, 2, 16, 2)
    model.set_normalization(torch.randn(100, 4), torch.randn(100, 2))
    x = perturbed(model).sample(np.random.RandomState(0).randn(5, 2).astype(np.float32), 7)
    assert x.shape == (5, 7, 4)
    if angle_dims:
        a = x[..., list(angle_dims)]
        assert (a > -math.pi).all() and (a <= math.pi).all()


def test_embedding_round_trip_and_projection():
    model = ConditionalFFF(4, 2, 8, 1, angle_dims=(1, 2, 3))
    model.set_normalization(torch.randn(100, 4), torch.randn(100, 2))
    x = torch.rand(50, 4) * 6 - 3
    assert torch.allclose(model.unembed(model.embed(x)), x, atol=1e-5)
    u = model.project(torch.randn(50, 7))
    assert torch.allclose(u[:, 1:].view(50, 3, 2).norm(dim=2), torch.ones(50, 3), atol=1e-6)
    v = model.to_tangent(u, torch.randn(50, 7))
    assert torch.allclose((v[:, 1:].view(50, 3, 2) * u[:, 1:].view(50, 3, 2)).sum(2), torch.zeros(50, 3), atol=1e-6)


@torch.enable_grad()  # importing bayesflow (tests/test_npse.py) turns autograd off process-wide
@pytest.mark.parametrize('angle_dims', [(), ANGLE_DIMS['kinematics']])
def test_identity_at_init(angle_dims):
    """Zero output layers + skip connection: f = g = id, so reconstruction is exact."""
    model = ConditionalFFF(4, 2, 16, 2, angle_dims=angle_dims)
    x, y = torch.randn(64, 4) * 0.5, torch.randn(64, 2)
    model.set_normalization(x, y)
    terms = model.loss_terms(x, y)
    assert terms['recon'].item() < 1e-10
    terms['loss'].backward()  # forward-mode JVP + VJP graph is differentiable


@torch.enable_grad()
@pytest.mark.parametrize('angle_dims', [(), (0,)])
def test_density_normalizes(angle_dims):
    """exp(log_prob) integrates to 1 over R and over the circle: checks the
    tangent-space log-det and the latent densities."""
    torch.manual_seed(0)
    model = perturbed(ConditionalFFF(1, 1, 32, 2, angle_dims=angle_dims))
    if angle_dims:
        x = torch.linspace(-math.pi, math.pi, 20001)[:-1, None]
    else:
        x = torch.linspace(-8, 8, 20001)[:, None]
    lp = model.log_prob(x, torch.full((len(x), 1), 0.3)).detach()
    assert abs((lp.exp().sum() * (x[1] - x[0])).item() - 1) < 2e-3


@torch.enable_grad()
def test_torus_learns_distribution_across_the_cut():
    """phi | y ~ N(pi + y, 0.3^2) wrapped: the mass straddles +-pi, where a
    Euclidean model sees two far-apart clusters. The torus FFF recovers the
    circular mean, moving with y."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    y = rng.uniform(-0.5, 0.5, (20_000, 1)).astype(np.float32)
    phi = np.angle(np.exp(1j * (np.pi + y + 0.3 * rng.randn(*y.shape)))).astype(np.float32)
    x, y = torch.from_numpy(phi), torch.from_numpy(y)
    model = ConditionalFFF(1, 1, 64, 2, beta=10.0, angle_dims=(0,))
    model.set_normalization(x, y)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    for _ in range(800):
        idx = torch.randint(0, len(x), (256,))
        loss = model.loss(x[idx], y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    for y0 in (-0.4, 0.4):
        s = model.sample(np.array([[y0]], np.float32), 4000)[0, :, 0].numpy()
        mean = np.angle(np.exp(1j * s).mean())
        assert abs(np.angle(np.exp(1j * (mean - np.pi - y0)))) < 0.15
        assert 0.15 < np.sqrt(-2 * np.log(np.abs(np.exp(1j * s).mean()))) < 0.45  # circular sd ~ 0.3


@torch.enable_grad()
def test_wrapped_latent_density_normalizes():
    torch.manual_seed(0)
    model = perturbed(ConditionalFFF(1, 1, 32, 2, angle_dims=(0,), latent='wrapped'))
    model.angle_std.fill_(0.5)
    x = torch.linspace(-math.pi, math.pi, 20001)[:-1, None]
    lp = model.log_prob(x, torch.full((len(x), 1), 0.3)).detach()
    assert abs((lp.exp().sum() * (x[1] - x[0])).item() - 1) < 2e-3
