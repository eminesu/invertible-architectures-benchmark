import pytest
import torch

from models.ilnn import (ILNN, LiquidFlow, LiquidNet, count_parameters, flow_params, liquid_params,
                         width_for_budget)

N, DY, B = 4, 2, 12


def randomize(model, seed=0, scale=0.3):
    """The zero-initialised readouts make a fresh flow the identity; give every
    parameter random values so the tests exercise the real transformation."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def make_flow(dtype=torch.float64):
    flow = LiquidFlow(N, n_blocks=3, hidden=16, unfolds=3, perm_seed=1).to(dtype)
    return randomize(flow)


def rand(*shape, seed=0, dtype=torch.float64):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_fresh_flow_is_identity():
    flow = LiquidFlow(N, n_blocks=3, hidden=16, unfolds=3)
    u = rand(B, N, dtype=torch.float32)
    v, log_det = flow(u)
    # permutations reorder the coordinates, but values and the log-det are untouched
    assert torch.equal(v.sort(1).values, u.sort(1).values)
    assert torch.equal(log_det, torch.zeros(B))


def test_inverse_is_exact():
    flow = make_flow()
    u = rand(B, N)
    v, _ = flow(u)
    assert torch.allclose(flow.inverse(v), u, atol=1e-10)
    assert torch.allclose(flow(flow.inverse(u))[0], u, atol=1e-10)


def test_log_det_matches_autograd_jacobian():
    flow = make_flow()
    u = rand(5, N)
    _, log_det = flow(u)
    for i in range(len(u)):
        J = torch.autograd.functional.jacobian(lambda w: flow(w[None])[0][0], u[i])
        assert torch.allclose(log_det[i], torch.linalg.slogdet(J)[1], atol=1e-9)


def test_liquid_cell_matches_closed_form_fused_step():
    """One unfold equals Eq. (11): (h + dt g Lambda) / (1 + dt (1/tau + g))."""
    net = randomize(LiquidNet(3, 2, hidden=5, unfolds=1, dt=0.3)).double()
    inp = rand(B, 3)
    h0 = torch.tanh(net.encoder(inp))
    g = torch.sigmoid(net.w_rec(h0) + net.w_in(inp))
    tau = torch.nn.functional.softplus(net.raw_tau) + 1e-3
    ref = (h0 + 0.3 * g * net.lam) / (1 + 0.3 * (1 / tau + g))
    assert torch.allclose(net.cell(inp), ref)


def test_liquid_state_stays_bounded():
    """The fused step is a convex-type update: with g in (0, 1) and tau > 0 the state
    cannot blow up, however many unfolds (stability claim around Eq. 12)."""
    net = randomize(LiquidNet(3, 2, hidden=8, unfolds=200, dt=0.5), scale=1.0).double()
    h = net.cell(rand(64, 3) * 10)
    assert torch.isfinite(h).all() and h.abs().max() < 1e3


def test_flow_params_formula_and_budget():
    flow = LiquidFlow(N, n_blocks=4, hidden=37, unfolds=2)
    assert count_parameters(flow) == flow_params(N, 4, 37)
    assert count_parameters(LiquidNet(3, 2, 11, 4, 0.25)) == liquid_params(3, 2, 11)
    h = width_for_budget(N, 8, 3_000_000)
    assert flow_params(N, 8, h) <= 3_000_000 < flow_params(N, 8, h + 1)


def small_ilnn(**kw):
    return randomize(ILNN(N, DY, n_blocks=3, hidden=16, unfolds=3, **kw))


def test_ilnn_dim_z_matches_paper_table():
    assert ILNN(4, 2, n_blocks=2, hidden=8).z_dims == 2   # kinematics
    assert ILNN(4, 1, n_blocks=2, hidden=8).z_dims == 3   # ballistics


@pytest.mark.parametrize('latent_mmd', ['joint', 'marginal'])
def test_ilnn_loss_terms_are_finite_and_differentiable(latent_mmd):
    m = small_ilnn(latent_mmd=latent_mmd)
    t = m.losses(rand(64, N, dtype=torch.float32), rand(64, DY, seed=1, dtype=torch.float32))
    assert set(t) == {'y', 'z', 'x'} == set(m.loss_terms)
    (t['y'] + t['z'] + t['x']).backward()
    assert all(torch.isfinite(v) for v in t.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_backward_mmd_term_reaches_the_inverse_path():
    """L_x must produce gradients on its own (it is computed through f^-1)."""
    m = small_ilnn()
    m.losses(rand(64, N, dtype=torch.float32), rand(64, DY, seed=1, dtype=torch.float32))['x'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_sample_shape_and_roundtrip():
    m = small_ilnn().double()
    m.x_mean.copy_(torch.tensor([0.1, -0.2, 0.3, 5.0])); m.x_std.copy_(torch.tensor([0.5, 1.0, 2.0, 4.0]))
    m.y_mean.copy_(torch.tensor([1.0, -1.0])); m.y_std.copy_(torch.tensor([2.0, 0.5]))
    y = rand(3, DY, seed=5)
    g = torch.Generator().manual_seed(0)
    x = m.sample(y, n_samples=7, generator=g, chunk=10)   # chunk < B*n exercises the loop
    assert x.shape == (3, 7, N)
    y_hat_n, _, _ = m(x.reshape(-1, N))
    assert torch.allclose(y_hat_n * m.y_std + m.y_mean, y.repeat_interleave(7, 0), atol=1e-8)  # y pinned to y*


def test_sampling_is_reproducible_with_generator():
    m = small_ilnn()
    y = rand(2, DY, seed=3, dtype=torch.float32)
    a = m.sample(y, 5, generator=torch.Generator().manual_seed(7))
    b = m.sample(y, 5, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_ml_loss_is_finite_differentiable_and_uses_only_nll():
    model = randomize(ILNN(N, DY, n_blocks=2, hidden=8, unfolds=2, loss='ml').to(torch.float64), scale=0.1)
    assert model.loss_terms == ('nll',)
    terms = model.losses(rand(B, N, seed=1), rand(B, DY, seed=2))
    assert set(terms) == {'nll', 'y', 'z'} and all(torch.isfinite(v) for v in terms.values())
    terms['nll'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_loss_is_the_papers_eq16():
    model = ILNN(N, DY, n_blocks=2, hidden=8, unfolds=2)
    assert model.loss_terms == ('y', 'z', 'x')
