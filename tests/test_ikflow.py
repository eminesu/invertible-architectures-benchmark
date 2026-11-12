"""IKFlow (models/ikflow.py, experiments/train_ikflow.py): architecture against the paper's Table I,
invertibility with the SoftFlow condition, padding / noise / latent scaling, Lookahead."""

import pytest

torch = pytest.importorskip('torch')

from models.ikflow import IKFlow, count_parameters, flow_params, width_for_budget  # noqa: E402

N, DY, B = 7, 9, 16


def perturb(model, std=0.05, seed=0):
    """Random weights: the flow is the identity at initialisation (zero-initialised last layers)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in model.parameters():
            q.copy_(torch.randn(q.shape, generator=g) * std)
    return model


@pytest.mark.parametrize('width,layers,reported', [(15, 9, 3.836e7), (15, 12, 5.115e7), (15, 16, 6.820e7),
                                                   (9, 12, 5.093e7), (15, 8, 3.410e7)])
def test_table1_parameter_counts(width, layers, reported):
    """Paper Table I: 3 x 1024 subnets conditioned on the 7-D pose + the SoftFlow level (7 + 1)."""
    assert abs(flow_params(width, 7 + 1, layers, 1024, 3) - reported) / reported < 5e-4


def test_parameter_count_matches_formula():
    m = IKFlow(N, DY, n_blocks=3, hidden=16, n_layers=3)
    assert (m.width, m.z_dims, m.pad) == (9, 9, 2)
    assert count_parameters(m) == flow_params(9, DY + 1, 3, 16, 3)
    h = width_for_budget(9, DY + 1, 12, 3, 3_000_000)
    assert flow_params(9, DY + 1, 12, h, 3) <= 3_000_000 < flow_params(9, DY + 1, 12, h + 1, 3)


def test_width_below_dof_raises():
    with pytest.raises(ValueError):
        IKFlow(N, DY, width=N - 1, n_blocks=1, hidden=8)


def test_flow_inverts_with_the_softflow_condition():
    m = perturb(IKFlow(N, DY, n_blocks=3, hidden=16)).double()
    u = torch.randn(B, m.width, dtype=torch.float64)
    cond = torch.randn(B, DY + 1, dtype=torch.float64)
    v, log_det = m.flow(u, cond)
    assert torch.allclose(m.flow.inverse(v, cond), u, atol=1e-8) and torch.isfinite(log_det).all()


def test_losses_are_finite_and_differentiable():
    m = perturb(IKFlow(N, DY, n_blocks=3, hidden=16))
    x, y = torch.randn(32, N), torch.randn(32, DY)
    m.set_normalization(x, y)
    t = m.losses(x, y)
    assert set(t) == {'nll'} == set(m.loss_terms) and torch.isfinite(t['nll'])
    t['nll'].backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters())


def test_softflow_noise_is_in_raw_joint_units():
    m = IKFlow(N, DY, n_blocks=1, hidden=8, softflow_scale=1e-3)
    m.x_std.copy_(torch.arange(1, N + 1, dtype=torch.float32))
    s = m._noise_scale()
    assert s.shape == (m.width,)
    assert torch.allclose(s[:N] * m.x_std, torch.full((N,), 1e-3))
    assert torch.allclose(s[N:], torch.full((m.pad,), 1e-3))


def test_decode_uses_noise_level_zero_and_drops_the_padding():
    m = perturb(IKFlow(N, DY, n_blocks=3, hidden=16, latent_scale=0.5)).double()
    y_n = torch.randn(B, DY, dtype=torch.float64)
    z = torch.randn(B, m.width, dtype=torch.float64)
    ref = m.flow.inverse(z * 0.5, torch.cat([y_n, torch.zeros(B, 1, dtype=torch.float64)], dim=1))[:, :N]
    assert m._decode(y_n, z).shape == (B, N) and torch.allclose(m._decode(y_n, z), ref)


def test_sampling_shape_and_latent_scale():
    m = perturb(IKFlow(N, DY, n_blocks=3, hidden=16, latent_scale=0.0))
    x, y = torch.randn(32, N), torch.randn(32, DY)
    m.set_normalization(x, y)
    s = m.sample(y[:3], n_samples=5)
    assert s.shape == (3, 5, N) and torch.isfinite(s).all()
    assert torch.allclose(s, s[:, :1].expand_as(s))   # scale 0: every z maps to the same joint vector
    m.latent_scale = 1.0
    assert (m.sample(y[:3], n_samples=5).std(1) > 0).all()


def test_build_model_from_config():
    from experiments.train_ikflow import build_model
    cfg = dict(width_resolved=9, blocks=2, hidden_resolved=8, layers=3, activation='leaky_relu', clamp=2.0,
               softflow_scale=1e-3, latent_scale=0.25, seed=0)
    m = build_model(cfg, N, DY)
    assert isinstance(m, IKFlow) and m.latent_scale == 0.25 and count_parameters(m) == flow_params(9, DY + 1, 2, 8, 3)


def test_lookahead_pulls_the_fast_weights_to_the_slow_ones():
    from experiments.train_ikflow import Lookahead
    w = torch.nn.Parameter(torch.zeros(3))
    opt = Lookahead(torch.optim.SGD([w], lr=1.0), k=2, alpha=0.5)
    for _ in range(2):
        w.grad = torch.ones(3)
        opt.step()
    # two SGD steps take the fast weights to -2; slow = 0 + 0.5 * (-2 - 0) = -1, and the fast weights restart there
    assert torch.allclose(w, torch.full((3,), -1.0))


def test_ema_tracks_the_weights_and_restores_them():
    from experiments.train_ikflow import EMA
    w = torch.nn.Parameter(torch.zeros(2))
    ema = EMA([w], decay=0.5)
    with torch.no_grad():
        w.fill_(1.0)
    ema.update(step=10 ** 9)   # warm-up factor ~1: the decay is 0.5
    assert torch.allclose(ema.shadow[0], torch.full((2,), 0.5))
    ema.swap_in()
    assert torch.allclose(w, torch.full((2,), 0.5))
    ema.swap_out()
    assert torch.allclose(w, torch.full((2,), 1.0))


def test_steps_replace_epochs():
    from experiments.train_ikflow import resolve_epochs
    assert resolve_epochs(50, None, 1000) == 50
    assert resolve_epochs(50, 30000, 1562) == 19
    assert resolve_epochs(50, 10, 1000) == 1
