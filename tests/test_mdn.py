import math

import pytest
import torch
from torch.distributions import MultivariateNormal

from models.mdn import (MDN, build_cholesky, component_log_prob, nll_exact, nll_jensen,
                        n_tril_entries)

B, K, N = 16, 5, 4


def random_params(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, generator=g, dtype=dtype)
    log_w = torch.log_softmax(torch.randn(B, K, generator=g, dtype=dtype), -1)
    mu = torch.randn(B, K, N, generator=g, dtype=dtype)
    U = 0.5 * torch.randn(B, K, n_tril_entries(N), generator=g, dtype=dtype)
    return x, log_w, mu, U


def test_cholesky_is_upper_with_positive_diagonal():
    _, _, _, U_entries = random_params()
    U = build_cholesky(U_entries, N)
    assert torch.equal(U, torch.triu(U))
    assert (torch.diagonal(U, dim1=-2, dim2=-1) > 0).all()


def test_component_log_prob_matches_torch_mvn():
    x, _, mu, U_entries = random_params()
    U = build_cholesky(U_entries, N)
    precision = U.transpose(-1, -2) @ U
    ref = MultivariateNormal(mu, precision_matrix=precision).log_prob(x[:, None, :])
    assert torch.allclose(component_log_prob(x, mu, U_entries), ref, atol=1e-10)


def test_matches_freia_gaussian_mixture_model():
    """Same entry layout and same latent codes / log-Jacobians as FrEIA."""
    freia = pytest.importorskip('FrEIA.modules')
    x, log_w, mu, U_entries = random_params()
    gmm = freia.GaussianMixtureModel([(N,)], [(K,), (K, N), (K, n_tril_entries(N)), (0,)])
    (z,), log_jac = gmm([x], [log_w.exp(), mu, U_entries, None])

    ours = component_log_prob(x, mu, U_entries)
    theirs = -0.5 * (z ** 2).sum(-1) + log_jac - 0.5 * N * math.log(2 * math.pi)
    assert torch.allclose(ours, theirs, atol=1e-10)

    # FrEIA's (non-logsumexp) NLL equals ours up to the dropped Gaussian constant.
    freia_nll = gmm.nll_loss(log_w.exp(), z, log_jac)
    assert torch.allclose(nll_exact(log_w, ours), freia_nll + 0.5 * N * math.log(2 * math.pi), atol=1e-8)


def test_exact_nll_is_stable_far_from_mean():
    x, log_w, mu, U_entries = random_params()
    nll = nll_exact(log_w, component_log_prob(x + 1e3, mu, U_entries))
    assert torch.isfinite(nll).all()


def test_jensen_is_upper_bound():
    x, log_w, mu, U_entries = random_params()
    lp = component_log_prob(x, mu, U_entries)
    assert (nll_jensen(log_w, lp) >= nll_exact(log_w, lp) - 1e-12).all()


def test_sampling_reproduces_full_covariance():
    """Samples from a single fixed component must have covariance (U^T U)^{-1},
    including off-diagonals. This catches the U^{-T} vs U^{-1} mix-up."""
    torch.manual_seed(0)
    model = MDN(y_dims=1, x_dims=N, n_components=1, hidden=8, n_layers=1).double()
    g = torch.Generator().manual_seed(1)
    U_entries = torch.randn(n_tril_entries(N), generator=g, dtype=torch.float64) * 0.6
    mu = torch.randn(N, generator=g, dtype=torch.float64)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.cat([torch.zeros(1, dtype=torch.float64), mu, U_entries]))

    samples = model.sample(torch.zeros(1, 1, dtype=torch.float64), n_samples=400_000)[0]
    U = build_cholesky(U_entries, N)
    cov_true = torch.linalg.inv(U.T @ U)
    cov_emp = torch.cov(samples.T)
    assert torch.allclose(samples.mean(0), mu, atol=0.02)
    assert torch.allclose(cov_emp, cov_true, atol=0.03 * cov_true.abs().max())


def test_sampling_component_choice_follows_weights():
    torch.manual_seed(0)
    model = MDN(y_dims=1, x_dims=1, n_components=2, hidden=8, n_layers=1).double()
    with torch.no_grad():
        model.head.weight.zero_()
        # per component: [logit, mu, log-diag]; tight components at -5 and +5
        model.head.bias.copy_(torch.tensor([math.log(0.2), -5., 3., math.log(0.8), 5., 3.]))
    s = model.sample(torch.zeros(1, 1, dtype=torch.float64), n_samples=100_000)[0, :, 0]
    assert abs((s > 0).double().mean().item() - 0.8) < 0.01
