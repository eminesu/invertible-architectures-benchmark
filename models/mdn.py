"""Mixture Density Network with full covariance matrices.

Implements Kruse (2020), "Technical report: Training Mixture Density Networks
with full covariance matrices", arXiv:2003.05739, Sec. 2. Used as the MDN
baseline of Kruse et al. (2021), arXiv:2101.10763, Eq. (9).

Each component i is parameterized by its weight logit, mean mu_i and the
upper-triangular Cholesky factor U_bar_i of its *precision* matrix:

    Sigma_i^{-1} = U_bar_i^T U_bar_i

The network emits N*(N+1)/2 unconstrained entries per component, laid out
exactly like FrEIA's `GaussianMixtureModel`:

    U_entries[..., :N]  -> diagonal, in log space  (U_bar_jj = exp(entry))
    U_entries[..., N:]  -> strictly upper triangle, row-major, used as-is

so the weights of a trained head are interchangeable with the FrEIA block
(see tests/test_mdn.py, which checks this numerically).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_tril_entries(n_dims):
    return n_dims * (n_dims + 1) // 2


def build_cholesky(U_entries, n_dims):
    """Unconstrained entries (..., N(N+1)/2) -> upper-triangular U_bar (..., N, N)
    with strictly positive diagonal."""
    rows, cols = torch.triu_indices(n_dims, n_dims, offset=1, device=U_entries.device)
    U = U_entries.new_zeros(*U_entries.shape[:-1], n_dims, n_dims)
    U[..., rows, cols] = U_entries[..., n_dims:]
    diag = torch.arange(n_dims, device=U_entries.device)
    U[..., diag, diag] = U_entries[..., :n_dims].exp()
    return U


def component_log_prob(x, mu, U_entries):
    """Per-component log-density, MDN report Eq. (13), including the
    -N/2 log(2 pi) constant so values are true log-densities.

    x:          (B, N)
    mu:         (B, K, N)
    U_entries:  (B, K, N(N+1)/2)
    returns:    (B, K)
    """
    n_dims = x.shape[-1]
    U = build_cholesky(U_entries, n_dims)
    z = (U @ (x[:, None, :] - mu)[..., None])[..., 0]           # (B, K, N)
    log_det = U_entries[..., :n_dims].sum(-1)                     # log|Sigma^-1|^(1/2)
    return -0.5 * (z ** 2).sum(-1) + log_det - 0.5 * n_dims * math.log(2 * math.pi)


def nll_exact(log_w, comp_log_prob):
    """Exact mixture NLL, MDN report Eq. (14), via logsumexp. Returns (B,)."""
    return -torch.logsumexp(log_w + comp_log_prob, dim=-1)


def nll_jensen(log_w, comp_log_prob):
    """Jensen upper bound on the NLL:  -log sum_i w_i p_i  <=  -sum_i w_i log p_i.

    This is the valid bound. Note that Eq. (15) of the MDN report (and FrEIA's
    `nll_upper_bound`) drops the w_i weighting outside the log and instead adds
    log w_i inside an unweighted sum over components; that expression is *not*
    an upper bound in general and forces every component onto every data point.
    It is available as `nll_report_eq15` for comparison only.

    The bound is linear in w, so on its own it pushes all weight onto the
    single best-fitting component per y; use it as a short warm-up only.
    """
    return -(log_w.exp() * comp_log_prob).sum(-1)


def nll_report_eq15(log_w, comp_log_prob):
    """MDN report Eq. (15) taken literally (same as FrEIA `nll_upper_bound`)."""
    return -(log_w + comp_log_prob).sum(-1)


LOSSES = {
    'exact': nll_exact,
    'jensen': nll_jensen,
    'eq15': nll_report_eq15,
}


class MDN(nn.Module):
    """Feed-forward trunk y -> hidden, then a linear head emitting the GMM
    parameters (weight logits, means, Cholesky entries) for K components."""

    def __init__(self, y_dims, x_dims, n_components, hidden=512, n_layers=4, activation='relu'):
        super().__init__()
        self.x_dims = x_dims
        self.n_components = n_components
        act = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh}[activation]

        layers, d = [], y_dims
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        self.trunk = nn.Sequential(*layers)

        self.n_U = n_tril_entries(x_dims)
        self.head = nn.Linear(d, n_components * (1 + x_dims + self.n_U))

        # Start from near-identity precision matrices: log-diagonal and
        # off-diagonals at zero, so the initial loss is well behaved.
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()

        # Data normalization, set via `set_normalization` from training data.
        self.register_buffer('y_mean', torch.zeros(y_dims))
        self.register_buffer('y_std', torch.ones(y_dims))
        self.register_buffer('x_mean', torch.zeros(x_dims))
        self.register_buffer('x_std', torch.ones(x_dims))

    def set_normalization(self, x, y):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0))

    def forward(self, y):
        """Returns (log_w (B,K), mu (B,K,N), U_entries (B,K,N(N+1)/2)) in
        *normalized* x-space."""
        h = self.trunk((y - self.y_mean) / self.y_std)
        out = self.head(h).view(-1, self.n_components, 1 + self.x_dims + self.n_U)
        log_w = F.log_softmax(out[..., 0], dim=-1)
        mu = out[..., 1:1 + self.x_dims]
        U_entries = out[..., 1 + self.x_dims:]
        return log_w, mu, U_entries

    def loss(self, x, y, kind='exact'):
        """Mean NLL of x | y in normalized x-space (constant offset vs. raw space
        is sum(log x_std), irrelevant for optimization)."""
        log_w, mu, U_entries = self(y)
        x_n = (x - self.x_mean) / self.x_std
        return LOSSES[kind](log_w, component_log_prob(x_n, mu, U_entries)).mean()

    @torch.no_grad()
    def sample(self, y, n_samples=1, generator=None):
        """Draw n_samples x ~ p(x | y) for each row of y. Returns (B, n_samples, N)
        in raw x-space.

        Sampling: pick component i ~ Cat(w), eta ~ N(0, I), then
            x = mu_i + U_bar_i^{-1} eta.
        Since Sigma^{-1} = U_bar^T U_bar, Sigma = U_bar^{-1} U_bar^{-T}, so the
        covariance factor is L = U_bar^{-1}. (The MDN report writes U_bar^{-T},
        which yields covariance (U_bar U_bar^T)^{-1} != Sigma; FrEIA uses
        U_bar^{-1}, as here.) U_bar is upper triangular, so this is a
        back substitution, not a general inverse.
        """
        log_w, mu, U_entries = self(y)
        B, K, N = mu.shape
        idx = torch.multinomial(log_w.exp(), n_samples, replacement=True, generator=generator)  # (B, S)
        mu_s = torch.gather(mu, 1, idx[..., None].expand(B, n_samples, N))
        U_s = build_cholesky(
            torch.gather(U_entries, 1, idx[..., None].expand(B, n_samples, self.n_U)), N)
        eta = torch.randn(B, n_samples, N, 1, device=y.device, generator=generator)
        x_n = mu_s + torch.linalg.solve_triangular(U_s, eta, upper=True)[..., 0]
        return x_n * self.x_std + self.x_mean


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
