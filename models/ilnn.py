"""Invertible Liquid Neural Network (ILNN), Zhang et al. (2025), Sci. Rep. 15:42311,
doi:10.1038/s41598-025-26422-1, Secs. "Liquid neural networks" and "Invertible liquid
neural networks (ILNNs)", Eqs. (9)-(16).

The ILNN is the INN of models/inn.py (x -> [y_hat, z], Eq. 13) whose coupling subnets
are *liquid cells* instead of MLPs ("liquid-conditioned affine coupling", Algorithm 2).
Everything else (GLOW-style two-sub-step coupling, Eq. 15, soft-clamped log-scales,
fixed random permutations between blocks, zero-initialised readout so the flow starts
as the identity, standardised units, sampling) is identical to models.inn.INN, so any
difference in the benchmark comes from the conditioner.

Coupling block (Eq. 15), (s, t) produced by liquid cells:

    v1 = u1 * exp(s2(u2)) + t2(u2)
    v2 = u2 * exp(s1(v1)) + t1(v1)

Liquid cell (Eq. 9, fused explicit-implicit Euler step of Eq. 11), input I = the half of
the channels the coupling leaves untouched:

    h0     = tanh(W_e I + b_e)                        (context encoder, Algorithm 1)
    g      = sigmoid(W_h h + W_i I + b)               (> 0)
    h     <- (h + dt * g * Lambda) / (1 + dt * (1/tau + g)),   tau = softplus(raw) + 1e-3
    (s, t) = W_o h_K + b_o                            (linear readout, zero-initialised)

repeated `unfolds` times with shared weights. Because (s, t) depend only on the
untouched half, every block is exactly invertible with a triangular Jacobian. This is
the reading of "liquid-conditioned affine coupling"; a liquid update applied directly
to the transformed channels would not be invertible in general.

Loss, Eq. (16), with the three terms logged separately:
    y : |f_y(x) - y|^2
    z : MMD(q(y, z), p(y)p(z))   'joint': [y_hat.detach(), z] vs [y, eps], like INN 'l2mmd';
                                 'marginal': z vs eps alone
    x : MMD(p(x), q_g(y, z)),    q_g = f^-1([y, eps]) with the data y   (backward marginal)
combine as w_y * y + w_z * z + w_x * x.

Not specified by the paper (Algorithms 1/2 are figures): number of blocks, hidden size,
number of unfolds, dt, the nonlinearity g, clamp, w_y, w_z, w_x, optimizer. The time
argument of g(x, I, t) is dropped: the toy problems are static.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.mmd import mmd_torch
from models.inn import _FlowModel
from models.mdn import count_parameters  # noqa: F401  (shared helper, re-exported for the training scripts)


class LiquidNet(nn.Module):
    """Liquid cell (Eq. 9/11) + linear readout: d_in -> (s_raw, t) of size 2 * d_out."""

    def __init__(self, d_in, d_out, hidden, unfolds, dt):
        super().__init__()
        self.unfolds, self.dt = unfolds, dt
        self.encoder = nn.Linear(d_in, hidden)               # context -> h0
        self.w_rec = nn.Linear(hidden, hidden, bias=False)   # recurrent part of g
        self.w_in = nn.Linear(d_in, hidden)                  # input part of g (+ bias)
        self.raw_tau = nn.Parameter(torch.zeros(hidden))     # tau = softplus(raw_tau) + 1e-3
        self.lam = nn.Parameter(torch.zeros(hidden))         # Lambda in Eq. (9)
        self.readout = nn.Linear(hidden, 2 * d_out)
        nn.init.zeros_(self.readout.weight)                  # identity at initialisation
        nn.init.zeros_(self.readout.bias)

    def cell(self, inp):
        h = torch.tanh(self.encoder(inp))
        inv_tau = 1.0 / (F.softplus(self.raw_tau) + 1e-3)
        for _ in range(self.unfolds):
            g = torch.sigmoid(self.w_rec(h) + self.w_in(inp))
            h = (h + self.dt * g * self.lam) / (1.0 + self.dt * (inv_tau + g))
        return h

    def forward(self, inp):
        return self.readout(self.cell(inp))


def liquid_params(d_in, d_out, hidden):
    """Trainable parameters of a LiquidNet, in closed form (checked in tests)."""
    return (2 * d_in * hidden + 4 * hidden + hidden * hidden      # encoder, w_in, w_rec, raw_tau, lam
            + 2 * d_out * hidden + 2 * d_out)                     # readout


def flow_params(dim, n_blocks, hidden):
    """Trainable parameters of a LiquidFlow, in closed form (checked in tests)."""
    d1, d2 = dim // 2, dim - dim // 2
    return n_blocks * (liquid_params(d1, d2, hidden) + liquid_params(d2, d1, hidden))


def width_for_budget(dim, n_blocks, target):
    """Largest liquid hidden size such that the whole flow stays at or under `target`."""
    h = 1
    while flow_params(dim, n_blocks, h + 1) <= target:
        h += 1
    return h


class LiquidCouplingBlock(nn.Module):
    """GLOW-style affine coupling block with liquid conditioners; see the module docstring."""

    def __init__(self, dim, hidden, unfolds, dt, clamp):
        super().__init__()
        self.d1, self.d2 = dim // 2, dim - dim // 2
        self.clamp = clamp
        self.net1 = LiquidNet(self.d1, self.d2, hidden, unfolds, dt)  # v1 -> params of u2
        self.net2 = LiquidNet(self.d2, self.d1, hidden, unfolds, dt)  # u2 -> params of u1

    def _params(self, net, inp, d_out):
        out = net(inp)
        s, t = out[:, :d_out], out[:, d_out:]
        return self.clamp * 0.636 * torch.atan(s), t  # soft clamp, as in models.inn

    def forward(self, u):
        u1, u2 = u[:, :self.d1], u[:, self.d1:]
        s2, t2 = self._params(self.net2, u2, self.d1)
        v1 = u1 * s2.exp() + t2
        s1, t1 = self._params(self.net1, v1, self.d2)
        v2 = u2 * s1.exp() + t1
        return torch.cat([v1, v2], dim=1), s1.sum(1) + s2.sum(1)

    def inverse(self, v):
        v1, v2 = v[:, :self.d1], v[:, self.d1:]
        s1, t1 = self._params(self.net1, v1, self.d2)
        u2 = (v2 - t1) * (-s1).exp()
        s2, t2 = self._params(self.net2, u2, self.d1)
        u1 = (v1 - t2) * (-s2).exp()
        return torch.cat([u1, u2], dim=1)


class LiquidFlow(nn.Module):
    """Stack of liquid coupling blocks with fixed random permutations in between
    (none after the last block); same layout as models.inn.Flow, unconditional."""

    def __init__(self, dim, n_blocks=8, hidden=256, unfolds=4, dt=0.25, clamp=2.0, perm_seed=0):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            LiquidCouplingBlock(dim, hidden, unfolds, dt, clamp) for _ in range(n_blocks))
        g = torch.Generator().manual_seed(perm_seed)
        for i in range(n_blocks - 1):
            perm = torch.randperm(dim, generator=g)
            self.register_buffer(f'perm{i}', perm)
            self.register_buffer(f'inv_perm{i}', torch.argsort(perm))

    def forward(self, u):
        """u (B, dim) -> (v (B, dim), log|det dv/du| (B,))."""
        log_det = u.new_zeros(len(u))
        for i, block in enumerate(self.blocks):
            u, ld = block(u)
            log_det = log_det + ld
            if i < len(self.blocks) - 1:
                u = u[:, getattr(self, f'perm{i}')]
        return u, log_det

    def inverse(self, v):
        for i in reversed(range(len(self.blocks))):
            if i < len(self.blocks) - 1:
                v = v[:, getattr(self, f'inv_perm{i}')]
            v = self.blocks[i].inverse(v)
        return v


class ILNN(_FlowModel):
    """x -> [y_hat, z] with a liquid-conditioned flow, loss Eq. (16); see the module
    docstring. dim(y_hat) = dim(y), dim(z) = dim(x) - dim(y), like models.inn.INN."""

    loss_terms = ('y', 'z', 'x')

    def __init__(self, x_dims, y_dims, n_blocks=8, hidden=256, unfolds=4, dt=0.25, clamp=2.0,
                 latent_mmd='joint', perm_seed=0, loss='mmd', sigma_y=0.05):
        super().__init__(x_dims, y_dims)
        assert latent_mmd in ('joint', 'marginal')
        assert loss in ('mmd', 'ml')
        self.z_dims = x_dims - y_dims
        self.latent_mmd = latent_mmd
        # loss='mmd' is the paper's Eq. (16). loss='ml' trains the same flow by maximum likelihood
        # (the INN's Eq. 4: Gaussian y_hat | y with std sigma_y, standard normal z, log-det), which is
        # a deviation from the paper; report it as a separate variant.
        self.loss_kind, self.sigma_y = loss, sigma_y
        if loss == 'ml':
            self.loss_terms = ('nll',)
        self.flow = LiquidFlow(x_dims, n_blocks, hidden, unfolds, dt, clamp, perm_seed)

    def forward(self, x):
        """Raw x -> (y_hat_n, z, log_det), the first two in standardized units."""
        x_n = (x - self.x_mean) / self.x_std
        out, log_det = self.flow(x_n)
        return out[:, :self.y_dims], out[:, self.y_dims:], log_det

    def losses(self, x, y):
        x_n, y_n = self._norm(x, y)
        out, log_det = self.flow(x_n)
        y_hat, z = out[:, :self.y_dims], out[:, self.y_dims:]
        y_err = ((y_hat - y_n) ** 2).sum(1).mean()
        if self.loss_kind == 'ml':
            nll = (0.5 * (((y_hat - y_n) ** 2).sum(1) / self.sigma_y ** 2 + (z ** 2).sum(1)) - log_det).mean()
            return {'nll': nll, 'y': y_err.detach(), 'z': (z ** 2).sum(1).mean().detach()}
        eps = torch.randn_like(z)
        if self.latent_mmd == 'joint':
            a, b = torch.cat([y_hat.detach(), z], 1), torch.cat([y_n, eps], 1)
        else:
            a, b = z, eps
        x_gen = self._decode(y_n, torch.randn_like(z))   # q_g(y, z) with the data y
        return {'y': y_err, 'z': mmd_torch(a, b), 'x': mmd_torch(x_gen, x_n)}

    def _decode(self, y_n, z):
        return self.flow.inverse(torch.cat([y_n, z], dim=1))