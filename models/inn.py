"""Invertible Neural Network (INN) and conditional INN (cINN), Kruse et al. (2021),
arXiv:2101.10763, Sec. 2, Eqs. (1)-(5). Both are RealNVP-style flows built from
affine coupling layers (Dinh et al. 2016); the INN follows Ardizzone et al. (2019).

Building block: a GLOW-style coupling block (the same family as FrEIA's
`GLOWCouplingBlock`). Both halves of the input are transformed per block,

    v1 = u1 * exp(s2(u2, c)) + t2(u2, c)
    v2 = u2 * exp(s1(v1, c)) + t1(v1, c)

with soft-clamped log-scales  s = clamp * 0.636 * atan(s_raw)  (|s| <= clamp), a fixed
random permutation between blocks, and every subnet a plain MLP whose last layer is
zero-initialised, so the flow starts as the identity. `c` is the condition; it is
empty for the INN and the standardized observation y for the cINN, where it is fed
to *every* subnet in both the forward and the inverse pass (paper Sec. 2, "Conditional
INNs").

INN       x -> [y_hat, z]   dim(y_hat) = dim(y), dim(z) = dim(x) - dim(y)
          (no zero padding: the table in the paper gives dim(z) = 2 on kinematics and
          3 on ballistics, i.e. dim(x) - dim(y)). Two training losses:
            'ml'     Eq. (4): 1/2 (|y_hat - y|^2 / sigma^2 + |z|^2) - log|det J|
            'l2mmd'  Eq. (3): L2(y) + alpha * MMD(z)           (alpha = weight w_z)
cINN      x -> z | y        dim(z) = dim(x)
          loss Eq. (5): 1/2 |z|^2 - log|det J|

All losses are computed in standardized x / y units (training-set mean / std), like
models/mdn.py and models/autoencoder.py. `sample` takes and returns raw units.

Test time: z ~ N(0, I) and, for the INN, y_hat fixed to y*; x = f^{-1}([y*, z])
resp. f^{-1}(z; y*). Not specified by the paper (see docs/inn.md): subnet depth and
width, number of blocks, clamp, sigma, alpha, optimizer.
"""

import math

import torch
import torch.nn as nn

from metrics.mmd import mmd_torch
from models.mdn import count_parameters  # noqa: F401  (shared helper, re-exported for the training scripts)

ACTIVATIONS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}


def mlp(d_in, d_out, hidden, n_layers, activation):
    """n_layers hidden Linear+activation layers, then a linear output layer."""
    act = ACTIVATIONS[activation]
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def mlp_params(d_in, d_out, hidden, n_layers):
    return (d_in * hidden + hidden) + (n_layers - 1) * (hidden * hidden + hidden) + hidden * d_out + d_out


def flow_params(dim, cond_dim, n_blocks, hidden, n_layers):
    """Trainable parameters of a Flow, in closed form (checked in tests)."""
    d1, d2 = dim // 2, dim - dim // 2
    per_block = (mlp_params(d2 + cond_dim, 2 * d1, hidden, n_layers)
                 + mlp_params(d1 + cond_dim, 2 * d2, hidden, n_layers))
    return n_blocks * per_block


def width_for_budget(dim, cond_dim, n_blocks, n_layers, target):
    """Largest subnet width such that the whole flow stays at or under `target`."""
    h = 1
    while flow_params(dim, cond_dim, n_blocks, h + 1, n_layers) <= target:
        h += 1
    return h


class CouplingBlock(nn.Module):
    """GLOW-style affine coupling block; see the module docstring."""

    def __init__(self, dim, cond_dim, hidden, n_layers, activation, clamp):
        super().__init__()
        self.d1, self.d2 = dim // 2, dim - dim // 2
        self.clamp = clamp
        self.subnet1 = mlp(self.d1 + cond_dim, 2 * self.d2, hidden, n_layers, activation)  # v1 (+c) -> params of u2
        self.subnet2 = mlp(self.d2 + cond_dim, 2 * self.d1, hidden, n_layers, activation)  # u2 (+c) -> params of u1
        for net in (self.subnet1, self.subnet2):  # identity at initialisation
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def _params(self, net, inp, cond, d_out):
        if cond is not None:
            inp = torch.cat([inp, cond], dim=1)
        out = net(inp)
        s, t = out[:, :d_out], out[:, d_out:]
        return self.clamp * 0.636 * torch.atan(s), t  # soft clamp, as in FrEIA

    def forward(self, u, cond=None):
        u1, u2 = u[:, :self.d1], u[:, self.d1:]
        s2, t2 = self._params(self.subnet2, u2, cond, self.d1)
        v1 = u1 * s2.exp() + t2
        s1, t1 = self._params(self.subnet1, v1, cond, self.d2)
        v2 = u2 * s1.exp() + t1
        return torch.cat([v1, v2], dim=1), s1.sum(1) + s2.sum(1)

    def inverse(self, v, cond=None):
        v1, v2 = v[:, :self.d1], v[:, self.d1:]
        s1, t1 = self._params(self.subnet1, v1, cond, self.d2)
        u2 = (v2 - t1) * (-s1).exp()
        s2, t2 = self._params(self.subnet2, u2, cond, self.d1)
        u1 = (v1 - t2) * (-s2).exp()
        return torch.cat([u1, u2], dim=1)


class Flow(nn.Module):
    """Stack of coupling blocks with fixed random permutations in between
    (none after the last block, so the output layout is not scrambled)."""

    def __init__(self, dim, cond_dim=0, n_blocks=8, hidden=256, n_layers=3, activation='relu',
                 clamp=2.0, perm_seed=0):
        super().__init__()
        self.dim, self.cond_dim = dim, cond_dim
        self.blocks = nn.ModuleList(
            CouplingBlock(dim, cond_dim, hidden, n_layers, activation, clamp) for _ in range(n_blocks))
        g = torch.Generator().manual_seed(perm_seed)
        for i in range(n_blocks - 1):
            perm = torch.randperm(dim, generator=g)
            self.register_buffer(f'perm{i}', perm)
            self.register_buffer(f'inv_perm{i}', torch.argsort(perm))

    def forward(self, u, cond=None):
        """u (B, dim) -> (v (B, dim), log|det dv/du| (B,))."""
        log_det = u.new_zeros(len(u))
        for i, block in enumerate(self.blocks):
            u, ld = block(u, cond)
            log_det = log_det + ld
            if i < len(self.blocks) - 1:
                u = u[:, getattr(self, f'perm{i}')]
        return u, log_det

    def inverse(self, v, cond=None):
        for i in reversed(range(len(self.blocks))):
            if i < len(self.blocks) - 1:
                v = v[:, getattr(self, f'inv_perm{i}')]
            v = self.blocks[i].inverse(v, cond)
        return v


class _FlowModel(nn.Module):
    """Shared plumbing: standardization buffers and chunked sampling."""

    def __init__(self, x_dims, y_dims):
        super().__init__()
        self.x_dims, self.y_dims = x_dims, y_dims
        self.register_buffer('y_mean', torch.zeros(y_dims))
        self.register_buffer('y_std', torch.ones(y_dims))
        self.register_buffer('x_mean', torch.zeros(x_dims))
        self.register_buffer('x_std', torch.ones(x_dims))

    def set_normalization(self, x, y):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0))

    def _norm(self, x, y):
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def _decode(self, y_n, z):
        """Standardized (y, z) rows -> standardized x. Implemented by subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, y, n_samples=1, generator=None, chunk=65_536):
        """Draw n_samples x ~ p(x | y) for each row of y, z ~ N(0, I). Returns
        (B, n_samples, N) in raw x-space. Rows are processed in chunks to bound memory."""
        B = len(y)
        y_n = ((y - self.y_mean) / self.y_std).repeat_interleave(n_samples, dim=0)  # (B*n, dim_y)
        out = []
        for s in range(0, len(y_n), chunk):
            y_c = y_n[s:s + chunk]
            z = torch.randn(len(y_c), self.z_dims, device=y.device, dtype=y.dtype, generator=generator)
            out.append(self._decode(y_c, z))
        x_n = torch.cat(out).view(B, n_samples, self.x_dims)
        return x_n * self.x_std + self.x_mean


class INN(_FlowModel):
    """x -> [y_hat, z] with an unconditional flow; see the module docstring.

    loss='ml'     terms: nll (Eq. 4, constants dropped); y, z are logged only
    loss='l2mmd'  terms: y (Eq. 1), z (MMD, Eq. 2); combine as w_y * y + w_z * z (Eq. 3)
    latent_mmd='joint' takes the MMD on [y_hat, z] vs [y, eps] (y_hat detached) like
    models/autoencoder.py, 'marginal' on z vs eps alone, as literally written in Eq. (3).
    """

    def __init__(self, x_dims, y_dims, n_blocks=8, hidden=256, n_layers=3, activation='relu',
                 clamp=2.0, loss='ml', sigma_y=0.05, latent_mmd='joint', perm_seed=0):
        super().__init__(x_dims, y_dims)
        assert loss in ('ml', 'l2mmd') and latent_mmd in ('joint', 'marginal')
        self.z_dims = x_dims - y_dims
        self.loss_kind, self.sigma_y, self.latent_mmd = loss, sigma_y, latent_mmd
        self.loss_terms = ('nll',) if loss == 'ml' else ('y', 'z')
        self.flow = Flow(x_dims, 0, n_blocks, hidden, n_layers, activation, clamp, perm_seed)

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
        return {'y': y_err, 'z': mmd_torch(a, b)}

    def _decode(self, y_n, z):
        return self.flow.inverse(torch.cat([y_n, z], dim=1))


class CINN(_FlowModel):
    """x -> z | y with a conditional flow, loss Eq. (5); see the module docstring."""

    loss_terms = ('nll',)

    def __init__(self, x_dims, y_dims, n_blocks=8, hidden=256, n_layers=3, activation='relu',
                 clamp=2.0, perm_seed=0):
        super().__init__(x_dims, y_dims)
        self.z_dims = x_dims
        self.flow = Flow(x_dims, y_dims, n_blocks, hidden, n_layers, activation, clamp, perm_seed)

    def forward(self, x, y):
        """Raw (x, y) -> (z, log_det) with z in the latent space and log_det the
        log-Jacobian of the *standardized* x -> z map."""
        x_n, y_n = self._norm(x, y)
        return self.flow(x_n, y_n)

    def losses(self, x, y):
        z, log_det = self(x, y)
        return {'nll': (0.5 * (z ** 2).sum(1) - log_det).mean()}

    @torch.no_grad()
    def nll_raw(self, x, y):
        """Per-sample -log p(x | y) in raw x-space, a true density: includes the
        0.5 N log(2 pi) constant and the log x_std change of variables. Comparable
        with the MDN's reported NLL (experiments/train_mdn.py::evaluate_nll)."""
        z, log_det = self(x, y)
        return (0.5 * (z ** 2).sum(1) + 0.5 * self.x_dims * math.log(2 * math.pi)
                - log_det + self.x_std.log().sum())

    def _decode(self, y_n, z):
        return self.flow.inverse(z, y_n)

