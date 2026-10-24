"""Conditional Free-Form Flows (FFF), Euclidean and on a torus x R^k (M-FFF).

Draxler, Sorrenson, Zimmermann, Rousselot, Köthe (2024), "Free-form Flows: Make
Any Architecture a Normalizing Flow", AISTATS, arXiv:2310.16624 [FFF];
Sorrenson, Draxler, Rousselot, Hummerich, Köthe (2024), "Learning Distributions
on Manifolds with Free-Form Flows", NeurIPS, arXiv:2312.09852 [M-FFF].

An encoder f: (x, y) -> z and a separate decoder g: (z, y) -> x, both plain
dimension-preserving residual MLPs (no invertibility constraint). Sampling is
one decoder pass: z ~ p(z), x = g(z, y*). No ODE, so 1 NFE per sample.

Training, per sample (FFF Eq. 7 / Alg. 1, the authors' `fff.loss`):

    L = -log p(z) - v^T f'(x) SG(g'(z) v) + beta ||x - g(f(x))||^2,   z = f(x)

The middle term is a one-sample Hutchinson surrogate whose *gradient* w.r.t. the
encoder is that of log|det f'(x)| when g'(z) = f'(x)^-1; the reconstruction
term (weight beta) is what makes g the inverse of f. g'(z) v is a forward-mode
JVP through the decoder, v^T f'(x) a backward-mode VJP through the encoder, so
the cost is linear in the dimension. The surrogate's *value* is ~ tr(I) = const
and does not track log|det|; `log_prob` gives the exact encoder density.

Conditioning (FFF Sec. 5.3, the SBI benchmark): the standardized observation y
is concatenated to the input of both networks; the paper does this for the
Lueckmann SBI tasks. M-FFF has no conditional version, so the conditional
manifold model below is our combination of the two papers (docs/fff.md).

Manifold (M-FFF Sec. 3, `angle_dims` non-empty). Each listed coordinate of x is
periodic and embedded as a unit vector (cos phi, sin phi); the others stay
Euclidean (standardized). So the data space is the product manifold
T^k x R^(d-k), embedded in ambient R^(d+k), with ambient layout
[euclidean dims..., cos phi_1, sin phi_1, cos phi_2, sin phi_2, ...].
  * Both networks run in ambient space and are projected onto the manifold:
    f(u) = pi(u + net(u, y)), pi normalizing every (cos, sin) pair.
  * The Hutchinson vector is drawn in the tangent space at z and rescaled to
    norm sqrt(dim M) (M-FFF Sec. 3.3; the authors' `sample_v` with one sample).
  * The latent distribution is N(0, I) on R^(d-k) times, on T^k, either the
    uniform distribution (`latent='uniform'`, as M-FFF uses on compact
    manifolds) or a wrapped normal per circle with sigma = the training std of
    that angle (`latent='wrapped'`). The wrapped latent makes the torus model at
    initialization (f = g = id) the same distribution as the Euclidean FFF's
    N(0, I) in standardized units, so a comparison isolates the geometry from
    the choice of latent (docs/fff.md).
  * The embedding phi -> (cos phi, sin phi) is an isometry of the unit circle,
    so the metric-correction term of M-FFF Eq. 8 vanishes: the log-det is that
    of R^T f'(u) Q, Q and R orthonormal tangent bases at u and f(u).
Angles come back via atan2 in (-pi, pi]. Re-simulation is periodic in the
angles anyway; the kinematics ground truth never leaves (-2.8, 2.6), so this
branch choice does not affect Err_post.

Both networks start at the identity (zero-initialized output layer, global
skip connection), as FFF recommends. `sample` takes and returns raw units.
"""

import math

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn

ACTIVATIONS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}

# Periodic coordinates of x per benchmark for the torus model.
ANGLE_DIMS = {
    'kinematics': (1, 2, 3),  # the three joint angles; the rail offset x0 stays Euclidean: T^3 x R
    'ballistics': (2,),       # launch angle (bounded to 9-81 deg by the prior): T^1 x R^3
}


class Linear(nn.Linear):
    """nn.Linear as addmm: PyTorch has no forward-mode AD rule for `linear` on
    MPS, but has one for addmm. Same parameters and result."""

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight.t())


class ResidualBlock(nn.Module):
    """h + W2 act(W1 act(h))."""

    def __init__(self, hidden, activation):
        super().__init__()
        self.act = ACTIVATIONS[activation]()
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, hidden)

    def forward(self, h):
        return h + self.lin2(self.act(self.lin1(self.act(h))))


class ResidualNet(nn.Module):
    """u -> u + net([u, c]): residual MLP with a global skip connection. The
    output layer is zero, so the network starts as the identity."""

    def __init__(self, dims, context_dims, hidden, n_blocks, activation):
        super().__init__()
        self.inp = Linear(dims + context_dims, hidden)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden, activation) for _ in range(n_blocks)))
        self.act = ACTIVATIONS[activation]()
        self.out = Linear(hidden, dims)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, u, c):
        return u + self.out(self.act(self.blocks(self.inp(torch.cat([u, c], 1)))))


def n_params(ambient_dims, y_dims, hidden, n_blocks):
    """Encoder + decoder parameter count."""
    net = ((ambient_dims + y_dims) * hidden + hidden) + n_blocks * 2 * (hidden * hidden + hidden) \
        + (hidden * ambient_dims + ambient_dims)
    return 2 * net


def width_for_budget(ambient_dims, y_dims, n_blocks, target):
    """Largest hidden width whose encoder + decoder stay at or under `target` parameters."""
    h = 1
    while n_params(ambient_dims, y_dims, h + 1, n_blocks) <= target:
        h += 1
    return h


class ConditionalFFF(nn.Module):

    def __init__(self, x_dims, y_dims, hidden, n_blocks=6, activation='silu', beta=100.0, angle_dims=(),
                 beta_z=0.0, beta_p=0.0, latent='uniform'):
        """angle_dims: periodic coordinates of x (empty = Euclidean FFF).
        beta: x-reconstruction weight. beta_z: latent reconstruction ||z - f(g(z))||^2
        on z ~ p(z); beta_p: squared distance of the decoder's unprojected output
        from the manifold. beta_z / beta_p are the extra M-FFF regularizers, off by default.
        latent: 'uniform' or 'wrapped' (wrapped normal) on the circles."""
        assert latent in ('uniform', 'wrapped')
        super().__init__()
        self.x_dims, self.y_dims = x_dims, y_dims
        self.beta, self.beta_z, self.beta_p, self.latent = beta, beta_z, beta_p, latent
        self.angle_dims = tuple(angle_dims)
        self.euclid_dims = tuple(d for d in range(x_dims) if d not in self.angle_dims)
        self.n_euclid, self.n_angles = len(self.euclid_dims), len(self.angle_dims)
        self.ambient_dims = self.n_euclid + 2 * self.n_angles
        self.encoder = ResidualNet(self.ambient_dims, y_dims, hidden, n_blocks, activation)
        self.decoder = ResidualNet(self.ambient_dims, y_dims, hidden, n_blocks, activation)

        # Statistics of the Euclidean coordinates only; angles are not standardized.
        self.register_buffer('x_mean', torch.zeros(self.n_euclid))
        self.register_buffer('x_std', torch.ones(self.n_euclid))
        self.register_buffer('y_mean', torch.zeros(y_dims))
        self.register_buffer('y_std', torch.ones(y_dims))
        self.register_buffer('angle_std', torch.ones(self.n_angles))  # wrapped-normal latent scale

    def set_normalization(self, x, y):
        if self.n_euclid:
            e = list(self.euclid_dims)
            self.x_mean.copy_(x[:, e].mean(0)); self.x_std.copy_(x[:, e].std(0))
        if self.n_angles:
            self.angle_std.copy_(x[:, list(self.angle_dims)].std(0))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0))

    # --- the manifold -------------------------------------------------------

    def embed(self, x):
        """Raw x (B, x_dims) -> ambient u (B, ambient_dims)."""
        u = (x[:, list(self.euclid_dims)] - self.x_mean) / self.x_std
        if not self.n_angles:
            return u
        phi = x[:, list(self.angle_dims)]
        return torch.cat([u, torch.stack([torch.cos(phi), torch.sin(phi)], 2).flatten(1)], 1)

    def unembed(self, u):
        """Ambient u -> raw x; angles via atan2 in (-pi, pi]."""
        x = u.new_empty(len(u), self.x_dims)
        x[:, list(self.euclid_dims)] = u[:, :self.n_euclid] * self.x_std + self.x_mean
        if self.n_angles:
            pairs = u[:, self.n_euclid:].view(-1, self.n_angles, 2)
            x[:, list(self.angle_dims)] = torch.atan2(pairs[..., 1], pairs[..., 0])
        return x

    def project(self, u):
        """pi: normalize every (cos, sin) pair; Euclidean coordinates pass through."""
        if not self.n_angles:
            return u
        pairs = u[:, self.n_euclid:].view(-1, self.n_angles, 2)
        pairs = pairs / pairs.norm(dim=2, keepdim=True).clamp_min(1e-6)
        return torch.cat([u[:, :self.n_euclid], pairs.flatten(1)], 1)

    def to_tangent(self, u, v):
        """Project ambient vectors v onto the tangent space at u (u on the manifold)."""
        if not self.n_angles:
            return v
        p = u[:, self.n_euclid:].view(-1, self.n_angles, 2)
        w = v[:, self.n_euclid:].view(-1, self.n_angles, 2)
        w = w - (w * p).sum(2, keepdim=True) * p
        return torch.cat([v[:, :self.n_euclid], w.flatten(1)], 1)

    def tangent_basis(self, u):
        """Orthonormal tangent basis at u: (B, ambient_dims, x_dims). Euclidean unit
        vectors, then (-sin phi, cos phi) in each angle's pair of slots."""
        B = len(u)
        Q = u.new_zeros(B, self.ambient_dims, self.x_dims)
        Q[:, range(self.n_euclid), range(self.n_euclid)] = 1.0
        for k in range(self.n_angles):
            i, col = self.n_euclid + 2 * k, self.n_euclid + k
            Q[:, i, col] = -u[:, i + 1]
            Q[:, i + 1, col] = u[:, i]
        return Q

    def sample_latent(self, n, device, generator=None):
        z = torch.randn(n, self.n_euclid, device=device, generator=generator)
        if not self.n_angles:
            return z
        if self.latent == 'uniform':
            theta = (torch.rand(n, self.n_angles, device=device, generator=generator) * 2 - 1) * math.pi
        else:  # wrapping happens in cos / sin
            theta = torch.randn(n, self.n_angles, device=device, generator=generator) * self.angle_std
        return torch.cat([z, torch.stack([torch.cos(theta), torch.sin(theta)], 2).flatten(1)], 1)

    def latent_log_prob(self, z):
        """N(0, I) on the Euclidean part; on each circle uniform (density 1 / 2 pi)
        or the wrapped normal, summed over 7 windings (exact for sigma <~ 1.5)."""
        ze = z[:, :self.n_euclid]
        lp = -0.5 * (ze ** 2).sum(1) - 0.5 * self.n_euclid * math.log(2 * math.pi)
        if not self.n_angles:
            return lp
        if self.latent == 'uniform':
            return lp - self.n_angles * math.log(2 * math.pi)
        pairs = z[:, self.n_euclid:].view(-1, self.n_angles, 2)
        phi = torch.atan2(pairs[..., 1], pairs[..., 0])
        k = torch.arange(-3, 4, device=z.device, dtype=z.dtype) * 2 * math.pi
        a = -0.5 * ((phi[..., None] + k) / self.angle_std[:, None]) ** 2
        lw = torch.logsumexp(a, -1) - torch.log(self.angle_std) - 0.5 * math.log(2 * math.pi)
        return lp + lw.sum(1)

    # --- the flow -----------------------------------------------------------

    def context(self, y):
        return (y - self.y_mean) / self.y_std

    def encode(self, u, c):
        return self.project(self.encoder(u, c))

    def decode(self, z, c):
        return self.project(self.decoder(z, c))

    def loss_terms(self, x, y, v=None, z_prior=None):
        """Per-batch means of the FFF terms on raw (x, y). Returns a dict with
        'loss' (the training objective), 'nll_surrogate', 'recon' and, when
        enabled, 'recon_z' / 'proj'. `v` fixes the Hutchinson draw (ambient,
        e.g. for a comparable validation loss), `z_prior` the latent draw of the
        latent-reconstruction term. Needs autograd even at evaluation time."""
        c = self.context(y)
        u = self.embed(x).requires_grad_(True)
        z = self.encode(u, c)
        v = torch.randn_like(z) if v is None else v
        v = self.to_tangent(z.detach(), v)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(self.x_dims)
        # g'(z) v, forward mode; x1 = g(z) carries the graph for the reconstruction.
        with fwAD.dual_level():
            x1, v1 = fwAD.unpack_dual(self.decode(fwAD.make_dual(z, v), c))
        # v^T f'(u), backward mode, kept in the graph (it carries the log-det gradient).
        v2, = torch.autograd.grad(z, u, v, create_graph=True)
        surrogate = (v2 * v1.detach()).sum(1)
        nll = -self.latent_log_prob(z) - surrogate
        recon = ((u - x1) ** 2).sum(1)
        terms = {'nll_surrogate': nll.mean(), 'recon': recon.mean()}
        loss = nll + self.beta * recon
        if self.beta_z or self.beta_p:
            zp = self.sample_latent(len(x), x.device) if z_prior is None else z_prior
            raw = self.decoder(zp, c)
            if self.beta_z:
                rz = ((zp - self.encode(self.project(raw), c)) ** 2).sum(1)
                terms['recon_z'] = rz.mean()
                loss = loss + self.beta_z * rz
            if self.beta_p:
                rp = ((raw - self.project(raw)) ** 2).sum(1)
                terms['proj'] = rp.mean()
                loss = loss + self.beta_p * rp
        terms['loss'] = loss.mean()
        return terms

    def loss(self, x, y):
        return self.loss_terms(x, y)['loss']

    def log_prob(self, x, y):
        """Exact log p(x | y) in raw x units through the encoder's change of
        variables, log p_Z(f(u)) + log|det R^T f'(u) Q| - sum log x_std (the
        angle embedding is an isometry, so it adds no term). O(D^3), fine at D = 4.
        For a well-trained model the decoder density agrees (g = f^-1)."""
        from torch.func import jacrev, vmap
        c = self.context(y)
        u = self.embed(x)
        z = self.encode(u, c)
        J = vmap(jacrev(lambda ui, ci: self.encode(ui[None], ci[None])[0]))(u, c)
        Q, R = self.tangent_basis(u), self.tangent_basis(z)
        logdet = torch.linalg.slogdet(R.transpose(1, 2) @ J @ Q)[1]
        return self.latent_log_prob(z) + logdet - torch.log(self.x_std).sum()

    @torch.no_grad()
    def sample(self, y_star, n, chunk=500_000, generator=None):
        """Posterior samples for each y*: z ~ p(z), x = g(z, y*), one decoder pass.

        y_star: (M, dim_y) raw units, numpy or tensor. Returns (M, n, dim_x) raw
        units on the model's device: the `sample(y_star, n)` interface of
        metrics.evaluate."""
        dev = self.y_mean.device
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float32) if not torch.is_tensor(y_star) else y_star,
                            dtype=torch.float32, device=dev)
        M = y.shape[0]
        c = self.context(y).repeat_interleave(n, dim=0)
        out = torch.empty(M * n, self.x_dims, device=dev)
        for s in range(0, M * n, chunk):
            c_s = c[s:s + chunk]
            out[s:s + chunk] = self.unembed(self.decode(self.sample_latent(len(c_s), dev, generator), c_s))
        return out.view(M, n, self.x_dims)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
