"""Flow Matching Posterior Estimation (FMPE), hand-rolled in plain PyTorch.

Dax, Wildberger, Buchholz, Green, Macke, Schölkopf (2023), "Flow Matching for
Scalable Simulation-Based Inference", NeurIPS 2023, arXiv:2305.17161, Sec. 3,
which applies the conditional flow matching objective of Lipman et al. (2023),
arXiv:2210.02747, to posteriors p(x | y).

A plain network v(t, x_t, y) (no invertibility constraint) is a velocity field.
Integrating dx/dt = v(t, x, y*) from x(0) ~ N(0, I) to t = 1 gives x(1) ~ p(x | y*).

Training (FMPE Eqs. 5-6, optimal-transport path): for (x1, y) from the
simulator, x0 ~ N(0, I) and t ~ p_alpha(t),

    x_t = (1 - (1 - sigma_min) t) x0 + t x1
    u   = x1 - (1 - sigma_min) x0             (Eq. 6 evaluated at x_t)
    L   = || v(t, x_t, y) - u ||^2

Time prior (FMPE Sec. 3.3): t = u^(1 / (1 + alpha)), u ~ U(0, 1), as in the
authors' code (dingo, `cflow_base.py::sample_t`). Its density is p(t) ∝ t^alpha:
uniform for alpha = 0, more weight near the data end t = 1 for alpha > 0. The
paper prints "p_alpha(t) ∝ t^(1/(1+alpha))", which is the sampling exponent, not
the density (that density is not uniform at alpha = 0, as the paper says it is).
BayesFlow's FlowMatching samples t = u^(1 + alpha) instead, i.e. p(t) ∝
t^(-alpha/(1+alpha)), which favours the *noise* end for alpha > 0 (docs/fmpe.md).

Network (FMPE Sec. 3.2 / Sec. 4): residual MLP on the concatenation (t, x_t, y),
with every residual block gated by a gated linear unit (GLU) on (t, x_t). The
paper reports GLU conditioning on (t, theta) beats plain concatenation.

x and y are standardized with training-set statistics, like models/mdn.py; the
flow lives in standardized x-space. `sample` takes and returns raw units.
"""

import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}

# Butcher tableaux (a, b, c) of the explicit fixed-step solvers; NFE = stages * steps.
SOLVERS = {
    'euler': ([], [1.0], [0.0]),
    'midpoint': ([[0.5]], [0.0, 1.0], [0.0, 0.5]),
    'rk4': ([[0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6], [0.0, 0.5, 0.5, 1.0]),
}


def nfe(method, steps):
    return len(SOLVERS[method][1]) * steps


class GatedResidualBlock(nn.Module):
    """h + GLU(context) * W2 act(W1 act(h)): two residual layers whose output is
    gated element-wise by sigmoid(W_g context)."""

    def __init__(self, hidden, context_dims, activation):
        super().__init__()
        self.act = ACTIVATIONS[activation]()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(context_dims, hidden)

    def forward(self, h, context):
        r = self.lin2(self.act(self.lin1(self.act(h))))
        return h + torch.sigmoid(self.gate(context)) * r


def n_params(x_dims, y_dims, hidden, n_blocks):
    d_in, ctx = 1 + x_dims + y_dims, 1 + x_dims
    block = 2 * (hidden * hidden + hidden) + (ctx * hidden + hidden)
    return (d_in * hidden + hidden) + n_blocks * block + (hidden * x_dims + x_dims)


def width_for_budget(x_dims, y_dims, n_blocks, target):
    """Largest hidden width whose network stays at or under `target` parameters."""
    h = 1
    while n_params(x_dims, y_dims, h + 1, n_blocks) <= target:
        h += 1
    return h


class FMPE(nn.Module):

    def __init__(self, x_dims, y_dims, hidden, n_blocks=6, activation='silu', sigma_min=1e-4, alpha=0.0):
        super().__init__()
        assert alpha > -1
        self.x_dims, self.y_dims = x_dims, y_dims
        self.sigma_min, self.alpha = sigma_min, alpha
        self.inp = nn.Linear(1 + x_dims + y_dims, hidden)
        self.blocks = nn.ModuleList(GatedResidualBlock(hidden, 1 + x_dims, activation) for _ in range(n_blocks))
        self.act = ACTIVATIONS[activation]()
        self.out = nn.Linear(hidden, x_dims)
        # Zero output layer: the initial field is v = 0, a neutral start.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.register_buffer('x_mean', torch.zeros(x_dims))
        self.register_buffer('x_std', torch.ones(x_dims))
        self.register_buffer('y_mean', torch.zeros(y_dims))
        self.register_buffer('y_std', torch.ones(y_dims))

    def set_normalization(self, x, y):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0))

    def velocity(self, t, x_t, y_n):
        """v(t, x_t, y) in standardized units. t: (B, 1), x_t: (B, x_dims), y_n: (B, y_dims)."""
        context = torch.cat([t, x_t], 1)
        h = self.inp(torch.cat([context, y_n], 1))
        for block in self.blocks:
            h = block(h, context)
        return self.out(self.act(h))

    def sample_time(self, n, device, generator=None):
        u = torch.rand(n, 1, device=device, generator=generator)
        return u ** (1 / (1 + self.alpha))

    def loss(self, x, y, noise=None):
        """Conditional flow matching loss on a raw-unit batch (x, y): mean over the
        batch of the squared error summed over x dimensions. `noise` = (x0, t)
        fixes the random draws, e.g. for a comparable validation loss."""
        x1 = (x - self.x_mean) / self.x_std
        y_n = (y - self.y_mean) / self.y_std
        if noise is None:
            x0, t = torch.randn_like(x1), self.sample_time(len(x1), x1.device)
        else:
            x0, t = noise
        x_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        u = x1 - (1 - self.sigma_min) * x0
        return ((self.velocity(t, x_t, y_n) - u) ** 2).sum(1).mean()

    @torch.no_grad()
    def integrate(self, x0, y_n, steps, method='rk4'):
        """Fixed-step explicit Runge-Kutta from t = 0 to t = 1."""
        a, b, c = SOLVERS[method]
        dt = 1.0 / steps
        x = x0
        t_col = x0.new_empty(len(x0), 1)
        for i in range(steps):
            t0 = i * dt
            k = []
            for s in range(len(b)):
                xs = x
                for j, a_sj in enumerate(a[s - 1] if s else []):
                    if a_sj:
                        xs = xs + dt * a_sj * k[j]
                k.append(self.velocity(t_col.fill_(t0 + c[s] * dt), xs, y_n))
            x = x + dt * sum(b_s * k_s for b_s, k_s in zip(b, k) if b_s)
        return x

    @torch.no_grad()
    def sample(self, y_star, n, steps=32, method='rk4', chunk=200_000, generator=None):
        """Posterior samples for each y*: x(0) ~ N(0, I)^n, integrate the ODE to t = 1.

        y_star: (M, dim_y) raw units, numpy or tensor. Returns (M, n, dim_x) raw
        units, a tensor on the model's device. This is the `sample(y_star, n)`
        interface metrics.evaluate expects (bind steps / method with a lambda).
        Uses nfe(method, steps) network evaluations per sample. `chunk` bounds the
        number of ODE states integrated at once (memory only)."""
        dev = self.x_mean.device
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float32) if not torch.is_tensor(y_star) else y_star,
                            dtype=torch.float32, device=dev)
        M = y.shape[0]
        y_n = ((y - self.y_mean) / self.y_std).repeat_interleave(n, dim=0)
        out = torch.empty(M * n, self.x_dims, device=dev)
        for s in range(0, M * n, chunk):
            y_c = y_n[s:s + chunk]
            x0 = torch.randn(len(y_c), self.x_dims, device=dev, generator=generator)
            out[s:s + chunk] = self.integrate(x0, y_c, steps, method)
        return (out * self.x_std + self.x_mean).view(M, n, self.x_dims)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
