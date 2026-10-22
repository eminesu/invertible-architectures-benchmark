"""Prior-conditioned FMPE: one flow for the posteriors under a family of priors.

FMPE (models/fmpe.py) learns v(t, x, y) and samples p(x | y*) for the one prior
it was trained on. Here the prior's parameter lambda is a second conditioning
input,

    v(t, x, y, lambda)   ->   samples from  p(x | y*, lambda),

trained on (x, y, lambda) with lambda ~ U[LAM_RANGE], x ~ p(x | lambda),
y = f(x). The flow matching objective is unchanged; the network learns the
posterior amortized over y* *and* over priors (an amortized posterior over
priors), so a new lambda needs no retraining. For ballistics lambda is the
Poisson rate of v0 (data/toy_data.py; the benchmark prior is lambda = 15).

Implementation: lambda is appended to y as one more conditioning dimension,
c = [y, lambda], standardized with training-set statistics like y. It enters
the network through the same path as y (the input layer); the gated residual
blocks still gate on (t, x_t), exactly as in plain FMPE. `sample` takes and
returns raw units.
"""

import numpy as np
import torch

from models.fmpe import FMPE


class ConditionalFMPE(FMPE):
    """FMPE whose conditioning vector is [y, lambda]. y_mean / y_std hold the
    statistics of that concatenated vector; the parameter count is that of an
    FMPE with y_dims + 1 (models.fmpe.n_params / width_for_budget)."""

    def __init__(self, x_dims, y_dims, hidden, n_blocks=6, activation='silu', sigma_min=1e-4, alpha=0.0):
        super().__init__(x_dims, y_dims + 1, hidden, n_blocks, activation, sigma_min, alpha)
        self.obs_dims = y_dims

    @staticmethod
    def context(y, lam):
        """[y, lambda]; lam is a scalar or one value per row of y."""
        lam = torch.as_tensor(lam, dtype=y.dtype, device=y.device)
        return torch.cat([y, lam.reshape(-1, 1).expand(len(y), 1)], 1)

    def set_normalization(self, x, y, lam):
        super().set_normalization(x, self.context(y, lam))

    def loss(self, x, y, lam, noise=None):
        return super().loss(x, self.context(y, lam), noise=noise)

    @torch.no_grad()
    def sample(self, y_star, lam, n, **kw):
        """Samples of p(x | y*, lambda) for each y*. y_star: (M, dim_y) raw units;
        lam: a scalar (the same prior for every y*) or M values. Returns (M, n,
        dim_x), as FMPE.sample; keyword arguments (steps, method, ...) pass through.
        `lambda y, n: model.sample(y, lam, n, ...)` is the metrics.evaluate interface."""
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float32) if not torch.is_tensor(y_star) else y_star,
                            dtype=torch.float32, device=self.x_mean.device)
        return super().sample(self.context(y, lam), n, **kw)
