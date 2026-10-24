"""Prior-conditioned FMPE: one flow for the posteriors under a family of priors.

FMPE (models/fmpe.py) learns v(t, x, y) and samples p(x | y*) for the one prior
it was trained on. Here the prior's parameters c are a second conditioning
input,

    v(t, x, y, c)   ->   samples from  p(x | y*, c),

trained on (x, y, c) with c ~ U[range], x ~ p(x | c), y = f(x). The flow
matching objective is unchanged; the network learns the posterior amortized over
y* *and* over priors (an amortized posterior over priors), so a new prior needs
no retraining. c is data.toy_data.PRIOR_PARAMS: for ballistics the Poisson rate
lambda of v0 (benchmark 15), for kinematics the joint-angle sigmas 1..3
(benchmark 0.5 each).

Implementation: c is appended to y, conditioning vector [y, c], standardized
with training-set statistics like y. It enters the network through the same path
as y (the input layer); the gated residual blocks still gate on (t, x_t),
exactly as in plain FMPE. `sample` takes and returns raw units.
"""

import numpy as np
import torch

from models.fmpe import FMPE


class ConditionalFMPE(FMPE):
    """FMPE whose conditioning vector is [y, c] with c the prior_dims prior
    parameters. y_mean / y_std hold the statistics of that concatenated vector;
    the parameter count is that of an FMPE with y_dims + prior_dims
    (models.fmpe.n_params / width_for_budget)."""

    def __init__(self, x_dims, y_dims, hidden, n_blocks=6, activation='silu', sigma_min=1e-4, alpha=0.0,
                 prior_dims=1):
        super().__init__(x_dims, y_dims + prior_dims, hidden, n_blocks, activation, sigma_min, alpha)
        self.obs_dims, self.prior_dims = y_dims, prior_dims

    def context(self, y, c):
        """[y, c]; c is one prior (prior_dims values, the same for every row of y)
        or one per row, (len(y), prior_dims)."""
        c = torch.as_tensor(np.asarray(c, dtype=np.float32) if not torch.is_tensor(c) else c,
                            dtype=y.dtype, device=y.device).reshape(-1, self.prior_dims)
        return torch.cat([y, c.expand(len(y), -1) if len(c) == 1 else c], 1)

    def set_normalization(self, x, y, c):
        super().set_normalization(x, self.context(y, c))

    def loss(self, x, y, c, noise=None):
        return super().loss(x, self.context(y, c), noise=noise)

    @torch.no_grad()
    def sample(self, y_star, c, n, **kw):
        """Samples of p(x | y*, c) for each y*. y_star: (M, dim_y) raw units; c: one
        prior (prior_dims values, used for every y*) or (M, prior_dims). Returns
        (M, n, dim_x), as FMPE.sample; keyword arguments (steps, method, ...) pass
        through. `lambda y, n: model.sample(y, c, n, ...)` is the metrics.evaluate
        interface."""
        y = torch.as_tensor(np.asarray(y_star, dtype=np.float32) if not torch.is_tensor(y_star) else y_star,
                            dtype=torch.float32, device=self.x_mean.device)
        return super().sample(self.context(y, c), n, **kw)
