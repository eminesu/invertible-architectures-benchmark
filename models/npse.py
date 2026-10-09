"""Neural posterior score estimation (NPSE): score-based diffusion for p(x | y),
built on BayesFlow's DiffusionModel (bayesflow 2.0.14, `bf.networks.DiffusionModel`).

Geffner, Papamakarios, Mnih (2023), "Compositional Score Modeling for
Simulation-Based Inference", arXiv:2209.14249, in the SDE framework of Song et
al. (2021), arXiv:2011.13456. See docs/npse.md for the method and the choices.

All BayesFlow defaults are kept except the width of the default subnet (TimeMLP,
5 residual blocks), which `width_for_budget` solves to a parameter budget:
variance-preserving cosine schedule on log-SNR in [-12, 12], velocity prediction
trained with the noise loss and sigmoid weighting. x and y are standardized by
the approximator (`standardize='all'`).

Sampling runs the reverse process from noise to t = 0: the probability-flow ODE
(`euler`, `rk45`, `tsit5`) or the reverse SDE (`euler_maruyama`,
`two_step_adaptive`, ...). `NPSE.sample(y_star, n)` is the interface
metrics.evaluate expects.

Keras picks its torch device at import time; set KERAS_TORCH_DEVICE (and
KERAS_BACKEND=torch) before importing this module.
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

N_BLOCKS = 5  # TimeMLP default depth


def build(x_dims, y_dims, width):
    """A BayesFlow ContinuousApproximator around a DiffusionModel, built for the given dims."""
    network = bf.networks.DiffusionModel(subnet_kwargs={'widths': (width,) * N_BLOCKS})
    approximator = bf.ContinuousApproximator(inference_network=network, standardize='all')
    approximator.build_from_data({'inference_variables': np.zeros((2, x_dims), 'float32'),
                                  'inference_conditions': np.zeros((2, y_dims), 'float32')})
    return approximator


def count_parameters(approximator):
    return sum(int(np.prod(v.shape)) for v in approximator.trainable_weights)


def width_for_budget(x_dims, y_dims, target):
    """Largest width whose network stays at or under `target` parameters.
    Returns (width, parameter count)."""
    def count(w):
        return count_parameters(build(x_dims, y_dims, w))
    lo, hi = 8, 2048
    while lo < hi:  # largest width with count <= target
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if count(mid) <= target else (lo, mid - 1)
    return lo, count(lo)


def count_nfe(approximator, y, method, steps):
    """Network evaluations per sample, counted by wrapping DiffusionModel.velocity,
    which both the ODE and the SDE samplers call once per evaluation (adaptive
    steps: averaged over the given conditions)."""
    net = approximator.inference_network
    calls, velocity = [0], net.velocity
    def counted(*a, **kw):
        calls[0] += 1
        return velocity(*a, **kw)
    net.velocity = counted
    try:
        approximator.sample(num_samples=1, conditions={'inference_conditions': y}, method=method, steps=steps)
    finally:
        net.velocity = velocity
    return calls[0]


class NPSE:
    """A trained (or freshly built) diffusion posterior with a fixed sampler.

    `sample(y_star, n)` is the shared interface: y_star (M, dim_y) raw units ->
    x (M, n, dim_x) raw units, numpy."""

    def __init__(self, approximator, method='euler', steps=32, chunk=100):
        self.approximator = approximator
        self.method, self.steps, self.chunk = method, steps, chunk

    @classmethod
    def load(cls, path, **kwargs):
        """Load an approximator saved by experiments/bayesflow_npse.py (approximator.keras)."""
        return cls(keras.saving.load_model(path), **kwargs)

    def sample(self, y_star, n):
        y = np.asarray(y_star, dtype='float32')
        # Chunk the conditions: BayesFlow integrates all M * n states at once.
        xs = [self.approximator.sample(num_samples=n, conditions={'inference_conditions': y[s:s + self.chunk]},
                                       method=self.method, steps=self.steps)['inference_variables']
              for s in range(0, len(y), self.chunk)]
        return np.concatenate(xs)

    def nfe(self, y):
        return count_nfe(self.approximator, np.asarray(y, dtype='float32'), self.method, self.steps)

    @property
    def n_params(self):
        return count_parameters(self.approximator)
