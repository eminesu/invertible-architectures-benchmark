"""Priors and forward processes for the two benchmark problems.

Vendored from https://github.com/vislearn/inn_toy_data (MIT License,
Copyright (c) 2020 Visual Learning Lab Heidelberg), commit 357bc08.
Only `sample_prior` and `forward_process` are kept, with the numerics unchanged.

Why vendored instead of imported: upstream imports `sklearn.neighbors.kde`
(removed in scikit-learn 0.24) and sets `text.usetex=True` at import time, so
the files do not import in a current environment. The plotting/MAP helpers are
not needed for training or evaluation.

One addition: `InverseBallisticsModel.forward_process` checks that every
trajectory produces exactly one ground impact inside the simulated time window
and raises (or returns NaN with strict=False) otherwise. Upstream silently
returns fewer rows than inputs in that case, which would misalign x and y.
"""

import numpy as np


class InverseKinematicsModel:
    """Planar arm: x = (vertical start offset, three joint angles), y = 2-D end point."""

    n_parameters = 4
    n_observations = 2
    name = 'inverse-kinematics'

    def __init__(self, lens=(0.5, 0.5, 1.0), sigmas=(0.25, 0.5, 0.5, 0.5)):
        self.lens = np.array(lens)
        self.sigmas = np.array(sigmas)

    def sample_prior(self, n, rng=np.random):
        return rng.randn(n, 4) * self.sigmas

    @staticmethod
    def _segment(p, length, angle):
        p = np.array(p)
        p[:, 0] += length * np.cos(angle)
        p[:, 1] += length * np.sin(angle)
        return p

    def joint_positions(self, x):
        """All four joint positions, each of shape (n, 2). Used for plotting."""
        x0 = np.stack([np.zeros(x.shape[0]), x[:, 0]], axis=1)
        x1 = self._segment(x0, self.lens[0], x[:, 1])
        x2 = self._segment(x1, self.lens[1], x[:, 1] + x[:, 2])
        x3 = self._segment(x2, self.lens[2], x[:, 1] + x[:, 2] + x[:, 3])
        return x0, x1, x2, x3

    def forward_process(self, x, strict=True):
        return self.joint_positions(x)[-1]


class InverseBallisticsModel:
    """Projectile with drag: x = (x0, y0, launch angle, v0), y = 1-D impact position."""

    n_parameters = 4
    n_observations = 1
    name = 'inverse-ballistics'

    def __init__(self, g=9.81, k=0.25, m=0.2):
        self.g = g  # gravity
        self.k = k  # drag coefficient
        self.m = m  # object mass
        self.xy_mu = np.array((0, 1.5))
        self.xy_std = np.array((0.5, 0.5))

    def sample_prior(self, n, rng=np.random):
        x = rng.randn(n, 1) * self.xy_std[0] + self.xy_mu[0]
        y = rng.randn(n, 1) * self.xy_std[1] + self.xy_mu[1]
        y = np.maximum(y, 0)
        angle = rng.rand(n, 1) * np.pi / 2 * 0.8 + np.pi / 2 * 0.1
        v0 = rng.poisson(15, (n, 1))
        return np.concatenate([x, y, angle, v0], axis=1)

    def trajectories_from_parameters(self, x):
        x0, y0, angle, v0 = np.split(x, 4, axis=1)
        v0 = np.repeat(v0, 1500, axis=-1)
        angle = np.repeat(angle, 1500, axis=-1)
        t = np.repeat(np.linspace(0, 6, 1500)[None, :], x.shape[0], axis=0)
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)

        expterm = np.exp(-self.k * t / self.m) - 1
        xt = x0 - (vx * self.m / self.k) * expterm
        yt = y0 - (self.m / (self.k * self.k)) * ((self.g * self.m + vy * self.k) * expterm + self.g * t * self.k)
        return xt, yt

    def impact_from_trajectories(self, xs, ys, strict=True):
        ys_peak = np.argmax(ys, axis=1)
        ys_after_peak = np.where(xs < xs[np.arange(xs.shape[0]), ys_peak][:, None], 0.1, ys)
        crossings = np.diff(np.signbit(ys_after_peak), axis=1)
        n_cross = crossings.sum(axis=1)
        if strict and np.any(n_cross != 1):
            raise ValueError('ballistics forward process: '
                             f'{np.sum(n_cross != 1)} of {xs.shape[0]} trajectories lack exactly one impact')
        impact = xs[np.arange(xs.shape[0]), crossings.argmax(axis=1)]
        return np.where(n_cross == 1, impact, np.nan)

    def forward_process(self, x, strict=True, chunk=100_000):
        """strict=True raises if any trajectory lacks exactly one impact (upstream
        would silently drop that row). strict=False returns NaN for those rows,
        which is what re-simulation of model samples needs."""
        # Trajectories are (n, 1500) float64 arrays; chunk to bound memory.
        out = []
        for s in range(0, x.shape[0], chunk):
            xs, ys = self.trajectories_from_parameters(x[s:s + chunk])
            out.append(self.impact_from_trajectories(xs, ys, strict))
        return np.concatenate(out)[:, None]


MODELS = {
    'kinematics': InverseKinematicsModel,
    'ballistics': InverseBallisticsModel,
}


def make_dataset(problem, n, seed):
    """Sample n (x, y) pairs from prior + forward process, as float32 arrays."""
    model = MODELS[problem]()
    rng = np.random.RandomState(seed)
    x = model.sample_prior(n, rng)
    y = model.forward_process(x)
    return x.astype(np.float32), y.astype(np.float32)
