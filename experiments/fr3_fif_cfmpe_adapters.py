"""Experiment-local FR3 adapters for FIF and prior-conditioned FMPE.

Shared data/model registries, task classes, and ground-truth code are consumed
without modification or runtime monkey-patching. Only these experiment entry
points opt into the FR3 prior family and normalization safeguard.
"""

import numpy as np

from data.toy_data import (
    MODELS,
    PRIOR_PARAMS as TOY_PRIOR_PARAMS,
    make_conditional_dataset as toy_conditional_dataset,
    prior_tag as toy_prior_tag,
    sample_prior_given as toy_sample_prior_given,
)
from models.conditional_fmpe import ConditionalFMPE


FR3_CONSTRAINT_DIMS = {'fr3_6d': 6, 'fr3_5d': 5}
PRIOR_SPECS = {
    **TOY_PRIOR_PARAMS,
    **{name: {'names': tuple(f'width{i}' for i in range(1, 8)),
              'default': (1.0,) * 7, 'range': (0.5, 1.0), 'tag': 'width'}
       for name in FR3_CONSTRAINT_DIMS},
}


def fif_latent_dims(problem):
    """Use intrinsic FR3 constraints, leaving the shared task classes intact."""
    simulator = MODELS[problem]()
    return simulator.n_parameters - FR3_CONSTRAINT_DIMS.get(problem, simulator.n_observations)


class Fr3ConditionalFMPE(ConditionalFMPE):
    """Local normalization guard; the shared ConditionalFMPE stays unchanged.

    Inherits the existing network and loss without adding parameters, so its
    state dict is compatible with the same-dimension base implementation.
    """

    def set_normalization(self, x, y, c):
        super().set_normalization(x, y, c)
        self.y_std.clamp_min_(1e-6)


def conditional_model_class(problem):
    return Fr3ConditionalFMPE if problem in FR3_CONSTRAINT_DIMS else ConditionalFMPE


def prior_tag(problem, params):
    if problem not in FR3_CONSTRAINT_DIMS:
        return toy_prior_tag(problem, params)
    return 'width' + '-'.join(f'{float(v):g}' for v in np.ravel(params))


def sample_prior_given(problem, n, rng, params):
    if problem not in FR3_CONSTRAINT_DIMS:
        return toy_sample_prior_given(problem, n, rng, params)
    widths = np.asarray(params, dtype=np.float64)
    if (widths.shape not in ((7,), (n, 7)) or not np.isfinite(widths).all()
            or np.any((widths <= 0) | (widths > 1))):
        raise ValueError('FR3 prior widths must have shape (7,) or (n, 7) and lie in (0, 1]')
    x = MODELS[problem]().sample_prior(n, rng)
    if np.all(widths == 1):
        return x  # exact same nominal draws and RNG use as the shared prior
    from robot_ik.kinematics import joint_limits
    lo, hi = joint_limits()
    mid = (lo + hi) / 2
    return mid + (x - mid) * widths


def make_conditional_dataset(problem, n, seed, prior_range=None, nominal_fraction=0.0):
    """Local prior-family data, with unchanged default toy-data delegation."""
    if problem not in FR3_CONSTRAINT_DIMS and nominal_fraction == 0:
        return toy_conditional_dataset(problem, n, seed, prior_range)
    spec = PRIOR_SPECS[problem]
    bounds = spec['range'] if prior_range is None else prior_range
    if (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
            or not np.isfinite(nominal_fraction) or not 0 <= nominal_fraction < 1):
        raise ValueError('Need an increasing finite prior range and nominal_fraction in [0, 1)')
    if problem in FR3_CONSTRAINT_DIMS and not 0 < bounds[0] < bounds[1] <= 1:
        raise ValueError('FR3 prior-width training range must be inside (0, 1]')
    rng = np.random.RandomState(seed)
    c = rng.uniform(*bounds, (n, len(spec['names'])))
    if nominal_fraction:
        c[rng.rand(n) < nominal_fraction] = spec['default']
    x = sample_prior_given(problem, n, rng, c)
    y = MODELS[problem]().forward_process(x)
    return x.astype(np.float32), y.astype(np.float32), c.astype(np.float32)


def evaluation_prior_params(problem, params):
    """Adapt nominal FR3 widths to the existing evaluator's fixed-prior API."""
    if problem not in FR3_CONSTRAINT_DIMS:
        return params
    if not np.array_equal(params, PRIOR_SPECS[problem]['default']):
        raise ValueError('FR3 evaluation requires the nominal prior: seven width scales equal to 1')
    return None
