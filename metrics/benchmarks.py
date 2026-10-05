"""Prior, forward process and the fixed 1000 test conditions y* for each benchmark.

Kruse et al. (2021) Sec. 4: every model is evaluated on 1000 unseen conditions
y* = f(x_true), x_true ~ prior. The set is generated once from TEST_SEED and
stored in metrics/conditions/<benchmark>.npz (committed), so every model is
scored on exactly the same y*. TEST_SEED does not overlap the seeds used for
training/validation data (10_000 + s, 20_000 + s, 30_000 + s).
"""

from pathlib import Path

import numpy as np

from data.toy_data import MODELS

BENCHMARKS = tuple(MODELS)  # ('kinematics', 'ballistics')
N_CONDITIONS = 1000
TEST_SEED = 20210126  # arXiv:2101.10763 submission date
CONDITIONS_DIR = Path(__file__).parent / 'conditions'


def get_model(benchmark):
    return MODELS[benchmark]()


def prior(benchmark, n, rng):
    """n samples x ~ p(x), shape (n, 4)."""
    return get_model(benchmark).sample_prior(n, rng)


def forward_fn(benchmark):
    """The true forward process f: (n, 4) -> (n, dim_y).

    For ballistics, rows whose trajectory does not have exactly one ground
    impact in the simulated window come back as NaN instead of raising, since
    model samples can land anywhere (see data/toy_data.py). The chunk size only
    changes memory use and speed, not the result."""
    model = get_model(benchmark)
    kw = {'chunk': 5_000} if benchmark == 'ballistics' else {}  # ~2x faster than the default
    return lambda x: model.forward_process(np.asarray(x, dtype=np.float64), strict=False, **kw)


def make_test_conditions(benchmark, n=N_CONDITIONS, seed=TEST_SEED):
    """Draw x_true ~ prior and y* = f(x_true). Returns (y_star (n, dim_y), x_true (n, 4)).

    Prior draws without a valid ballistics impact are redrawn (they have no y*
    at all); this happens for a small fraction of draws and keeps the set at n."""
    rng = np.random.RandomState(seed)
    f = forward_fn(benchmark)
    xs, ys, have = [], [], 0
    while have < n:
        x = prior(benchmark, n, rng)
        y = f(x)
        ok = np.isfinite(y).all(1)
        xs.append(x[ok]); ys.append(y[ok]); have += ok.sum()
    return np.concatenate(ys)[:n], np.concatenate(xs)[:n]


def conditions_path(benchmark):
    return CONDITIONS_DIR / f'{benchmark}.npz'


def save_test_conditions(benchmark):
    y_star, x_true = make_test_conditions(benchmark)
    CONDITIONS_DIR.mkdir(exist_ok=True)
    np.savez(conditions_path(benchmark), y_star=y_star, x_true=x_true, seed=TEST_SEED)
    return y_star, x_true


def load_test_conditions(benchmark):
    """The committed, shared set of test conditions: (y_star, x_true)."""
    with np.load(conditions_path(benchmark)) as f:
        return f['y_star'], f['x_true']


if __name__ == '__main__':
    for b in BENCHMARKS:
        y_star, _ = save_test_conditions(b)
        print(f'{b}: saved {y_star.shape} to {conditions_path(b)}')
