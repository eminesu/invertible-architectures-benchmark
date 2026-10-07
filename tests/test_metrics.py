import numpy as np
import pytest

from metrics.benchmarks import (forward_fn, load_test_conditions, make_test_conditions, prior)
from metrics.evaluate import err_post, err_resim
from metrics.ground_truth import rejection_sample
from metrics.mmd import imq_kernel, mmd, sq_dists


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def test_sq_dists_matches_direct(rng):
    a, b = rng.randn(20, 4), rng.randn(30, 4)
    direct = ((a[:, None] - b[None]) ** 2).sum(-1)
    assert np.allclose(sq_dists(a, b), direct)


def test_mmd_same_distribution_near_zero(rng):
    a, b = rng.randn(1000, 4), rng.randn(1000, 4)
    assert abs(mmd(a, b)) < 2e-3
    assert abs(mmd(a, a, unbiased=False)) < 1e-12


def test_mmd_grows_with_shift(rng):
    a = rng.randn(500, 4) * 0.3
    b = rng.randn(500, 4) * 0.3
    values = [mmd(a, b + s) for s in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0)]
    assert np.all(np.diff(values) > 0), values


def test_mmd_unbiased_removes_diagonal_bias(rng):
    # The biased V-statistic is inflated by roughly k(0) (1/n + 1/m).
    n = 200
    a, b = rng.randn(n, 2), rng.randn(n, 2)
    k0 = imq_kernel(np.zeros(1))[0]
    gap = mmd(a, b, unbiased=False) - mmd(a, b)
    assert np.isclose(gap, 2 * k0 / n, rtol=0.5)


def test_test_conditions_reproducible():
    for b in ('kinematics', 'ballistics'):
        y_star, x_true = load_test_conditions(b)
        y2, x2 = make_test_conditions(b)
        assert np.array_equal(y_star, y2) and np.array_equal(x_true, x2)
        assert np.allclose(forward_fn(b)(x_true), y_star)
        assert y_star.shape == (1000, {'kinematics': 2, 'ballistics': 1}[b])


def test_err_resim_zero_for_exact_preimages():
    y_star, x_true = load_test_conditions('kinematics')
    x = np.repeat(x_true[:50, None], 7, axis=1)
    per_y, mean, failed = err_resim(x, y_star[:50], forward_fn('kinematics'))
    assert per_y.shape == (50,) and mean < 1e-20 and not failed.any()


def test_err_resim_counts_ballistics_failures():
    y_star, x_true = load_test_conditions('ballistics')
    x = np.repeat(x_true[:3, None], 4, axis=1).copy()
    x[0, :2, 3] = -50.0   # thrown into the ground: no single impact in the window
    x[1, :, 3] = -50.0
    per_y, _, failed = err_resim(x, y_star[:3], forward_fn('ballistics'))
    assert np.allclose(failed, [0.5, 1.0, 0.0])
    assert per_y[0] < 1e-20 and np.isnan(per_y[1]) and per_y[2] < 1e-20


def test_rejection_samples_hit_target():
    # Small run: accepted x must re-simulate to within a few sigma of y*.
    y_star, _ = load_test_conditions('kinematics')
    sigma = 0.02
    samples, _ = rejection_sample('kinematics', y_star[:5], n_gt=200, sigma=sigma,
                                  chunk=500_000, max_proposals=50_000_000, verbose=False)
    f = forward_fn('kinematics')
    for y, s in zip(y_star[:5], samples):
        assert len(s) == 200
        d2 = ((f(s) - y) ** 2).sum(1)
        assert d2.max() < (4 * sigma) ** 2
        assert np.isclose(d2.mean(), 2 * sigma ** 2, rtol=0.3)  # Gaussian likelihood, 2-D


def test_prior_scores_worse_than_posterior():
    """The prior ignores y*; true posterior samples must beat it on both metrics."""
    y_star, _ = load_test_conditions('kinematics')
    y_star = y_star[:5]
    gt, _ = rejection_sample('kinematics', y_star, n_gt=400, sigma=0.02, seed=1,
                             chunk=500_000, max_proposals=50_000_000, verbose=False)
    ref, _ = rejection_sample('kinematics', y_star, n_gt=400, sigma=0.02, seed=2,
                              chunk=500_000, max_proposals=50_000_000, verbose=False)
    model = np.stack(ref)
    bad = prior('kinematics', 5 * 400, np.random.RandomState(3)).reshape(5, 400, 4)
    f = forward_fn('kinematics')
    post_good, _ = err_post(model, gt)
    post_bad, _ = err_post(bad, gt)
    assert np.all(post_bad > 10 * np.abs(post_good))
    assert err_resim(bad, y_star, f)[1] > 100 * err_resim(model, y_star, f)[1]
