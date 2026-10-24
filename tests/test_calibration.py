import numpy as np
import pytest

from metrics.calibration import calibration_report, central_coverage, ranks


def make(c, M=400, n=500, d=4, seed=0):
    """True posterior N(mu, I). The model reports N(mu, c^2 I): c<1 over-confident, c>1 under-confident."""
    rng = np.random.RandomState(seed)
    mu = rng.randn(M, d)
    x_true = mu + rng.randn(M, d)
    samples = mu[:, None] + c * rng.randn(M, n, d)
    return samples, x_true


@pytest.mark.parametrize('c, expected', [(1.0, 'calibrated'), (0.5, 'over-confident'), (2.0, 'under-confident')])
def test_verdicts(c, expected):
    s, x = make(c)
    rep = calibration_report(s, x, scale=np.ones(4))
    assert rep['marginal']['verdict'] == expected
    assert rep['joint']['verdict'] == expected


def test_ranks_uniform_when_calibrated():
    s, x = make(1.0)
    r = ranks(s, x)
    assert abs(r.mean() - 0.5) < 0.02
    assert abs(((r < 0.05) | (r > 0.95)).mean() - 0.10) < 0.02


def test_ties_are_randomised():
    s = np.zeros((300, 100, 1))
    x = np.zeros((300, 1))
    r = ranks(s, x)
    assert 0.4 < r.mean() < 0.6 and r.std() > 0.2


def test_spread_ratio_against_ground_truth():
    s, x = make(0.5, M=60, n=500)
    rng = np.random.RandomState(1)
    mu = s.mean(1)
    gt = [m + rng.randn(500, 4) for m in mu]
    rep = calibration_report(s, x, gt, scale=np.ones(4))
    assert rep['spread_vs_ground_truth']['verdict'] == 'over-confident'
    assert 0.4 < rep['spread_vs_ground_truth']['median_ratio'] < 0.6


def test_coverage_curve_of_perfect_ranks():
    r = np.random.RandomState(0).uniform(size=20000)
    levels = np.array([0.1, 0.5, 0.9])
    assert np.allclose(central_coverage(r, levels), levels, atol=0.02)