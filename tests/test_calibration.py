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


def test_nonfinite_samples_are_excluded_and_reported():
    s, x = make(1.0, M=200, n=300)
    s[:10, 0, 0] = np.nan
    s[10:15, 3, 1] = np.inf
    rep = calibration_report(s, x, scale=np.ones(4))
    assert rep['n_conditions'] == 185 and rep['n_conditions_total'] == 200
    assert abs(rep['nonfinite_condition_fraction'] - 0.075) < 1e-9
    assert rep['marginal']['verdict'] == 'calibrated'


def test_mostly_nonfinite_raises():
    s, x = make(1.0, M=100, n=50)
    s[:80, 0, 0] = np.nan
    with pytest.raises(ValueError):
        calibration_report(s, x, scale=np.ones(4))


def test_detect_kind():
    from experiments.evaluate_calibration import detect_kind
    cases = [
        ('kin_cinn_s0', {'model': 'cinn', 'problem': 'kinematics'}, 'inn'),
        ('kin_inn_ml_s0', {'model': 'inn'}, 'inn'),
        ('kin_ilnn_s1', {'model': 'ilnn'}, 'ilnn'),
        ('kin_ilnn_s2', {}, 'ilnn'),
        ('kin_fmpe_s0', {'sigma_min': 1e-4}, 'fmpe'),
        ('kin_cfmpe_s0', {'sigma_min': 1e-4, 'prior_range': [0.25, 0.75]}, 'cfmpe'),
        ('kin_K16_s0', {'components': 16}, 'mdn'),
        ('kin_standard_s0', {'family': 'standard'}, 'ae'),
        ('kin_ae_s0', {'model': 'autoencoder'}, 'ae'),
        ('kin_fff_s0', {'manifold': 'euclidean'}, 'fff'),
        ('kin_mfff_wrapped_b400_s0', {'manifold': 'torus'}, 'fff'),
        ('bal_cfmpe_s0', {'sigma_min': 1e-4, 'lam_min': 10, 'lam_max': 20}, 'cfmpe'),
        ('kin_fif_s0', {'method': 'FIF'}, 'fif'),
        ('kin_bf_npse_s0', {'model': 'npse'}, None),
    ]
    for run, cfg, kind in cases:
        assert detect_kind(run, cfg) == kind, run