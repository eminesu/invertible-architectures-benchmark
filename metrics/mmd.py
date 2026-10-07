"""Maximum Mean Discrepancy between two sample sets, Kruse et al. (2021) Eq. (2),
following Gretton et al. (2012), JMLR 13:723-773.

    MMD^2(A, B) = E k(a, a') - 2 E k(a, b) + E k(b, b')

Kernel: a sum of inverse multiquadratic (IMQ) kernels over several bandwidths,

    k(a, b) = sum_h  1 / (1 + ||a - b||^2 / h^2),    h in IMQ_BANDWIDTHS.

Neither the benchmark paper nor vislearn/inn_toy_data pins a kernel. This is the
multi-scale IMQ kernel used by the same group's INN code (Ardizzone et al. 2019,
FrEIA toy demos: `a**2 / (a**2 + dxx)` for a in [0.05, 0.2, 0.9]), so it is the
closest thing to a group convention. IMQ has heavy tails, so the gradient/signal
does not vanish for far-apart sets the way a narrow Gaussian kernel's does, and
summing three widths makes the value insensitive to any single length scale.

The kernel is applied to x in raw (prior) units. Every model is scored with the
same kernel, so this is a fair comparison whichever way the units go.
"""

import numpy as np

IMQ_BANDWIDTHS = (0.05, 0.2, 0.9)


def sq_dists(a, b):
    """Pairwise squared Euclidean distances, (n, d) x (m, d) -> (n, m)."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)  # clip tiny negatives from cancellation


def imq_kernel(d2, bandwidths=IMQ_BANDWIDTHS):
    """Multi-scale IMQ kernel evaluated on a matrix of squared distances."""
    return sum(1.0 / (1.0 + d2 / (h * h)) for h in bandwidths)


def mmd(a, b, kernel=imq_kernel, unbiased=True):
    """Estimate MMD^2 between samples a (n, d) and b (m, d).

    unbiased=True (default) drops the i == j terms of the within-set means
    (the U-statistic of Gretton et al., Lemma 6). The biased V-statistic keeps
    them; it adds roughly k(0)(1/n + 1/m) = 3(1/n + 1/m) for this kernel, i.e.
    +0.006 at n = m = 1000, which is the same order as the paper's best
    numbers, so it is not used for reporting. The unbiased estimate can be
    slightly negative when the two distributions match.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    k_aa = kernel(sq_dists(a, a))
    k_bb = kernel(sq_dists(b, b))
    k_ab = kernel(sq_dists(a, b))
    if unbiased:
        term_aa = (k_aa.sum() - np.trace(k_aa)) / (n * (n - 1))
        term_bb = (k_bb.sum() - np.trace(k_bb)) / (m * (m - 1))
    else:
        term_aa, term_bb = k_aa.mean(), k_bb.mean()
    return float(term_aa + term_bb - 2.0 * k_ab.mean())


def mmd_torch(a, b, kernel=imq_kernel, unbiased=True):
    """Differentiable torch version of `mmd`, for MMD training losses (e.g. the
    autoencoder's latent loss, Kruse et al. Eq. 3). Same kernel, same estimator;
    tests/test_metrics.py checks it agrees with `mmd`. Returns a 0-d tensor."""
    import torch

    def d2(p, q):
        return torch.clamp((p * p).sum(1)[:, None] + (q * q).sum(1)[None, :] - 2.0 * p @ q.T, min=0.0)

    n, m = len(a), len(b)
    k_aa, k_bb, k_ab = kernel(d2(a, a)), kernel(d2(b, b)), kernel(d2(a, b))
    if unbiased:
        term_aa = (k_aa.sum() - k_aa.diagonal().sum()) / (n * (n - 1))
        term_bb = (k_bb.sum() - k_bb.diagonal().sum()) / (m * (m - 1))
    else:
        term_aa, term_bb = k_aa.mean(), k_bb.mean()
    return term_aa + term_bb - 2.0 * k_ab.mean()
