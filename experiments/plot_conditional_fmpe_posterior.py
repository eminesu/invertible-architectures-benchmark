"""Prior-conditioned FMPE posterior vs. the lambda-dependent rejection ground
truth, per prior lambda, for fixed test y*.

Mirrors experiments/plot_fmpe_posterior.py: one row per lambda; left, projectile
trajectories of ground-truth (gray) and conditional-FMPE (blue) samples with the
target impact y* marked; right, the four 1-D marginals of x. The y* is the same
in every row, only the conditioned prior v0 ~ Poisson(lambda) changes, so the
rows show the posterior shifting with the prior.

Valid region (shaded green), the support / bulk of the prior:
    launch x0     |x0| <= 1.5        (3 sigma of N(0, 0.5^2))
    launch y0     0 <= y0 <= 3.0     (prior clamps y0 at 0; 3 sigma above N(1.5, 0.5^2))
    angle         9 deg .. 81 deg    (the prior's hard bounds, 0.1 .. 0.9 x pi/2)
    v0            Poisson(lambda) 0.1 % .. 99.9 % quantiles (depends on lambda)
The trajectory panel shades the (x0, y0) launch box; each marginal shades its
range. The row title gives the fraction of samples inside all four bounds.

    .venv/bin/python experiments/plot_conditional_fmpe_posterior.py bal_cfmpe_s0 --index 0 --lam 12 15 18
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.evaluate_conditional_fmpe import load_run
from metrics.benchmarks import forward_fn, get_model, load_test_conditions
from metrics.ground_truth import ground_truth_posterior
from models.fmpe import nfe

VALID_COLOR = 'tab:green'


def valid_ranges(lam):
    """Per-dimension (lo, hi) of the valid region for prior lambda."""
    return [(-1.5, 1.5), (0.0, 3.0), (0.1 * np.pi / 2, 0.9 * np.pi / 2),
            (poisson.ppf(0.001, lam), poisson.ppf(0.999, lam))]


def valid_fraction(x, lam):
    ok = np.ones(len(x), bool)
    for d, (lo, hi) in enumerate(valid_ranges(lam)):
        ok &= (x[:, d] >= lo) & (x[:, d] <= hi)
    return ok.mean()


def draw_trajectories(ax, sim, x, color, alpha, label):
    xs, ys = sim.trajectories_from_parameters(x)
    for i, (xt, yt) in enumerate(zip(xs, ys)):
        keep = yt > -0.5
        ax.plot(xt[keep], yt[keep], color=color, alpha=alpha, lw=1, label=label if i == 0 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0], help='test y* indices, one figure each')
    p.add_argument('--lam', type=float, nargs='+', default=[12, 15, 18])
    p.add_argument('--solver', default='midpoint:16')
    p.add_argument('--n-draw', type=int, default=100, help='trajectories drawn per source')
    args = p.parse_args()

    model, cfg = load_run(args.run)
    method, steps = args.solver.split(':')
    y_star, _ = load_test_conditions('ballistics')
    sim, f = get_model('ballistics'), forward_fn('ballistics')
    gts = {lam: ground_truth_posterior('ballistics', y_star, verbose=False, lam=lam) for lam in args.lam}
    names = ['x0 (launch x)', 'y0 (launch height)', 'angle [rad]', 'v0']

    for i in args.index:
        y = y_star[i]
        fig, axs = plt.subplots(len(args.lam), 5, figsize=(17, 3.4 * len(args.lam)), squeeze=False,
                                gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
        for a in axs[1:, 0]:
            a.sharex(axs[0, 0]); a.sharey(axs[0, 0])
        for d in range(1, 5):  # same x range per marginal across lambda, so shifts read directly
            for a in axs[1:, d]:
                a.sharex(axs[0, d])
        lims = [[np.inf, -np.inf] for _ in range(4)]
        for r, (row, lam) in enumerate(zip(axs, args.lam)):
            torch.manual_seed(0)
            x = model.sample(y[None], lam, 1000, steps=int(steps), method=method)[0].numpy().astype(np.float64)
            gt = gts[lam][i]
            resim = f(x)
            err = np.nanmean(((resim - y) ** 2).sum(1))
            ranges = valid_ranges(lam)

            ax = row[0]
            (xl, xh), (yl, yh) = ranges[0], ranges[1]
            ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, color=VALID_COLOR, alpha=0.15, lw=0,
                                       label='valid launch region'))
            draw_trajectories(ax, sim, gt[:args.n_draw], 'gray', 0.15, 'ground truth')
            draw_trajectories(ax, sim, x[:args.n_draw], 'tab:blue', 0.12, 'conditional FMPE')
            ax.axhline(0, color='gray', lw=0.5)
            ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', zorder=3,
                       label='f(x), FMPE samples')
            ax.scatter(y[0], 0, marker='+', s=200, color='k', zorder=4, label='y*')
            ax.set_title(f'lambda = {lam:g}  y* #{i} = {y[0]:.2f}  resim {err:.1e}  valid: '
                         f'FMPE {100 * valid_fraction(x, lam):.1f}%, GT {100 * valid_fraction(gt, lam):.1f}%',
                         fontsize=9)
            ax.set_ylabel('height')
            if r == 0:
                ax.legend(loc='upper right', fontsize=7)

            for d, axd in enumerate(row[1:]):
                lo, hi = np.quantile(np.concatenate([x[:, d], gt[:, d]]), [0.002, 0.998])
                lo, hi = min(lo, ranges[d][0]), max(hi, ranges[d][1])  # show the valid range's edges
                if d == 3:  # integer v0: one bin per value
                    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
                else:
                    bins = np.linspace(lo, hi, 50)
                axd.axvspan(*ranges[d], color=VALID_COLOR, alpha=0.12, lw=0, label='valid range')
                axd.hist(gt[:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt)})')
                axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5,
                         label='conditional FMPE')
                pad = 0.05 * (bins[-1] - bins[0])
                lims[d] = [min(lims[d][0], bins[0] - pad), max(lims[d][1], bins[-1] + pad)]
                axd.set_xlabel(names[d])
                axd.set_yticks([])
                if d == 3:
                    axd.axvline(lam, color='k', lw=1, ls=':', label=f'prior mean {lam:g}')
            if r == 0:
                row[1].legend(fontsize=7)
            row[4].legend(fontsize=7)
        for d in range(4):
            axs[0, d + 1].set_xlim(*lims[d])
        fig.suptitle(f'Prior-conditioned FMPE vs. rejection ground truth under v0 ~ Poisson(lambda), '
                     f'run {args.run}, {method} x {steps} ({nfe(method, int(steps))} NFE)')
        fig.tight_layout()
        out = Path(__file__).parent / 'figures' / f'cfmpe_posterior_{args.run}_y{i}.png'
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
