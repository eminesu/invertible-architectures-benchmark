"""Prior-conditioned FMPE posterior vs. the prior-dependent rejection ground
truth, one row per prior setting, for fixed test y*.

Mirrors experiments/plot_fmpe_posterior.py: left, ground-truth (gray) and
conditional-FMPE (blue) samples drawn as projectile trajectories (ballistics) or
arm configurations (kinematics), with y* marked; right, the four 1-D marginals
of x. The y* is the same in every row, only the conditioned prior changes, so
the rows show the posterior shifting with the prior.

Valid region (shaded green), the support / bulk of the prior. Ballistics:
    launch x0     |x0| <= 1.5        (3 sigma of N(0, 0.5^2))
    launch y0     0 <= y0 <= 3.0     (prior clamps y0 at 0; 3 sigma above N(1.5, 0.5^2))
    angle         9 deg .. 81 deg    (the prior's hard bounds, 0.1 .. 0.9 x pi/2)
    v0            Poisson(lambda) 0.1 % .. 99.9 % quantiles (depends on lambda)
Kinematics: |x_j| <= 3 sigma_j for each of the four dimensions (rail offset with
the fixed sigma_0 = 0.25, joint angles with the conditioned sigma_1..3), i.e.
joint limits at 3 standard deviations of the assumed prior.
The left panel shades the launch box (ballistics) or the rail range
(kinematics); each marginal shades its range. The row title gives the fraction
of samples inside all four bounds.

    .venv/bin/python experiments/plot_conditional_fmpe_posterior.py bal_cfmpe_s0 --index 200 346 966 --prior 12 15 18
    .venv/bin/python experiments/plot_conditional_fmpe_posterior.py kin_cfmpe_s0 --index 0 1 2 \
        --prior 0.35,0.35,0.35 0.5,0.5,0.5 0.65,0.65,0.65
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


NAMES = {'ballistics': ['x0 (launch x)', 'y0 (launch height)', 'angle [rad]', 'v0'],
         'kinematics': ['x0 (rail offset)', 'joint angle 1 [rad]', 'joint angle 2 [rad]', 'joint angle 3 [rad]']}


def valid_ranges(problem, c):
    """Per-dimension (lo, hi) of the valid region for prior parameters c."""
    if problem == 'ballistics':
        return [(-1.5, 1.5), (0.0, 3.0), (0.1 * np.pi / 2, 0.9 * np.pi / 2),
                (poisson.ppf(0.001, c[0]), poisson.ppf(0.999, c[0]))]
    sigmas = [get_model('kinematics').sigmas[0], *c]
    return [(-3 * s, 3 * s) for s in sigmas]


def valid_fraction(x, ranges):
    ok = np.ones(len(x), bool)
    for d, (lo, hi) in enumerate(ranges):
        ok &= (x[:, d] >= lo) & (x[:, d] <= hi)
    return ok.mean()


def prior_label(problem, c):
    return f'lambda = {c[0]:g}' if problem == 'ballistics' else 'sigma_1..3 = (' + ', '.join(f'{v:g}' for v in c) + ')'


def draw_trajectories(ax, sim, x, color, alpha, label):
    xs, ys = sim.trajectories_from_parameters(x)
    for i, (xt, yt) in enumerate(zip(xs, ys)):
        keep = yt > -0.5
        ax.plot(xt[keep], yt[keep], color=color, alpha=alpha, lw=1, label=label if i == 0 else None)


def draw_arms(ax, sim, x, color, alpha, label):
    for i, arm in enumerate(np.stack(sim.joint_positions(x), axis=1)):
        ax.plot(arm[:, 0], arm[:, 1], color=color, alpha=alpha, lw=1, label=label if i == 0 else None)


def draw_left(ax, problem, sim, x, gt, resim, y, ranges, n_draw):
    """Samples in the problem's physical space, valid region, re-simulated end points, y*."""
    if problem == 'ballistics':
        (xl, xh), (yl, yh) = ranges[0], ranges[1]
        ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, color=VALID_COLOR, alpha=0.15, lw=0,
                                   label='valid launch region'))
        draw_trajectories(ax, sim, gt[:n_draw], 'gray', 0.15, 'ground truth')
        draw_trajectories(ax, sim, x[:n_draw], 'tab:blue', 0.12, 'conditional FMPE')
        ax.axhline(0, color='gray', lw=0.5)
        ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', zorder=3, label='f(x), FMPE samples')
        ax.scatter(y[0], 0, marker='+', s=200, color='k', zorder=4, label='y*')
        ax.set_ylabel('height')
    else:
        lo, hi = ranges[0]
        ax.add_patch(plt.Rectangle((-0.06, lo), 0.12, hi - lo, color=VALID_COLOR, alpha=0.35, lw=0,
                                   label='valid rail range'))
        draw_arms(ax, sim, gt[:n_draw], 'gray', 0.15, 'ground truth')
        draw_arms(ax, sim, x[:n_draw], 'tab:blue', 0.12, 'conditional FMPE')
        ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', zorder=3, label='f(x), FMPE samples')
        ax.scatter(*y, marker='+', s=200, color='k', zorder=4, label='y*')
        ax.set_aspect('equal', adjustable='box')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0], help='test y* indices, one figure each')
    p.add_argument('--prior', nargs='+', required=True, help='prior settings, e.g. 12 or 0.35,0.35,0.35')
    p.add_argument('--solver', default='midpoint:16')
    p.add_argument('--n-draw', type=int, default=100, help='trajectories drawn per source')
    args = p.parse_args()

    model, cfg = load_run(args.run)
    problem = cfg['problem']
    method, steps = args.solver.split(':')
    y_star, _ = load_test_conditions(problem)
    sim, f = get_model(problem), forward_fn(problem)
    priors = [tuple(float(v) for v in s.split(',')) for s in args.prior]
    gts = {c: ground_truth_posterior(problem, y_star, verbose=False, prior_params=c) for c in priors}

    for i in args.index:
        y = y_star[i]
        fig, axs = plt.subplots(len(priors), 5, figsize=(17, 3.4 * len(priors)), squeeze=False,
                                gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
        for a in axs[1:, 0]:
            a.sharex(axs[0, 0]); a.sharey(axs[0, 0])
        for d in range(1, 5):  # same x range per marginal across priors, so shifts read directly
            for a in axs[1:, d]:
                a.sharex(axs[0, d])
        lims = [[np.inf, -np.inf] for _ in range(4)]
        titles = []
        for r, (row, c) in enumerate(zip(axs, priors)):
            torch.manual_seed(0)
            x = model.sample(y[None], c, 1000, steps=int(steps), method=method)[0].numpy().astype(np.float64)
            gt = gts[c][i]
            resim = f(x)
            err = np.nanmean(((resim - y) ** 2).sum(1))
            ranges = valid_ranges(problem, c)

            ax = row[0]
            draw_left(ax, problem, sim, x, gt, resim, y, ranges, args.n_draw)
            y_txt = f'{y[0]:.2f}' if problem == 'ballistics' else f'({y[0]:.2f}, {y[1]:.2f})'
            # Row title as figure text after layout (an axes title is clipped by the narrow arm panel);
            # the blank axes title reserves its space.
            titles.append(f'{prior_label(problem, c)}   y* #{i} = {y_txt}   resim {err:.1e}   valid: '
                          f'FMPE {100 * valid_fraction(x, ranges):.1f}%, GT {100 * valid_fraction(gt, ranges):.1f}%')
            ax.set_title(' ', fontsize=9)
            if r == 0:
                ax.legend(loc='upper right', fontsize=7)

            for d, axd in enumerate(row[1:]):
                lo, hi = np.quantile(np.concatenate([x[:, d], gt[:, d]]), [0.002, 0.998])
                lo, hi = min(lo, ranges[d][0]), max(hi, ranges[d][1])  # show the valid range's edges
                if problem == 'ballistics' and d == 3:  # integer v0: one bin per value
                    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
                else:
                    bins = np.linspace(lo, hi, 50)
                axd.axvspan(*ranges[d], color=VALID_COLOR, alpha=0.12, lw=0, label='valid range')
                axd.hist(gt[:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt)})')
                axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5,
                         label='conditional FMPE')
                pad = 0.05 * (bins[-1] - bins[0])
                lims[d] = [min(lims[d][0], bins[0] - pad), max(lims[d][1], bins[-1] + pad)]
                axd.set_xlabel(NAMES[problem][d])
                axd.set_yticks([])
                if problem == 'ballistics' and d == 3:
                    axd.axvline(c[0], color='k', lw=1, ls=':', label=f'prior mean {c[0]:g}')
            if r == 0:
                row[1].legend(fontsize=7)
            if problem == 'ballistics':
                row[4].legend(fontsize=7)
        for d in range(4):
            axs[0, d + 1].set_xlim(*lims[d])
        prior_txt = 'v0 ~ Poisson(lambda)' if problem == 'ballistics' else 'joint angles ~ N(0, sigma_j^2)'
        fig.suptitle(f'Prior-conditioned FMPE vs. rejection ground truth under {prior_txt}, '
                     f'run {args.run}, {method} x {steps} ({nfe(method, int(steps))} NFE)')
        fig.tight_layout(h_pad=2.2)
        for row, title in zip(axs, titles):
            fig.text(0.01, row[1].get_position().y1 + 0.004, title, fontsize=9, ha='left', va='bottom')
        out = Path(__file__).parent / 'figures' / f'cfmpe_posterior_{args.run}_y{i}.png'
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
