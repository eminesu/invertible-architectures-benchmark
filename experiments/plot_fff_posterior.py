"""Conditional FFF posterior vs. the rejection ground truth for fixed test y*,
one row per FFF run (Euclidean FFF vs. torus M-FFF).

Mirrors the conditional-FMPE posterior figure
(experiments/plot_conditional_fmpe_posterior.py on emine-conditional-fmpe;
same layout, colours, y* marker and green valid region; its helpers are copied
here so this script runs on main); only the rows differ: there one row per
prior lambda, here one row per model, so the figure shows what embedding the
periodic angles on a torus changes against the plain Euclidean FFF. Left:
kinematics arms / ballistics trajectories of ground-truth (gray) and FFF
(blue) samples with y* marked; right: the four 1-D marginals of x. One figure
per y*.

Valid region (shaded green), the bulk of the prior:
  kinematics  rail offset |x0| <= 0.75 and joint angles |phi_i| <= 1.5 rad
              (3 sigma of N(0, 0.25^2) and N(0, 0.5^2)); the arm panel shades
              the rail segment the arm may start on.
  ballistics  as in the conditional-FMPE figure at lambda = 15: |x0| <= 1.5,
              0 <= y0 <= 3, angle 9-81 deg, v0 in the Poisson(15) 0.1-99.9 %
              quantiles; the trajectory panel shades the launch box.
The row title gives the fraction of samples inside all four bounds.

    .venv/bin/python experiments/plot_fff_posterior.py kin_fff_s0 kin_mfff_s0 --index 52 6 296
    .venv/bin/python experiments/plot_fff_posterior.py bal_fff_s0 --index 200 346 966
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
from experiments.evaluate_fff import load_run
from metrics.benchmarks import forward_fn, get_model, load_test_conditions
from metrics.ground_truth import ground_truth_posterior

VALID_COLOR = 'tab:green'

NAMES = {
    'kinematics': ['x0 (rail offset)', 'phi1 [rad]', 'phi2 [rad]', 'phi3 [rad]'],
    'ballistics': ['x0 (launch x)', 'y0 (launch height)', 'angle [rad]', 'v0'],
}
LABELS = {'euclidean': 'Euclidean FFF', 'torus': 'torus M-FFF'}


def valid_ranges(bench, lam=15):
    """Per-dimension (lo, hi) of the valid region; ballistics as in the
    conditional-FMPE figure, with v0 between the Poisson(lam) 0.1 / 99.9 % quantiles."""
    if bench == 'kinematics':
        return [(-0.75, 0.75)] + [(-1.5, 1.5)] * 3
    return [(-1.5, 1.5), (0.0, 3.0), (0.1 * np.pi / 2, 0.9 * np.pi / 2),
            (poisson.ppf(0.001, lam), poisson.ppf(0.999, lam))]


def valid_fraction(x, ranges):
    ok = np.ones(len(x), bool)
    for d, (lo, hi) in enumerate(ranges):
        ok &= (x[:, d] >= lo) & (x[:, d] <= hi)
    return ok.mean()


def draw_trajectories(ax, sim, x, color, alpha, label):
    xs, ys = sim.trajectories_from_parameters(x)
    for i, (xt, yt) in enumerate(zip(xs, ys)):
        keep = yt > -0.5
        ax.plot(xt[keep], yt[keep], color=color, alpha=alpha, lw=1, label=label if i == 0 else None)


def draw_arms(ax, sim, x, color, alpha, label):
    for i, arm in enumerate(np.stack(sim.joint_positions(x), axis=1)):
        ax.plot(arm[:, 0], arm[:, 1], color=color, alpha=alpha, lw=1, label=label if i == 0 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='FFF runs of one benchmark, one row each')
    p.add_argument('--index', type=int, nargs='+', default=[0], help='test y* indices, one figure each')
    p.add_argument('--n-draw', type=int, default=100, help='arms / trajectories drawn per source')
    args = p.parse_args()

    models = [load_run(r) for r in args.runs]
    bench = models[0][1]['problem']
    assert all(cfg['problem'] == bench for _, cfg in models), 'all runs must be on one benchmark'
    y_star, _ = load_test_conditions(bench)
    gt_all = ground_truth_posterior(bench, y_star, verbose=False)
    sim, f = get_model(bench), forward_fn(bench)
    ranges, names = valid_ranges(bench), NAMES[bench]

    for i in args.index:
        y, gt = y_star[i], gt_all[i]
        y_text = f'({y[0]:.2f}, {y[1]:.2f})' if bench == 'kinematics' else f'{y[0]:.2f}'
        fig, axs = plt.subplots(len(models), 5, figsize=(17, 3.4 * len(models)), squeeze=False,
                                gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
        if bench == 'ballistics':  # kinematics arms use equal aspect, which needs unshared axes
            for a in axs[1:, 0]:
                a.sharex(axs[0, 0]); a.sharey(axs[0, 0])
        for d in range(1, 5):  # same x range per marginal across rows, so differences read directly
            for a in axs[1:, d]:
                a.sharex(axs[0, d])
        lims = [[np.inf, -np.inf] for _ in range(4)]
        for r, (row, (model, cfg)) in enumerate(zip(axs, models)):
            torch.manual_seed(0)
            x = model.sample(y[None], 1000)[0].numpy().astype(np.float64)
            resim = f(x)
            err = np.nanmean(((resim - y) ** 2).sum(1))
            label = LABELS[cfg['manifold']]
            if cfg['manifold'] == 'torus':
                label += f" ({cfg.get('latent', 'uniform')})"

            ax = row[0]
            if bench == 'kinematics':
                lo, hi = ranges[0]
                ax.add_patch(plt.Rectangle((-0.06, lo), 0.12, hi - lo, color=VALID_COLOR, alpha=0.3, lw=0,
                                           label='valid rail range'))
                draw_arms(ax, sim, gt[:args.n_draw], 'gray', 0.15, 'ground truth')
                draw_arms(ax, sim, x[:args.n_draw], 'tab:blue', 0.12, label)
                ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', zorder=3, label='f(x), FFF samples')
                ax.scatter(*y, marker='+', s=200, color='k', zorder=4, label='y*')
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_ylabel('y')
            else:
                (xl, xh), (yl, yh) = ranges[0], ranges[1]
                ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, color=VALID_COLOR, alpha=0.15, lw=0,
                                           label='valid launch region'))
                draw_trajectories(ax, sim, gt[:args.n_draw], 'gray', 0.15, 'ground truth')
                draw_trajectories(ax, sim, x[:args.n_draw], 'tab:blue', 0.12, label)
                ax.axhline(0, color='gray', lw=0.5)
                ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', zorder=3,
                           label='f(x), FFF samples')
                ax.scatter(y[0], 0, marker='+', s=200, color='k', zorder=4, label='y*')
                ax.set_ylabel('height')
            ax.set_title(f'{label}  y* #{i} = {y_text}  resim {err:.1e}  valid: '
                         f'FFF {100 * valid_fraction(x, ranges):.1f}%, GT {100 * valid_fraction(gt, ranges):.1f}%',
                         fontsize=8)
            if r == 0:
                ax.legend(loc='upper right', fontsize=7)

            for d, axd in enumerate(row[1:]):
                lo, hi = np.quantile(np.concatenate([x[:, d], gt[:, d]]), [0.002, 0.998])
                lo, hi = min(lo, ranges[d][0]), max(hi, ranges[d][1])  # show the valid range's edges
                if bench == 'ballistics' and d == 3:  # integer v0: one bin per value
                    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
                else:
                    bins = np.linspace(lo, hi, 50)
                axd.axvspan(*ranges[d], color=VALID_COLOR, alpha=0.12, lw=0, label='valid range')
                axd.hist(gt[:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt)})')
                axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5, label=label)
                pad = 0.05 * (bins[-1] - bins[0])
                lims[d] = [min(lims[d][0], bins[0] - pad), max(lims[d][1], bins[-1] + pad)]
                axd.set_xlabel(names[d])
                axd.set_yticks([])
            row[1].legend(fontsize=7)
        for d in range(4):
            axs[0, d + 1].set_xlim(*lims[d])
        fig.suptitle(f'Conditional FFF vs. rejection ground truth, {bench}, runs {", ".join(args.runs)} '
                     f'(one decoder pass, 1 NFE)')
        fig.tight_layout()
        out = Path(__file__).parent / 'figures' / f'fff_posterior_{"_".join(args.runs)}_y{i}.png'
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
