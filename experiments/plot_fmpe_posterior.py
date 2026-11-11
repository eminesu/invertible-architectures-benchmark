"""Sanity plot (FMPE build-order step 4): FMPE posterior samples for a few test
y* next to the rejection-sampling ground truth.

Per y*, left: arms (kinematics) or trajectories (ballistics) of FMPE samples,
with their re-simulated end points; right: the four 1-D marginals of x, FMPE vs
ground truth.

FR3 runs (fr3_6d, fr3_5d): per y*, the seven joint marginals, FMPE vs the exact
ground truth, with the URDF joint limits as dashed lines (no arm drawing). For
fr3_5d the flat density of the free joint 7 is drawn as a dotted line.

    .venv/bin/python experiments/plot_fmpe_posterior.py kin_fmpe_s0 --index 0 1 2
    .venv/bin/python experiments/plot_fmpe_posterior.py fr3_6d_fmpe_s0 --solver midpoint:16
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.evaluate_fmpe import load_run
from metrics.benchmarks import forward_fn, get_model, load_test_conditions
from metrics.ground_truth import ground_truth_posterior
from models.fmpe import nfe


def plot_fr3(args, model, bench, method, steps):
    """Seven joint marginals per y*, FMPE vs ground truth, joint limits marked."""
    from robot_ik.kinematics import joint_limits
    lo_lim, hi_lim = (np.asarray(a, dtype=np.float64) for a in joint_limits())
    y_star, _ = load_test_conditions(bench)
    gt = ground_truth_posterior(bench, y_star, verbose=False)
    torch.manual_seed(0)
    x_all = model.sample(y_star[args.index], 1000, steps=int(steps), method=method).numpy().astype(np.float64)

    fig, axs = plt.subplots(len(args.index), 7, figsize=(22, 2.9 * len(args.index)), squeeze=False)
    for row, i, x in zip(axs, args.index, x_all):
        out = ((x < lo_lim) | (x > hi_lim)).any(1).mean()
        for d, axd in enumerate(row):
            lo, hi = np.quantile(np.concatenate([x[:, d], gt[i][:, d]]), [0.002, 0.998])
            lo, hi = min(lo, lo_lim[d]), max(hi, hi_lim[d])
            pad = 0.03 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 60)
            axd.hist(gt[i][:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt[i])})')
            axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5, label='FMPE')
            for b in (lo_lim[d], hi_lim[d]):
                axd.axvline(b, color='tab:red', ls='--', lw=1, label='joint limit' if b == lo_lim[d] else None)
            if bench == 'fr3_5d' and d == 6:
                axd.axhline(1 / (hi_lim[d] - lo_lim[d]), color='k', ls=':', lw=1.2, label='uniform')
            axd.set_xlabel(f'joint {d + 1} (rad)')
            axd.set_yticks([])
        row[0].set_ylabel(f'y* #{i}\noutside limits {out:.1%}', fontsize=9)
        row[0].legend(fontsize=7)
        if bench == 'fr3_5d':
            row[6].legend(fontsize=7)
    fig.suptitle(f'FMPE joint marginals vs. exact ground truth, run {args.run}, '
                 f'{method} x {steps} ({nfe(method, int(steps))} NFE)')
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--solver', default='rk4:8')
    p.add_argument('--n-draw', type=int, default=150)
    args = p.parse_args()

    model, cfg = load_run(args.run)
    bench = cfg['problem']
    method, steps = args.solver.split(':')
    out = Path(__file__).parent / 'figures' / 'fmpe' / 'posterior' / f'fmpe_posterior_{args.run}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    if bench.startswith('fr3'):
        plot_fr3(args, model, bench, method, steps).savefig(out, dpi=110)
        print(f'saved {out}')
        return
    y_star, _ = load_test_conditions(bench)
    gt = ground_truth_posterior(bench, y_star, verbose=False)
    sim, f = get_model(bench), forward_fn(bench)
    torch.manual_seed(0)
    x_all = model.sample(y_star[args.index], 1000, steps=int(steps), method=method).numpy().astype(np.float64)

    fig, axs = plt.subplots(len(args.index), 5, figsize=(17, 3.4 * len(args.index)), squeeze=False,
                            gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
    for row, i, x in zip(axs, args.index, x_all):
        y, ax = y_star[i], row[0]
        resim = f(x)
        err = np.nanmean(((resim - y) ** 2).sum(1))
        draw = x[:args.n_draw]
        if bench == 'kinematics':
            for arm in np.stack(sim.joint_positions(draw), axis=1):
                ax.plot(arm[:, 0], arm[:, 1], color='tab:blue', alpha=0.12, lw=1)
            ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', label='f(x), FMPE samples')
            ax.scatter(*y, marker='+', s=200, color='k', label='y*')
            ax.set_aspect('equal')
        else:
            xs, ys = sim.trajectories_from_parameters(draw)
            for xt, yt in zip(xs, ys):
                keep = yt > -0.5
                ax.plot(xt[keep], yt[keep], color='tab:blue', alpha=0.12, lw=1)
            ax.axhline(0, color='gray', lw=0.5)
            ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', label='f(x), FMPE samples')
            ax.scatter(y[0], 0, marker='+', s=200, color='k', label='y*')
        ax.set_title(f'y* #{i}  resim {err:.1e}', fontsize=10)
        ax.legend(loc='best', fontsize=7)
        for d, axd in enumerate(row[1:]):
            lo, hi = np.quantile(np.concatenate([x[:, d], gt[i][:, d]]), [0.002, 0.998])
            bins = np.linspace(lo, hi, 50)
            axd.hist(gt[i][:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt[i])})')
            axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5, label='FMPE')
            axd.set_xlabel(f'x{d + 1}')
            axd.set_yticks([])
        row[1].legend(fontsize=7)
    fig.suptitle(f'FMPE posterior vs. rejection ground truth, run {args.run}, '
                 f'{method} x {steps} ({nfe(method, int(steps))} NFE)')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
