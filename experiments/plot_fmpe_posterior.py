"""Sanity plot (FMPE build-order step 4): FMPE posterior samples for a few test
y* next to the rejection-sampling ground truth.

Per y*, left: arms (kinematics) or trajectories (ballistics) of FMPE samples,
with their re-simulated end points; right: the four 1-D marginals of x, FMPE vs
ground truth.

    .venv/bin/python experiments/plot_fmpe_posterior.py kin_fmpe_s0 --index 0 1 2
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
    out = Path(__file__).parent / 'figures' / f'fmpe_posterior_{args.run}.png'
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
