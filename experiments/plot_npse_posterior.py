"""Sanity check (NPSE build-order step 4): diffusion posterior samples for a few
test y* next to the rejection-sampling ground truth.

Per y*, left: arms (kinematics) or trajectories (ballistics) of the samples, with
their re-simulated end points f(x) against y*; right: the four 1-D marginals of
x, diffusion vs ground truth. Also prints, per y*, the re-simulation error, the
per-dimension std ratio (model / ground truth) and the MMD^2 of metrics.evaluate.

    .venv/bin/python experiments/plot_npse_posterior.py kin_bf_npse_s0 --index 0 1 2 --sampler euler_maruyama:128
"""

import argparse
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
os.environ.setdefault('TQDM_DISABLE', '1')

import keras  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import bayesflow  # noqa: E402,F401  (registers the serializable classes)
from metrics.benchmarks import forward_fn, get_model, load_test_conditions  # noqa: E402
from metrics.ground_truth import ground_truth_posterior  # noqa: E402
from metrics.mmd import mmd  # noqa: E402

HERE = os.path.dirname(__file__)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--sampler', default='euler_maruyama:128', help='method:steps')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--n-draw', type=int, default=150)
    args = p.parse_args()

    approximator = keras.saving.load_model(os.path.join(HERE, 'runs', args.run, 'approximator.keras'))
    bench = 'kinematics' if args.run.startswith('kin') else 'ballistics'
    method, steps = args.sampler.split(':')
    steps = steps if steps == 'adaptive' else int(steps)
    y_star, _ = load_test_conditions(bench)
    gt = ground_truth_posterior(bench, y_star, verbose=False)
    sim, f = get_model(bench), forward_fn(bench)
    keras.utils.set_random_seed(0)
    x_all = approximator.sample(num_samples=args.n, conditions={'inference_conditions': y_star[args.index]},
                                method=method, steps=steps)['inference_variables'].astype(np.float64)

    fig, axs = plt.subplots(len(args.index), 5, figsize=(17, 3.4 * len(args.index)), squeeze=False,
                            gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
    for row, i, x in zip(axs, args.index, x_all):
        y, ax = y_star[i], row[0]
        resim = f(x)
        err = np.nanmean(((resim - y) ** 2).sum(1))
        ratio = x.std(0) / gt[i].std(0)
        print(f'y* #{i} = {np.round(y, 3)}: resim {err:.2e}, MMD^2 {mmd(x, gt[i]):.4f}, '
              f'std ratio model/gt {np.round(ratio, 2)}, f(x) undefined {np.isnan(resim).any(1).mean():.1%}')
        draw = x[:args.n_draw]
        if bench == 'kinematics':
            for arm in np.stack(sim.joint_positions(draw), axis=1):
                ax.plot(arm[:, 0], arm[:, 1], color='tab:blue', alpha=0.12, lw=1)
            ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', label='f(x), diffusion samples')
            ax.scatter(*y, marker='+', s=200, color='k', label='y*')
            ax.set_aspect('equal')
        else:
            xs, ys = sim.trajectories_from_parameters(draw)
            for xt, yt in zip(xs, ys):
                keep = yt > -0.5
                ax.plot(xt[keep], yt[keep], color='tab:blue', alpha=0.12, lw=1)
            ax.axhline(0, color='gray', lw=0.5)
            ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', label='f(x), diffusion samples')
            ax.scatter(y[0], 0, marker='+', s=200, color='k', label='y*')
        ax.set_title(f'y* #{i}  resim {err:.1e}', fontsize=10)
        ax.legend(loc='best', fontsize=7)
        for d, axd in enumerate(row[1:]):
            lo, hi = np.quantile(np.concatenate([x[:, d], gt[i][:, d]]), [0.002, 0.998])
            bins = np.linspace(lo, hi, 50)
            axd.hist(gt[i][:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt[i])})')
            axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5, label='diffusion')
            axd.set_xlabel(f'x{d + 1}')
            axd.set_yticks([])
        row[1].legend(fontsize=7)
    fig.suptitle(f'Diffusion (BayesFlow) posterior vs. rejection ground truth, run {args.run}, {method} x {steps}')
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', f'npse_posterior_{args.run}_{method}x{steps}.png')
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
