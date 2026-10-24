"""Shared posterior figure: one model's samples vs. the rejection ground truth for
a few test y*, in one format for every model so the figures compare at a glance.

Per y*, one row of 5 panels. Left: ground-truth (gray) and model (blue) samples
drawn as arm configurations (kinematics) or projectile trajectories (ballistics),
the re-simulated end points f(x) of the model samples, y* marked, and the valid
region shaded green. Right: the four 1-D marginals of x, ground truth as gray
filled bars vs. the model as a blue step outline.

Valid region (shaded green), the bulk / support of the prior:
    kinematics   |x_j| <= 3 sigma_j: rail |x0| <= 0.75, joint angles |phi_i| <= 1.5
    ballistics   |x0| <= 1.5, 0 <= y0 <= 3.0, angle 9 deg .. 81 deg,
                 v0 in the Poisson(15) 0.1 % .. 99.9 % quantile band
The row title gives the fraction of samples inside all four bounds.

Any model plugs in through `posterior_figure(problem, sample, label, ...)` with
`sample(y_star (M, dim_y), n) -> x (M, n, 4)`. The command line loads a trained
run (MDN, hand-rolled FMPE, BayesFlow FMPE, diffusion, standard AE, cVAE,
invertible AE or FIF) and calls it:

    .venv/bin/python experiments/plot_posterior.py kin_K16_s0
    .venv/bin/python experiments/plot_posterior.py bal_bf_fmpe_s0 --sampler euler:32
    .venv/bin/python experiments/plot_posterior.py kin_fmpe_s0 --sampler midpoint:16 --index 0 1 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
os.environ.setdefault('TQDM_DISABLE', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import poisson  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics.benchmarks import forward_fn, get_model, load_test_conditions  # noqa: E402
from metrics.ground_truth import ground_truth_posterior  # noqa: E402

HERE = Path(__file__).parent
RUNS = HERE / 'runs'
FIGURES = HERE / 'figures'
VALID_COLOR = 'tab:green'
V0_LAMBDA = 15  # ballistics prior v0 ~ Poisson(15), data/toy_data.py

NAMES = {'ballistics': ['x0 (launch x)', 'y0 (launch height)', 'angle [rad]', 'v0'],
         'kinematics': ['x0 (rail offset)', 'joint angle 1 [rad]', 'joint angle 2 [rad]', 'joint angle 3 [rad]']}


def valid_ranges(problem):
    """Per-dimension (lo, hi) of the valid region."""
    if problem == 'ballistics':
        return [(-1.5, 1.5), (0.0, 3.0), (0.1 * np.pi / 2, 0.9 * np.pi / 2),
                (poisson.ppf(0.001, V0_LAMBDA), poisson.ppf(0.999, V0_LAMBDA))]
    return [(-3 * s, 3 * s) for s in get_model('kinematics').sigmas]


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


def draw_left(ax, problem, sim, x, gt, resim, y, ranges, label, n_draw):
    """Samples in the problem's physical space, valid region, re-simulated end points, y*."""
    if problem == 'ballistics':
        (xl, xh), (yl, yh) = ranges[0], ranges[1]
        ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, color=VALID_COLOR, alpha=0.15, lw=0,
                                   label='valid launch region'))
        draw_trajectories(ax, sim, gt[:n_draw], 'gray', 0.15, 'ground truth')
        draw_trajectories(ax, sim, x[:n_draw], 'tab:blue', 0.12, label)
        ax.axhline(0, color='gray', lw=0.5)
        ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', zorder=3, label=f'f(x), {label}')
        ax.scatter(y[0], 0, marker='+', s=200, color='k', zorder=4, label='y*')
        ax.set_ylabel('height')
    else:
        lo, hi = ranges[0]
        ax.add_patch(plt.Rectangle((-0.06, lo), 0.12, hi - lo, color=VALID_COLOR, alpha=0.35, lw=0,
                                   label='valid rail range'))
        draw_arms(ax, sim, gt[:n_draw], 'gray', 0.15, 'ground truth')
        draw_arms(ax, sim, x[:n_draw], 'tab:blue', 0.12, label)
        ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', zorder=3, label=f'f(x), {label}')
        ax.scatter(*y, marker='+', s=200, color='k', zorder=4, label='y*')
        ax.set_aspect('equal', adjustable='box')


def posterior_figure(problem, sample, label, indices=(0, 1, 2), n=1000, n_draw=100, suptitle=None, out=None):
    """The 5-column posterior figure for `sample(y_star, n) -> (M, n, 4)` on the
    test y* `indices`. Saves to `out` if given and returns the figure."""
    y_star, _ = load_test_conditions(problem)
    gt_all = ground_truth_posterior(problem, y_star, verbose=False)
    sim, f = get_model(problem), forward_fn(problem)
    ranges = valid_ranges(problem)
    indices = list(indices)
    x_all = np.asarray(sample(y_star[indices], n), dtype=np.float64)

    fig, axs = plt.subplots(len(indices), 5, figsize=(17, 3.4 * len(indices)), squeeze=False,
                            gridspec_kw={'width_ratios': [1.6, 1, 1, 1, 1]})
    titles = []
    for r, (row, i, x) in enumerate(zip(axs, indices, x_all)):
        y, gt = y_star[i], gt_all[i]
        resim = f(x)
        err = np.nanmean(((resim - y) ** 2).sum(1))
        draw_left(row[0], problem, sim, x, gt, resim, y, ranges, label, n_draw)
        y_txt = f'{y[0]:.2f}' if problem == 'ballistics' else f'({y[0]:.2f}, {y[1]:.2f})'
        # Row title as figure text after layout (an axes title is clipped by the narrow arm panel);
        # the blank axes title reserves its space.
        titles.append(f'y* #{i} = {y_txt}   resim {err:.1e}   valid: '
                      f'{label} {100 * valid_fraction(x, ranges):.1f}%, GT {100 * valid_fraction(gt, ranges):.1f}%')
        row[0].set_title(' ', fontsize=9)
        if r == 0 and problem == 'ballistics':
            row[0].legend(loc='upper right', fontsize=7)
        elif r == 0:  # the equal-aspect arm panel is narrow: legend in the free space to its right
            row[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)

        for d, axd in enumerate(row[1:]):
            lo, hi = np.nanquantile(np.concatenate([x[:, d], gt[:, d]]), [0.002, 0.998])
            lo, hi = min(lo, ranges[d][0]), max(hi, ranges[d][1])  # show the valid range's edges
            if problem == 'ballistics' and d == 3:  # integer v0: one bin per value
                bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
            else:
                bins = np.linspace(lo, hi, 50)
            axd.axvspan(*ranges[d], color=VALID_COLOR, alpha=0.12, lw=0, label='valid range')
            axd.hist(gt[:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt)})')
            axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5,
                     label=f'{label} ({len(x)})')
            axd.set_xlabel(NAMES[problem][d])
            axd.set_yticks([])
        if r == 0:
            row[1].legend(fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(h_pad=2.2)
    for row, title in zip(axs, titles):
        fig.text(0.01, row[1].get_position().y1 + 0.004, title, fontsize=9, ha='left', va='bottom')
    if out is not None:
        fig.savefig(out, dpi=110)
        print(f'saved {out}')
    return fig


# --- Loaders for the trained runs on this branch: run name -> (problem, sample, label, tag) ---

def load_mdn(run, args):
    """MDN run (experiments/train_mdn.py). The architecture is read off the state dict, so
    runs without a config.json load too; the activation is train_mdn.py's default, relu."""
    from models.mdn import MDN, n_tril_entries
    sd = torch.load(RUNS / run / 'model.pt', map_location='cpu')
    y_dims, x_dims = len(sd['y_mean']), len(sd['x_mean'])
    hidden, n_layers = sd['trunk.0.weight'].shape[0], sum(k.endswith('.weight') for k in sd if k.startswith('trunk.'))
    K = sd['head.weight'].shape[0] // (1 + x_dims + n_tril_entries(x_dims))
    model = MDN(y_dims, x_dims, K, hidden, n_layers, 'relu')
    model.load_state_dict(sd)
    model.eval()
    problem = 'kinematics' if y_dims == 2 else 'ballistics'
    sample = lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32), n).numpy()  # noqa: E731
    return problem, sample, 'MDN', f'mdn_posterior_{run}', f'MDN (K = {K}) posterior vs. rejection ground truth, run {run}'


def load_fmpe(run, args):
    """Hand-rolled FMPE run (experiments/train_fmpe.py)."""
    from experiments.evaluate_fmpe import load_run
    from models.fmpe import nfe
    model, cfg = load_run(run)
    method, steps = args.sampler.split(':')
    sample = lambda y, n: model.sample(y, n, steps=int(steps), method=method).numpy()  # noqa: E731
    return (cfg['problem'], sample, 'FMPE', f'fmpe_posterior_{run}_{method}x{steps}',
            f'FMPE posterior vs. rejection ground truth, run {run}, {method} x {steps} ({nfe(method, int(steps))} NFE)')


def load_bayesflow(run, args):
    """BayesFlow FMPE (experiments/bayesflow_fmpe.py) or diffusion (bayesflow_npse.py) run."""
    import keras
    import bayesflow  # noqa: F401  (registers the serializable classes)
    approximator = keras.saving.load_model(RUNS / run / 'approximator.keras')
    keras.utils.set_random_seed(0)
    problem = json.load(open(RUNS / run / 'config.json'))['problem']
    method, steps = args.sampler.split(':')
    steps = steps if steps == 'adaptive' else int(steps)
    sample = lambda y, n: approximator.sample(  # noqa: E731
        num_samples=n, conditions={'inference_conditions': np.asarray(y, 'float32')},
        method=method, steps=steps)['inference_variables']
    kind, label = ('npse', 'diffusion') if '_bf_npse' in run else ('bf_fmpe', 'BayesFlow FMPE')
    return (problem, sample, label, f'{kind}_posterior_{run}_{method}x{steps}',
            f'{label} posterior vs. rejection ground truth, run {run}, {method} x {steps}')


AE_LABELS = {'standard': 'standard AE', 'cvae': 'cVAE', 'invertible': 'invertible AE'}


def load_autoencoder(run, args):
    """Standard AE, cVAE or invertible AE run (experiments/autoencoder_bench.py)."""
    from experiments.autoencoder_bench import load_run
    model, cfg, _ = load_run(run)
    label = AE_LABELS[cfg['family']]
    sample = lambda y, n: model.sample(y, n).detach().cpu().numpy()  # noqa: E731
    return (cfg['problem'], sample, label, f'ae_posterior_{run}',
            f'{label} posterior vs. rejection ground truth, run {run}')


def load_fif(run, args):
    """Free-form injective flow run (experiments/train_fif.py)."""
    from experiments.evaluate_fif import load_run
    model, cfg, _ = load_run(run)
    sample = lambda y, n: model.sample(y, n).detach().cpu().numpy()  # noqa: E731
    return cfg['problem'], sample, 'FIF', f'fif_posterior_{run}', f'FIF posterior vs. rejection ground truth, run {run}'


def load(run, args):
    if '_bf_' in run:
        return load_bayesflow(run, args)
    if any(f'_{family}_' in run for family in AE_LABELS):
        return load_autoencoder(run, args)
    if '_fif' in run:
        return load_fif(run, args)
    if '_fmpe' in run:
        return load_fmpe(run, args)
    if '_K' in run:
        return load_mdn(run, args)
    raise ValueError(f'no loader for run {run!r}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0, 1, 2], help='test y* indices, one row each')
    p.add_argument('--sampler', default='euler:32',
                   help='method:steps for the ODE / SDE models (comparison headline: FMPE midpoint:16, '
                        'BayesFlow FMPE and diffusion euler:32)')
    p.add_argument('--n', type=int, default=1000, help='samples per y*')
    p.add_argument('--n-draw', type=int, default=100, help='arms / trajectories drawn per source')
    args = p.parse_args()

    problem, sample, label, tag, suptitle = load(args.run, args)
    torch.manual_seed(0)
    np.random.seed(0)
    fig = posterior_figure(problem, sample, label, args.index, args.n, args.n_draw, suptitle, FIGURES / f'{tag}.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
