"""Sanity plot (FMPE build-order step 4): FMPE posterior samples for a few test
y* next to the rejection-sampling ground truth.

Per y*, left: arms (kinematics) or trajectories (ballistics) of FMPE samples,
with their re-simulated end points; right: the four 1-D marginals of x, FMPE vs
ground truth.

FR3 runs (fr3_6d, fr3_5d): per y*, the seven joint marginals, FMPE vs the exact
ground truth, with the URDF joint limits as dashed lines (no arm drawing). For
fr3_5d the flat density of the free joint 7 is drawn as a dotted line. FR3 figures
are saved as .png and .svg.

--joint7 RUN [RUN ...] (fr3_5d): one more figure, the joint-7 histogram pooled over
all 1000 test poses x 1000 samples for each FMPE run given and for the ground truth,
against the flat density (as plot_fff_posterior.py --joint7 does for FFF / M-FFF).
Non-finite samples are dropped there and counted.

    .venv/bin/python experiments/plot_fmpe_posterior.py kin_fmpe_s0 --index 0 1 2
    .venv/bin/python experiments/plot_fmpe_posterior.py fr3_6d_fmpe_s0 --solver midpoint:16
    .venv/bin/python experiments/plot_fmpe_posterior.py fr3_5d_fmpe_s0 --solver euler:64         --joint7 fr3_5d_fmpe_s0 fr3_5d_fmpe_a0_s0 --device cuda
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


def fr3_marginals_figure(bench, index, x_all, label, title):
    """Seven joint marginals per y*, model samples x_all (len(index), n, 7) vs the exact
    ground truth, joint limits marked; on fr3_5d the flat density of joint 7 as a dotted line.
    Shared with plot_fff_posterior.py."""
    from robot_ik.kinematics import joint_limits
    lo_lim, hi_lim = (np.asarray(a, dtype=np.float64) for a in joint_limits())
    y_star, _ = load_test_conditions(bench)
    gt = ground_truth_posterior(bench, y_star, verbose=False)

    fig, axs = plt.subplots(len(index), 7, figsize=(22, 2.9 * len(index)), squeeze=False)
    for row, i, x in zip(axs, index, x_all):
        out = ((x < lo_lim) | (x > hi_lim)).any(1).mean()
        for d, axd in enumerate(row):
            lo, hi = np.quantile(np.concatenate([x[:, d], gt[i][:, d]]), [0.002, 0.998])
            lo, hi = min(lo, lo_lim[d]), max(hi, hi_lim[d])
            pad = 0.03 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, 60)
            axd.hist(gt[i][:, d], bins, density=True, alpha=0.5, color='gray', label=f'ground truth ({len(gt[i])})')
            axd.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', lw=1.5, label=label)
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
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_fr3(args, model, bench, method, steps):
    """Seven joint marginals per y*, FMPE vs ground truth, joint limits marked."""
    y_star, _ = load_test_conditions(bench)
    torch.manual_seed(0)
    x_all = model.sample(y_star[args.index], 1000, steps=int(steps), method=method).cpu().numpy().astype(np.float64)
    return fr3_marginals_figure(bench, args.index, x_all, 'FMPE',
                                f'FMPE joint marginals vs. exact ground truth, run {args.run}, '
                                f'{method} x {steps} ({nfe(method, int(steps))} NFE)')


def plot_joint7(runs, method, steps, device):
    """fr3_5d: joint 7 pooled over all test poses, per FMPE run and for the ground truth, against
    the flat density; per-pose KS distance from uniform (samples outside the limits moved to the
    nearest limit for the KS test only, as in plot_fff_posterior.py)."""
    from robot_ik.kinematics import joint_limits
    lo, hi = (float(a[6]) for a in joint_limits())
    y_star, _ = load_test_conditions('fr3_5d')
    gt = ground_truth_posterior('fr3_5d', y_star, verbose=False)

    def ks(v):
        u = np.sort((np.clip(v, lo, hi) - lo) / (hi - lo))
        k = np.arange(1, len(u) + 1) / len(u)
        return max((k - u).max(), (u - (k - 1 / len(u))).max())

    sources = []
    for run in runs:
        model, cfg = load_run(run, device)
        if cfg['problem'] != 'fr3_5d':
            raise ValueError(f'{run} is not an fr3_5d run')
        torch.manual_seed(0)
        x7 = model.sample(y_star, 1000, steps=int(steps), method=method).cpu().numpy()[..., 6].astype(np.float64)
        sources.append((f"{run} (alpha = {cfg['alpha']:g})", x7))
    sources.append(('ground truth', np.stack([g[:1000, 6] for g in gt])))

    fig, axs = plt.subplots(1, len(sources), figsize=(6 * len(sources), 3.8), squeeze=False, sharey=True)
    bins = np.linspace(-np.pi, np.pi, 64)
    for ax, (label, x7) in zip(axs[0], sources):
        finite = np.isfinite(x7)
        n_bad = int((~finite).sum())
        ks_m = np.array([ks(v[np.isfinite(v)]) for v in x7])
        v = x7[finite]
        outside = ((v < lo) | (v > hi)).mean()
        ratio = np.histogram(v, np.linspace(lo, hi, 21))[0] / (v.size / 20)
        print(f'{label}: non-finite {n_bad}; KS per pose median {np.median(ks_m):.4f} p95 {np.quantile(ks_m, 0.95):.4f}; '
              f'outside limits {outside:.4%}; max |x7| {np.abs(v).max():.3f}; '
              f'inner bin ratio min {ratio[1:-1].min():.3f} max {ratio[1:-1].max():.3f}; '
              f'edge bins {ratio[0]:.3f} {ratio[-1]:.3f}')
        ax.hist(v, bins, density=True, color='gray' if label == 'ground truth' else 'tab:blue', alpha=0.6, label=label)
        ax.axhline(1 / (hi - lo), color='k', ls=':', lw=1.2, label='uniform within limits')
        for b in (lo, hi):
            ax.axvline(b, color='tab:red', ls='--', lw=1, label='joint limit' if b == lo else None)
        ax.set_title(f'{label}\nKS vs uniform per pose: median {np.median(ks_m):.3f}, '
                     f'outside limits {outside:.2%}', fontsize=8)
        ax.set_xlabel('joint 7 (rad), all 1000 test poses x 1000 samples')
        ax.legend(fontsize=7, loc='lower center')
    fig.suptitle(f'fr3_5d: free joint 7 vs. the flat density, FMPE {method} x {steps} '
                 f'({nfe(method, int(steps))} NFE), test set', fontsize=10)
    fig.tight_layout()
    out = Path(__file__).parent / 'figures' / 'fmpe' / 'posterior' / 'fr3_5d_joint7_fmpe_alpha'
    for ext in ('png', 'svg'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=110)
    plt.close(fig)
    print(f'saved {out}.png/.svg')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--index', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--solver', default='rk4:8')
    p.add_argument('--n-draw', type=int, default=150)
    p.add_argument('--device', default='cpu')
    p.add_argument('--joint7', nargs='+', default=None, metavar='RUN',
                   help='fr3_5d: also draw the pooled joint-7 histogram for these FMPE runs')
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    bench = cfg['problem']
    method, steps = args.solver.split(':')
    out = Path(__file__).parent / 'figures' / 'fmpe' / 'posterior' / f'fmpe_posterior_{args.run}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    if bench.startswith('fr3'):
        fig = plot_fr3(args, model, bench, method, steps)
        for ext in ('png', 'svg'):
            fig.savefig(out.with_suffix(f'.{ext}'), dpi=110)
        plt.close(fig)
        print(f'saved {out} (.png/.svg)')
        if args.joint7:
            plot_joint7(args.joint7, method, steps, args.device)
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
