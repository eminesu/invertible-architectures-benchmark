"""FR3 projection figure: the FR3 counterpart of the arm / f(x) panel of the toy posterior plots.

One figure per task with every FR3 model and the ground truth side by side (one column each),
for the same 3 test poses as the posterior marginal figures. Per pose, two rows:

  arms      40 sampled arm configurations (link origins 0-8) seen in the vertical plane
            through the target: horizontal axis = distance along the horizontal direction
            of p* (signed), vertical axis = z. Target flange position p* as a cross. Shows
            which configuration families a model covers. Same limits across a row.
  flange    flange position of all 1000 samples relative to p*, in mm, projected on the base
            x-y plane, with 1 mm and 10 mm circles. Shows the position error the pose metrics
            measure (the toy panels' f(x) points around y*). Axis limits differ per panel
            (p95 of the radial error, at least 12 mm) because the models' errors span three
            orders of magnitude; the circles give the scale.

Samples outside the joint limits are drawn in red in both rows. Samples are drawn with the same
call and seed as the posterior figures (torch.manual_seed(0), then 1000 samples for the 3 poses at
once; FMPE Euler x 64), so they are the same points when --device matches the device those figures
were drawn on (random draws differ between CPU and CUDA).

    .venv/bin/python experiments/plot_fr3_projection.py fr3_6d --device cuda

Output: experiments/figures/shared/projection/fr3_projection_<task>.{png,svg}
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
from experiments import evaluate_fff, evaluate_fmpe
from metrics.benchmarks import load_test_conditions
from metrics.ground_truth import ground_truth_posterior
from robot_ik.kinematics import _chain, joint_limits

# (label, run suffix, kind); runs missing on disk are skipped and reported
SOURCES = (('FMPE alpha = 1 (3M)', 'fmpe_s0', 'fmpe'),
           ('FMPE alpha = 0 (3M)', 'fmpe_a0_s0', 'fmpe'),
           ('FMPE 1M', 'fmpe_p1000000_s0', 'fmpe'),
           ('FMPE 300k', 'fmpe_p300000_s0', 'fmpe'),
           ('FFF (Euclidean)', 'fff_b400_s0', 'fff'),
           ('M-FFF (torus, wrapped)', 'mfff_wrapped_b400_s0', 'fff'))
LINKS = [f'fr3_link{i}' for i in range(9)]
N_ARMS = 40


def link_points(q):
    """Link origins (N, 9, 3) in the base frame for configurations q (N, 7)."""
    q = torch.as_tensor(q, dtype=torch.float64)
    frames = _chain(torch.float64, 'cpu').forward_kinematics(q, end_only=False)
    return np.stack([frames[name].get_matrix()[:, :3, 3].numpy() for name in LINKS], axis=1)


def draw_samples(task, index, device):
    y_star, _ = load_test_conditions(task)
    gt = ground_truth_posterior(task, y_star, verbose=False)
    sources = [('Ground truth', np.stack([gt[i][:1000] for i in index]))]
    skipped = []
    for label, suffix, kind in SOURCES:
        run = f'{task}_{suffix}'
        if not (Path(evaluate_fmpe.RUNS) / run / 'model.pt').exists():
            skipped.append(run)
            continue
        if kind == 'fmpe':
            model, _ = evaluate_fmpe.load_run(run, device)
            torch.manual_seed(0)
            x = model.sample(y_star[index], 1000, steps=64, method='euler')
        else:
            model, _ = evaluate_fff.load_run(run, device)
            torch.manual_seed(0)
            x = model.sample(y_star[index], 1000)
        sources.append((label, x.detach().cpu().numpy().astype(np.float64)))
    return y_star, sources, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=['fr3_6d', 'fr3_5d'])
    p.add_argument('--index', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    y_star, sources, skipped = draw_samples(args.task, args.index, args.device)
    lo, hi = joint_limits()
    n_rows, n_cols = 2 * len(args.index), len(sources)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.55 * n_rows), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, i in enumerate(args.index):
        p_star = np.asarray(y_star[i][:3], dtype=np.float64)
        u = p_star[:2] / max(np.linalg.norm(p_star[:2]), 1e-9)   # horizontal direction of the target
        arm_ax, err_ax = axs[2 * r], axs[2 * r + 1]
        drawn = []
        for label, x_all in sources:
            x = x_all[r]
            x = x[np.isfinite(x).all(1)]
            drawn.append((x, ((x < lo) | (x > hi)).any(1), link_points(x)))
        arms = np.concatenate([pts[:N_ARMS] for _, _, pts in drawn])
        h_all, z_all = arms[..., :2] @ u, arms[..., 2]
        h_lo, h_hi = np.quantile(h_all, [0.005, 0.995])
        z_lo, z_hi = min(0.0, np.quantile(z_all, 0.005)), np.quantile(z_all, 0.995)
        pad = 0.08 * max(h_hi - h_lo, z_hi - z_lo)
        for c, ((label, _), (x, outside, pts)) in enumerate(zip(sources, drawn)):
            # arms in the vertical plane through the target
            ax = arm_ax[c]
            for k in range(min(N_ARMS, len(x))):
                ax.plot(pts[k, :, :2] @ u, pts[k, :, 2], color='tab:red' if outside[k] else 'tab:blue',
                        alpha=0.3, lw=0.9)
            ax.scatter(p_star[:2] @ u, p_star[2], marker='+', s=120, color='k', zorder=4)
            ax.set_xlim(h_lo - pad, h_hi + pad)
            ax.set_ylim(z_lo - pad, z_hi + pad)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'y* #{i}: arms\nz (m)', fontsize=8)
            ax.set_xlabel('along target direction (m)', fontsize=7, labelpad=1)
            # flange position error, x-y plane, mm
            d = (pts[:, -1, :] - p_star) * 1000
            radial = np.linalg.norm(d, axis=1)
            lim = max(12.0, 1.15 * np.quantile(radial, 0.95))
            ax = err_ax[c]
            ax.scatter(d[~outside, 0], d[~outside, 1], s=2, color='tab:blue', alpha=0.5, lw=0)
            ax.scatter(d[outside, 0], d[outside, 1], s=2, color='tab:red', alpha=0.7, lw=0)
            for rad in (1, 10):
                ax.plot(rad * np.cos(theta), rad * np.sin(theta), color='k', lw=0.7,
                        ls='-' if rad == 1 else '--')
            ax.scatter(0, 0, marker='+', s=80, color='k', zorder=4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            ax.set_title(f'median {np.median(radial):.2g} mm, p95 {np.quantile(radial, 0.95):.2g} mm\n'
                         f'outside limits {outside.mean():.1%}', fontsize=7.5)
            ax.set_xlabel('Δx (mm)', fontsize=7, labelpad=1)
            if c == 0:
                ax.set_ylabel(f'y* #{i}: flange − p*\nΔy (mm)', fontsize=8)
    note = (f'{args.task}, test poses {args.index}, 1000 samples per model and pose (FMPE Euler x 64, FFF / M-FFF one '
            f'decoder pass, seed 0). Arm rows: {N_ARMS} sampled configurations in the vertical plane through the '
            "target (horizontal axis: distance along the target's horizontal direction), same limits across a row; "
            '+ = target flange position. Flange rows: all samples, flange position minus target in mm, base x-y plane; solid circle '
            '1 mm, dashed 10 mm; axis limits per panel (p95 of the radial error, at least 12 mm). Red: sample '
            'outside the joint limits.')
    if skipped:
        note += f' Not drawn (no trained run): {", ".join(skipped)}.'
    fig.suptitle(f'{args.task}: samples projected into the workspace, all models', fontsize=13, fontweight='bold')
    fig.text(0.01, 0.003, note, fontsize=8, color='#4c5967', wrap=True, va='bottom')
    fig.tight_layout(rect=(0, 0.035, 1, 0.98), h_pad=0.6, w_pad=0.4)
    out = Path(__file__).parent / 'figures' / 'shared' / 'projection' / f'fr3_projection_{args.task}'
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=110)
    print(f'saved {out}.png/.svg; skipped: {skipped}')


if __name__ == '__main__':
    main()
