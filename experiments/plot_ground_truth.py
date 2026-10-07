"""Sanity plot: ground-truth posterior samples for one test condition y*.

Kinematics: draws the arms of the accepted x; their end points should cluster on y*.
Ballistics: draws the trajectories; their impacts should cluster on y*.

    .venv/bin/python experiments/plot_ground_truth.py kinematics --index 0
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics.benchmarks import forward_fn, get_model, load_test_conditions
from metrics.ground_truth import ground_truth_posterior

p = argparse.ArgumentParser()
p.add_argument('benchmark', choices=['kinematics', 'ballistics'])
p.add_argument('--index', type=int, default=0)
p.add_argument('--n-draw', type=int, default=150)
args = p.parse_args()

y_star, _ = load_test_conditions(args.benchmark)
y = y_star[args.index]
x = ground_truth_posterior(args.benchmark, y_star)[args.index]
model = get_model(args.benchmark)
resim = forward_fn(args.benchmark)(x)
err = ((resim - y) ** 2).sum(1)

fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})
draw = x[:args.n_draw]
if args.benchmark == 'kinematics':
    joints = np.stack(model.joint_positions(draw), axis=1)  # (n, 4, 2)
    for arm in joints:
        ax.plot(arm[:, 0], arm[:, 1], color='tab:blue', alpha=0.15, lw=1)
    ax.scatter(resim[:, 0], resim[:, 1], s=2, color='tab:orange', label='f(x) of all accepted x')
    ax.scatter(*y, marker='+', s=200, color='k', label='y*')
    ax.set_aspect('equal')
else:
    xs, ys = model.trajectories_from_parameters(draw)
    for xt, yt in zip(xs, ys):
        keep = yt > -0.5
        ax.plot(xt[keep], yt[keep], color='tab:blue', alpha=0.15, lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.scatter(resim[:, 0], np.zeros(len(resim)), s=2, color='tab:orange', label='f(x) of all accepted x')
    ax.scatter(y[0], 0, marker='+', s=200, color='k', label='y*')
ax.legend(loc='best', fontsize=8)
ax.set_title(f'{args.benchmark}: {len(x)} ground-truth samples, y* #{args.index}')

ax_h.hist(np.sqrt(err), bins=40)
ax_h.set_xlabel('||f(x) - y*||')
ax_h.set_title(f'mean squared resim error {err.mean():.1e}')
fig.tight_layout()

out = Path(__file__).parent / 'figures' / f'ground_truth_{args.benchmark}_{args.index}.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f'saved {out}')
