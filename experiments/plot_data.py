"""Build-order step 1: sanity plots of prior samples and forward process.

    .venv/bin/python experiments/plot_data.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import InverseBallisticsModel, InverseKinematicsModel

OUT = os.path.join(os.path.dirname(__file__), 'figures')


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    km = InverseKinematicsModel()
    x = km.sample_prior(300, rng)
    joints = km.joint_positions(x)
    ax = axs[0]
    for a, b, c in zip(joints[:-1], joints[1:], ['C0', 'C1', 'C2']):
        ax.plot(np.stack([a[:, 0], b[:, 0]]), np.stack([a[:, 1], b[:, 1]]), c=c, alpha=0.08, lw=1)
    ax.scatter(joints[-1][:, 0], joints[-1][:, 1], s=4, c='k', zorder=3, label='end point y')
    ax.set_title(f'inverse kinematics: {len(x)} prior samples'); ax.set_aspect('equal'); ax.legend()

    bm = InverseBallisticsModel()
    x = bm.sample_prior(150, rng)
    xs, ys = bm.trajectories_from_parameters(x)
    y = bm.forward_process(x)[:, 0]
    ax = axs[1]
    for i in range(len(x)):
        above = ys[i] >= -0.5
        ax.plot(xs[i, above], ys[i, above], c='C0', alpha=0.15, lw=1)
    ax.scatter(x[:, 0], x[:, 1], s=6, c='C1', zorder=3, label='launch point (x1, x2)')
    ax.scatter(y, np.zeros_like(y), s=6, c='k', zorder=3, label='impact y')
    ax.axhline(0, c='gray', ls=':')
    ax.set_title(f'inverse ballistics: {len(x)} prior samples'); ax.set_ylim(-0.5, None); ax.legend()

    fig.tight_layout()
    path = os.path.join(OUT, 'data_overview.png')
    fig.savefig(path, dpi=120)
    print('wrote', path)


if __name__ == '__main__':
    main()
