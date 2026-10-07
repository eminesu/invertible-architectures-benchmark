"""Plot MDN posterior samples for a few y* of a trained kinematics run.

    .venv/bin/python experiments/plot_posterior.py kin_K16_s0
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import InverseKinematicsModel
from models.mdn import MDN

HERE = os.path.dirname(__file__)


def main(run):
    cfg = json.load(open(os.path.join(HERE, 'runs', run, 'config.json')))
    assert cfg['problem'] == 'kinematics', 'only kinematics is plotted here'
    model = MDN(2, 4, cfg['components'], cfg['hidden_resolved'], cfg['layers'], cfg['activation'])
    model.load_state_dict(torch.load(os.path.join(HERE, 'runs', run, 'model.pt'), map_location='cpu'))
    model.eval()

    km = InverseKinematicsModel()
    targets = np.array([[1.5, 0.0], [1.0, 1.2], [0.6, -1.5], [1.9, 0.4]], dtype=np.float32)
    x = model.sample(torch.from_numpy(targets), n_samples=1000).numpy()

    fig, axs = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4.2))
    for ax, y_star, xs in zip(axs, targets, x):
        joints = km.joint_positions(xs.astype(np.float64))
        for a, b, c in zip(joints[:-1], joints[1:], ['C0', 'C1', 'C2']):
            ax.plot(np.stack([a[:, 0], b[:, 0]]), np.stack([a[:, 1], b[:, 1]]), c=c, alpha=0.03, lw=1)
        ax.scatter(*joints[-1].T, s=1, c='k', alpha=0.3)
        ax.plot(*y_star, '+', c='r', ms=14, mew=2)
        resim = np.mean(np.sum((joints[-1] - y_star) ** 2, axis=1))
        ax.set_title(f'y* = ({y_star[0]:.1f}, {y_star[1]:.1f})   resim {resim:.4f}', fontsize=10)
        ax.set_xlim(-0.35, 2.25); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal')
    fig.suptitle(f'MDN posterior samples, run {run}')
    fig.tight_layout()
    path = os.path.join(HERE, 'figures', f'posterior_{run}.png')
    fig.savefig(path, dpi=110)
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1])
