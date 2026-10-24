"""Build-order step 4: sanity check of a trained kinematics autoencoder.

    .venv/bin/python experiments/plot_autoencoder_sanity.py kin_ae_sanity_s0

Top row: for a few fixed y*, decode 1000 z ~ N(0, I) and draw the arms. End
points must sit on y* (red cross) and the arms must spread over the solution
set, not collapse to one pose. The rightmost top panel overlays the shared
ground-truth posterior (rejection sampling, metrics/ground_truth.py) for
test condition 0.

Bottom row: diagnostics of the two "soft" parts of the model on held-out data:
the encoder's z against the N(0, I) prior it is trained to match (a mismatch
means test-time sampling is invalid), and the per-coordinate reconstruction
error D(E(x)) - x (soft invertibility is never exact).
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
from data.toy_data import InverseKinematicsModel, make_dataset
from metrics.benchmarks import load_test_conditions
from metrics.ground_truth import ground_truth_posterior
from models.autoencoder import Autoencoder

HERE = os.path.dirname(__file__)
TARGETS = np.array([[1.5, 0.0], [1.0, 1.2], [0.6, -1.5]], dtype=np.float32)  # same as plot_posterior.py


def load(run):
    cfg = json.load(open(os.path.join(HERE, 'runs', run, 'config.json')))
    assert cfg['problem'] == 'kinematics', 'only kinematics is plotted here'
    model = Autoencoder(4, 2, cfg['hidden_resolved'], cfg['layers'], cfg['activation'], cfg['latent_mmd'])
    model.load_state_dict(torch.load(os.path.join(HERE, 'runs', run, 'model.pt'), map_location='cpu'))
    return model.eval()


def draw_arms(ax, km, xs, y_star, gt=None):
    joints = km.joint_positions(xs.astype(np.float64))
    for a, b, c in zip(joints[:-1], joints[1:], ['C0', 'C1', 'C2']):
        ax.plot(np.stack([a[:, 0], b[:, 0]]), np.stack([a[:, 1], b[:, 1]]), c=c, alpha=0.03, lw=1)
    ax.scatter(*joints[-1].T, s=1, c='k', alpha=0.3)
    if gt is not None:  # ground-truth elbow / wrist positions for comparison
        gj = km.joint_positions(gt.astype(np.float64))
        ax.scatter(*gj[2].T, s=1, c='m', alpha=0.25, label='ground truth, joint 3')
        ax.legend(loc='lower left', fontsize=7, markerscale=6)
    ax.plot(*y_star, '+', c='r', ms=14, mew=2)
    resim = np.mean(np.sum((joints[-1] - y_star) ** 2, axis=1))
    ax.set_title(f'y* = ({y_star[0]:.2f}, {y_star[1]:.2f})   resim {resim:.4f}', fontsize=10)
    ax.set_xlim(-0.35, 2.25); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal')


def main(run):
    torch.manual_seed(0)
    model = load(run)
    km = InverseKinematicsModel()

    y_test, _ = load_test_conditions('kinematics')
    y_gt = y_test[:1]
    # The cache is keyed on the full y* set, so load all of it and take entry 0.
    gt = ground_truth_posterior('kinematics', y_test, verbose=False)[0]
    targets = np.concatenate([TARGETS, y_gt])
    x = model.sample(targets, 1000).numpy()

    fig, axs = plt.subplots(2, 4, figsize=(17, 8.8))
    for i, (ax, y_star, xs) in enumerate(zip(axs[0], targets, x)):
        draw_arms(ax, km, xs, y_star, gt if i == len(targets) - 1 else None)

    # Held-out data, never seen in training (seed outside the train/val/test seeds).
    x_ho, y_ho = (torch.from_numpy(a) for a in make_dataset('kinematics', 20_000, seed=99_999))
    with torch.no_grad():
        x_n = (x_ho - model.x_mean) / model.x_std
        y_hat, z = model.encode(x_n)
        x_rec = model.decode(y_hat, z) * model.x_std + model.x_mean
    z = z.numpy()
    grid = np.linspace(-4, 4, 200)
    for j in range(z.shape[1]):
        ax = axs[1, j]
        ax.hist(z[:, j], bins=80, range=(-4, 4), density=True, alpha=0.6, label=f'encoder z{j + 1}')
        ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), 'k', lw=1, label='N(0, 1)')
        ax.set_title(f'z{j + 1}: mean {z[:, j].mean():+.3f}, std {z[:, j].std():.3f}', fontsize=10)
        ax.legend(fontsize=8)
    ax = axs[1, 2]
    ax.scatter(z[:5000, 0], z[:5000, 1], s=1, alpha=0.3)
    ax.set_title(f'z1 vs z2 (corr {np.corrcoef(z[:, 0], z[:, 1])[0, 1]:+.3f})', fontsize=10)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax = axs[1, 3]
    err = (x_rec - x_ho).numpy()
    ax.boxplot([err[:, j] for j in range(4)], showfliers=False)
    ax.set_xticks([1, 2, 3, 4], ['x1 (offset)', 'x2', 'x3', 'x4'])
    ax.axhline(0, c='k', lw=0.5)
    ax.set_title(f'reconstruction error D(E(x)) - x, RMS {np.sqrt((err ** 2).mean()):.4f}', fontsize=10)

    fig.suptitle(f'Autoencoder sanity check, run {run}: decoder samples (top), encoder diagnostics on held-out x (bottom)')
    fig.tight_layout()
    path = os.path.join(HERE, 'figures', f'autoencoder_sanity_{run}.png')
    fig.savefig(path, dpi=110)
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1])
