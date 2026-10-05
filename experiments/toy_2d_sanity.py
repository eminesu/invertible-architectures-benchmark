"""Build-order step 5: fit the MDN to a fixed 2-D mixture of strongly tilted
Gaussians and check that the learned covariances (incl. off-diagonals) match.

The condition y is a constant, so the MDN reduces to an unconditional GMM.
A diagonal-only bug would show up as correlation ~0 on every component.

    .venv/bin/python experiments/toy_2d_sanity.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.mdn import MDN, build_cholesky

OUT = os.path.join(os.path.dirname(__file__), 'figures')


def rotated_cov(sx, sy, angle):
    R = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return R @ torch.diag(torch.tensor([sx ** 2, sy ** 2])) @ R.T


def main():
    torch.manual_seed(0)
    weights = torch.tensor([0.5, 0.3, 0.2])
    means = torch.tensor([[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]])
    covs = torch.stack([rotated_cov(1.0, 0.15, math.radians(45)),
                        rotated_cov(0.8, 0.1, math.radians(-60)),
                        rotated_cov(0.6, 0.2, math.radians(10))])
    gt = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(weights),
        torch.distributions.MultivariateNormal(means, covs))

    model = MDN(y_dims=1, x_dims=2, n_components=3, hidden=64, n_layers=2)
    x_all = gt.sample((50_000,))
    y_all = torch.zeros(len(x_all), 1)
    # y is constant, so only x gets normalized (y keeps mean 0 / std 1).
    model.x_mean.copy_(x_all.mean(0)); model.x_std.copy_(x_all.std(0))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(3000):
        idx = torch.randint(0, len(x_all), (512,))
        loss = model.loss(x_all[idx], y_all[idx], kind='jensen' if step < 200 else 'exact')
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        model_nll = model.loss(x_all, y_all).item() + model.x_std.log().sum().item()
        gt_nll = -gt.log_prob(x_all).mean().item()
    print(f'NLL model {model_nll:.4f}  vs  ground truth {gt_nll:.4f}')

    # Learned components, mapped back to raw x-space.
    with torch.no_grad():
        log_w, mu, U_entries = model(torch.zeros(1, 1))
        U = build_cholesky(U_entries[0], 2)
        cov_n = torch.linalg.inv(U.transpose(-1, -2) @ U)
        S = torch.diag(model.x_std)
        cov = S @ cov_n @ S
        mu = mu[0] * model.x_std + model.x_mean
        w = log_w[0].exp()

    def corr(c):
        return (c[..., 0, 1] / (c[..., 0, 0] * c[..., 1, 1]).sqrt())

    ok = True
    print('component   w (gt)        mean (gt)                      corr (gt)')
    for j in range(3):
        i = torch.argmin(((mu - means[j]) ** 2).sum(-1)).item()
        print(f'  {j}      {w[i]:.3f} ({weights[j]:.3f})   [{mu[i,0]:+.2f} {mu[i,1]:+.2f}] '
              f'([{means[j,0]:+.2f} {means[j,1]:+.2f}])   {corr(cov[i]):+.3f} ({corr(covs[j]):+.3f})')
        ok &= abs(corr(cov[i]) - corr(covs[j])).item() < 0.05
        ok &= torch.allclose(cov[i], covs[j], atol=0.05)
    print('PASS: off-diagonal structure recovered' if ok else 'FAIL')

    os.makedirs(OUT, exist_ok=True)
    samples = model.sample(torch.zeros(1, 1), n_samples=5000)[0]
    ref = gt.sample((5000,))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    for ax, s, title in zip(axs, [ref, samples], ['ground truth', 'MDN samples']):
        ax.scatter(s[:, 0], s[:, 1], s=1, alpha=0.3)
        ax.set_title(title); ax.set_aspect('equal')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'toy_2d_sanity.png'), dpi=120)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
