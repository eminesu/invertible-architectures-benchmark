"""Tail stress test for diverging FMPE samples (FR3 follow-up, alpha = 0 control).

On fr3_6d one test sample in ~1e6 diverged with alpha = 1. It started from the tail
of x(0) (|x(0)| = 5.36). A normal test evaluation sees too few such starting points
to compare runs, so this script integrates many x(0) drawn from the tail only:
x(0) ~ N(0, I) conditioned on |x(0)| > --radius (rejection), each paired with a test
pose (cycled over all 1000 test y*), and counts the non-finite end points. Every run
gets the same x(0) and y* (fixed seed), so the counts are paired.

    .venv/bin/python experiments/fmpe_tail_divergence.py fr3_6d_fmpe_s0 fr3_6d_fmpe_a0_s0 \
        --n 200000 --radius 4.5 --solver euler:64 --device cuda

Writes metrics/results/<benchmark>_fmpe_tail_divergence_r<radius>.json. Nothing is
redrawn; this is a diagnostic, not a benchmark score.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.evaluate_fmpe import load_run
from metrics.benchmarks import load_test_conditions
from metrics.evaluate import RESULTS_DIR


def tail_draws(n, dims, radius, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    kept, total = [], 0
    while sum(len(k) for k in kept) < n:
        z = torch.randn(4_000_000, dims, device=device, generator=g)
        total += len(z)
        kept.append(z[z.norm(dim=1) > radius])
    return torch.cat(kept)[:n], total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--n', type=int, default=200_000, help='tail starting points per run')
    p.add_argument('--radius', type=float, default=4.5)
    p.add_argument('--solver', default='euler:64')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--chunk', type=int, default=100_000)
    args = p.parse_args()
    method, steps = args.solver.split(':')
    steps = int(steps)

    models = [load_run(run, args.device) for run in args.runs]
    bench = models[0][1]['problem']
    assert all(cfg['problem'] == bench for _, cfg in models), 'all runs must be on one benchmark'
    y_star, _ = load_test_conditions(bench)
    dims = models[0][0].x_dims
    x0, drawn = tail_draws(args.n, dims, args.radius, args.seed, args.device)
    idx = torch.arange(args.n, device=args.device) % len(y_star)
    y = torch.as_tensor(np.asarray(y_star, dtype=np.float32), device=args.device)[idx]
    norms = x0.norm(dim=1)

    out = dict(benchmark=bench, solver=args.solver, radius=args.radius, n=args.n, seed=args.seed,
               draws_needed=drawn, tail_probability=args.n / drawn,
               x0_norm={'median': float(norms.median()), 'max': float(norms.max())}, runs={})
    for run, (model, cfg) in zip(args.runs, models):
        bad = torch.zeros(args.n, dtype=torch.bool, device=args.device)
        big = torch.zeros(args.n, dtype=torch.bool, device=args.device)
        for s in range(0, args.n, args.chunk):
            y_n = (y[s:s + args.chunk] - model.y_mean) / model.y_std
            x = model.integrate(x0[s:s + args.chunk].clone(), y_n, steps, method)
            x = x * model.x_std + model.x_mean
            bad[s:s + args.chunk] = ~torch.isfinite(x).all(1)
            big[s:s + args.chunk] = torch.isfinite(x).all(1) & (x.abs() > 10).any(1)
        n_bad = int(bad.sum())
        out['runs'][run] = dict(alpha=cfg['alpha'], n_nonfinite=n_bad, rate=n_bad / args.n,
                                n_finite_but_abs_gt_10rad=int(big.sum()),
                                nonfinite_x0_norms=sorted(float(v) for v in norms[bad].tolist())[:50],
                                nonfinite_pose_indices=sorted(set(int(v) for v in idx[bad].tolist()))[:50])
        print(f'{run} (alpha={cfg["alpha"]:g}): non-finite {n_bad} of {args.n} '
              f'({n_bad / args.n:.2e}); finite but |x| > 10 rad: {int(big.sum())}', flush=True)
    path = RESULTS_DIR / f'{bench}_fmpe_tail_divergence_r{args.radius:g}.json'
    path.write_text(json.dumps(out, indent=2))
    print(f'saved {path}')


if __name__ == '__main__':
    main()
