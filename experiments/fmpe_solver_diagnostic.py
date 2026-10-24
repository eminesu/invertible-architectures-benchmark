"""Why does RK4 lose to midpoint / Euler at equal NFE? (docs/fmpe.md, Finding 2)

Hypothesis tested: RK4's last stage evaluates the field at exactly t = 1, at the
extrapolated point x + dt k3, where the OT field (E[x1 | x_t] - x) / (1 - t) is
singular; Euler and midpoint never evaluate at t = 1.

Test: the same trained model and RK4 grid, but the time fed to the network is
capped at t_max < 1. Result (docs/fmpe.md): the cap does not help on kinematics
and closes only part of the gap on ballistics, so this is at most part of the
explanation. Prints Err_post / Err_resim (shared metric functions) on the
first --n-conditions test y*, plus the field magnitude just off the posterior samples.

    .venv/bin/python experiments/fmpe_solver_diagnostic.py kin_fmpe_s0
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.evaluate_fmpe import load_run
from metrics.benchmarks import forward_fn, load_test_conditions
from metrics.evaluate import err_post, err_resim
from metrics.ground_truth import ground_truth_posterior


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--n-conditions', type=int, default=100)
    args = p.parse_args()

    model, cfg = load_run(args.run)
    bench = cfg['problem']
    y_star, _ = load_test_conditions(bench)
    y_star = y_star[:args.n_conditions]
    gt = ground_truth_posterior(bench, load_test_conditions(bench)[0], verbose=False)[:args.n_conditions]
    f = forward_fn(bench)
    velocity = model.velocity

    def score(method, steps, t_max=1.0):
        model.velocity = lambda t, x, y: velocity(t.clamp(max=t_max), x, y)
        torch.manual_seed(0)
        x = model.sample(y_star, 1000, steps=steps, method=method).numpy().astype(np.float64)
        model.velocity = velocity
        return err_post(x, gt)[1], err_resim(x, y_star, f)[1]

    print(f'{bench}, first {len(y_star)} test y*, 1000 samples each')
    print(f"{'solver':<14}{'t_max':>10}{'Err_post':>10}{'Err_resim':>11}")
    for method, steps, t_max in [('midpoint', 16, 1.0), ('rk4', 8, 1.0), ('rk4', 8, 1 - 1 / 16),
                                 ('rk4', 8, 0.99), ('rk4', 16, 1.0), ('rk4', 16, 1 - 1 / 32)]:
        post, resim = score(method, steps, t_max)
        print(f'{method + " x " + str(steps):<14}{t_max:>10.4f}{post:>10.4f}{resim:>11.1e}')

    # Field magnitude near t = 1, at points slightly off the learned posterior.
    torch.manual_seed(0)
    y_n = ((torch.as_tensor(y_star, dtype=torch.float32) - model.y_mean) / model.y_std).repeat_interleave(100, 0)
    with torch.no_grad():
        x1 = model.integrate(torch.randn(len(y_n), model.x_dims), y_n, 16, 'midpoint')
        off = x1 + 0.01 * torch.randn_like(x1)
        print('\nmedian |v| at x = posterior sample + 0.01 noise (standardized units):')
        for t in (0.5, 0.9, 0.99, 0.999, 1.0):
            v = velocity(torch.full((len(off), 1), t), off, y_n)
            print(f'  t = {t:<6} {v.norm(dim=1).median().item():10.2f}')


if __name__ == '__main__':
    main()
