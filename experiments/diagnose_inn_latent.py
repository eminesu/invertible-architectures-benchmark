"""Why does an INN / cINN produce re-simulation outliers? Two quick checks on a trained run.

    python experiments/diagnose_inn_latent.py kin_inn_l2mmd_s0
    python experiments/diagnose_inn_latent.py kin_inn_ml_s0          # compare with a healthy model

1. Latent statistics on held-out data: the network maps real x to z. If training worked, z should look like
   N(0, I): per-dimension mean 0 / std 1, mean ||z||^2 equal to dim(z), and about 0.27% of entries beyond 3.
   A std well below 1 means z is too narrow, so a N(0, I) draw at test time lands where the inverse has
   never been trained; a heavy tail means rare large values.
2. Where the bad samples come from: draw z ~ N(0, I) for 300 held-out y* (200 draws each), re-simulate, and
   compare ||z|| of the worst re-simulation errors with ||z|| of all draws. If the worst samples have a much
   larger ||z||, the outliers come from the tail of the latent prior.

Takes seconds. Uses experiments/train_inn.py's data seeds (validation seed 20000 + seed).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import make_dataset
from experiments.evaluate_inn import load_run
from metrics.benchmarks import forward_fn


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-y', type=int, default=300)
    p.add_argument('--n-draws', type=int, default=200)
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    prob = cfg['problem']
    x, y = (torch.from_numpy(a).to(args.device) for a in make_dataset(prob, 20_000, seed=20_000 + cfg['seed']))
    torch.manual_seed(0)

    with torch.no_grad():
        z = model(x, y)[0] if cfg['model'] == 'cinn' else model(x)[1]
    z = z.cpu().numpy()
    d = z.shape[1]
    print(f"== 1. latent statistics of real held-out x  ({args.run}, dim z = {d}, {len(z)} samples)")
    print('   per-dim mean          :', np.round(z.mean(0), 3), ' (want 0)')
    print('   per-dim std           :', np.round(z.std(0), 3), ' (want 1)')
    print('   per-dim max |z|       :', np.round(np.abs(z).max(0), 2), f' (N(0,1) gives about {np.sqrt(2*np.log(len(z))):.1f})')
    print(f"   fraction |z| > 3      : {(np.abs(z) > 3).mean():.4f}  (N(0,1): 0.0027)")
    print(f"   mean ||z||^2          : {(z ** 2).sum(1).mean():.3f}  (want {d})")
    print(f"   99.9th pct ||z||^2    : {np.percentile((z ** 2).sum(1), 99.9):.1f}")

    ys = y[:args.n_y]
    y_n = (ys - model.y_mean) / model.y_std
    y_rep = y_n.repeat_interleave(args.n_draws, 0)
    z_draw = torch.randn(len(y_rep), d, device=args.device)
    with torch.no_grad():
        x_n = model._decode(y_rep, z_draw)
    x_s = (x_n * model.x_std + model.x_mean).cpu().numpy()
    f = forward_fn(prob)(x_s)
    ok = np.isfinite(f).all(1)
    err = ((f - ys.cpu().numpy().repeat(args.n_draws, 0)) ** 2).sum(1)
    znorm = np.linalg.norm(z_draw.cpu().numpy(), axis=1)
    err, znorm, xs = err[ok], znorm[ok], np.abs(x_n.cpu().numpy())[ok].max(1)
    print(f"\n== 2. re-simulation error of {len(err)} draws z ~ N(0, I)")
    print('   percentiles 50 / 90 / 99 / 99.9 / max :', ' / '.join(f'{np.percentile(err, q):.2e}' for q in (50, 90, 99, 99.9)), f'/ {err.max():.2e}')
    print(f"   fraction with error > 1               : {(err > 1).mean():.5f}")
    k = max(1, len(err) // 1000)
    worst = np.argsort(err)[-k:]
    print(f"   mean ||z|| of all draws               : {znorm.mean():.2f}")
    print(f"   mean ||z|| of the worst {k} draws        : {znorm[worst].mean():.2f}")
    print(f"   largest standardised |x| produced     : {xs.max():.1f}   (median over draws {np.median(xs):.1f})")


if __name__ == '__main__':
    main()
