"""Train the independent, fixed-prior conditional FIF benchmark model."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.toy_data import MODELS
from metrics.benchmarks import forward_fn
from metrics.ground_truth import GT_SIGMA
from models.fif import FIF, width_for_budget


def dataset(problem, size, seed):
    """Fixed benchmark prior, retaining only draws with a defined observation."""
    rng = np.random.RandomState(seed)
    simulator, forward = MODELS[problem](), forward_fn(problem)
    xs, ys, remaining, rejected = [], [], size, 0
    while remaining:
        x = simulator.sample_prior(remaining, rng)
        y = forward(x)
        valid = np.isfinite(y).all(1)
        rejected += int((~valid).sum())
        xs.append(x[valid])
        ys.append(y[valid])
        remaining -= int(valid.sum())
    return (torch.from_numpy(np.concatenate(xs).astype(np.float32)),
            torch.from_numpy(np.concatenate(ys).astype(np.float32)), rejected)


def validation(model, x, y, batch_size, problem):
    model.eval()
    totals = dict.fromkeys(('loss', 'recon', 'latent', 'trace'), 0.0)
    device = model.x_mean.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(991)
        for start in range(0, len(x), batch_size):
            batch_x, batch_y = x[start:start + batch_size], y[start:start + batch_size]
            terms = model.loss_terms(batch_x.to(device), batch_y.to(device))
            for key in totals:
                totals[key] += float(terms[key].detach()) * len(batch_x) / len(x)
        observations = y[:64].numpy()
        generator = torch.Generator(device=device).manual_seed(991)
        draws = model.sample(observations, 32, generator=generator).cpu().numpy()
    simulated = forward_fn(problem)(draws.reshape(-1, 4)).reshape(len(observations), 32, -1)
    error = ((simulated - observations[:, None, :])**2).sum(-1)
    totals['resim'] = float(np.where(np.isfinite(error), error, 100.0).mean())
    totals['failed'] = float((~np.isfinite(error)).mean())
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=tuple(MODELS), required=True)
    parser.add_argument('--run', help='Short folder name; default: kin_fif_s0 or bal_fif_s0')
    parser.add_argument('--out', type=Path, help='Explicit output folder, useful for small checks')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--n-train', type=int, default=1_000_000)
    parser.add_argument('--n-val', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--param-budget', type=int, default=3_000_000)
    parser.add_argument('--beta', type=float, default=100.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, help='Raw observation-noise std; default: shared GT sigma')
    parser.add_argument('--kde-points', type=int, default=1024)
    args = parser.parse_args(argv)
    if min(args.n_train, args.n_val, args.batch_size) < 2 or min(args.epochs, args.threads) < 1:
        parser.error('Need at least 2 data rows/batch and positive epochs/threads')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if args.hidden is not None and args.hidden < 1:
        parser.error('--hidden must be positive')
    name = args.run or f'{"kin" if args.problem == "kinematics" else "bal"}_fif_s{args.seed}'
    if name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name:
        parser.error('--run must be a folder name, without path components')
    folder = args.out or ROOT / 'experiments' / 'runs' / name
    if folder.exists():
        parser.error(f'Output already exists: {folder}; choose a new folder')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    y_dims = MODELS[args.problem].n_observations
    hidden = args.hidden or width_for_budget(y_dims, 4 - y_dims, args.param_budget, args.layers)
    settings = dict(x_dims=4, y_dims=y_dims, z_dims=4 - y_dims, hidden=hidden,
                    layers=args.layers, beta=args.beta,
                    sigma=GT_SIGMA[args.problem] if args.sigma is None else args.sigma,
                    kde_points=args.kde_points, discrete_speed=args.problem == 'ballistics')
    model = FIF(**settings)
    count = sum(p.numel() for p in model.parameters())
    if count > args.param_budget:
        parser.error(f'Model has {count:,} parameters, above --param-budget')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(f'{name} | device={device} | parameters={count:,} | '
          f'train={args.n_train:,} | validation={args.n_val:,}', flush=True)
    print('Preparing training data...', flush=True)
    x, y, rejected = dataset(args.problem, args.n_train, 10000 + args.seed)
    print('Preparing validation data...', flush=True)
    vx, vy, val_rejected = dataset(args.problem, args.n_val, 20000 + args.seed)
    print('Fitting normalization and observation KDE...', flush=True)
    model.fit(x, y, seed=args.seed)
    model.to(device)
    config = {**vars(args), 'out': str(folder), 'run': name, 'method': 'FIF', 'model': settings,
              'prior_mode': 'fixed', 'n_params': count,
              'data_seeds': [10000 + args.seed, 20000 + args.seed],
              'rejected_prior_draws': {'train': rejected, 'validation': val_rejected},
              'device_used': str(device), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'selection': 'validation resimulation, penalty 100 for invalid draws'}
    (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = (len(x) + args.batch_size - 1) // args.batch_size
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps)
    best, best_epoch = float('inf'), 0
    with (folder / 'log.csv').open('w', newline='', encoding='utf-8') as log:
        writer = None
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            print(f'Epoch {epoch}/{args.epochs} started | {steps} batches', flush=True)
            model.train()
            totals = dict.fromkeys(('loss', 'recon', 'latent', 'trace'), 0.0)
            order = torch.randperm(len(x))
            for batch, indices in enumerate(order.split(args.batch_size), start=1):
                optimizer.zero_grad(set_to_none=True)
                terms = model.loss_terms(x[indices].to(device), y[indices].to(device))
                if not all(torch.isfinite(value) for value in terms.values()):
                    raise RuntimeError(f'Non-finite training terms at epoch {epoch}')
                terms['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                for key in totals:
                    totals[key] += float(terms[key].detach()) * len(indices) / len(x)
                if batch == 1 or batch % 100 == 0 or batch == steps:
                    print(f'  batch {batch}/{steps} | loss={float(terms["loss"].detach()):.6g} | '
                          f'recon={float(terms["recon"].detach()):.6g}', flush=True)
            print(f'  Validating epoch {epoch}...', flush=True)
            valid = validation(model, vx, vy, args.batch_size, args.problem)
            if not all(np.isfinite(value) for value in valid.values()):
                raise RuntimeError(f'Non-finite validation terms at epoch {epoch}')
            improved = valid['resim'] < best
            if improved:
                best, best_epoch = valid['resim'], epoch
                torch.save(model.state_dict(), folder / 'model.pt')
            row = {'epoch': epoch, **{f'train_{k}': v for k, v in totals.items()},
                   **{f'val_{k}': v for k, v in valid.items()}, 'lr': schedule.get_last_lr()[0]}
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log.flush()
            print(f'Epoch {epoch}/{args.epochs} complete | '
                  f'train loss={totals["loss"]:.6g} recon={totals["recon"]:.6g} | '
                  f'val loss={valid["loss"]:.6g} recon={valid["recon"]:.6g} '
                  f'resim={valid["resim"]:.6g} failed={valid["failed"]:.2%} | '
                  f'lr={schedule.get_last_lr()[0]:.3g} | '
                  f'{time.perf_counter() - epoch_started:.1f}s'
                  + (' | saved best model.pt' if improved else ''), flush=True)
    torch.save(model.state_dict(), folder / 'last.pt')
    summary = {'run': name, 'problem': args.problem, 'prior_mode': 'fixed',
               'best_epoch': best_epoch, 'validation_resim': best, 'n_params': count,
               'seconds': time.perf_counter() - started}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Training complete | best epoch={best_epoch} | outputs: {folder}', flush=True)
    return folder


if __name__ == '__main__':
    main()
