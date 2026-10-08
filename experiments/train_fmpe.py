"""Train FMPE (flow matching posterior estimation) on one benchmark problem.

    .venv/bin/python experiments/train_fmpe.py --problem kinematics --device mps
    .venv/bin/python experiments/train_fmpe.py --problem ballistics --device mps

Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (hidden width, parameter count)
    log.csv         per-epoch train / val flow-matching loss
    summary.json    final losses, parameter count, provisional re-simulation error
    model.pt        state dict

Data, schedule and optimizer match train_mdn.py (1M train samples, 50 epochs,
batch 1000, Adam + cosine, ~3M parameters), so the comparison differs only in
the model. Those values are still team placeholders (docs/mdn.md). Final
numbers come from experiments/evaluate_fmpe.py (metrics.evaluate), not from this script.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS, make_dataset
from metrics.benchmarks import forward_fn
from models.fmpe import ACTIVATIONS, FMPE, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate_loss(model, x, y, noise, batch_size):
    total = 0.0
    for s in range(0, len(x), batch_size):
        sl = slice(s, s + batch_size)
        total += model.loss(x[sl], y[sl], noise=(noise[0][sl], noise[1][sl])).item() * len(x[sl])
    return total / len(x)


@torch.no_grad()
def provisional_resim(model, problem, y, n_conditions=200, n_samples=100, steps=16):
    """Mean ||f(x) - y||^2 over a few validation y: a cheap progress signal.
    Report metrics.evaluate numbers, not this."""
    y = y[:n_conditions]
    x = model.sample(y, n_samples, steps=steps).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn(problem)(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    p.add_argument('--blocks', type=int, default=6, help='gated residual blocks (2 linear layers each)')
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--activation', default='silu', choices=list(ACTIVATIONS))
    p.add_argument('--sigma-min', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.0, help='time prior p(t) ∝ t^alpha; 0 = uniform')
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--run-name', default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    y_dims, x_dims = problem.n_observations, problem.n_parameters

    hidden = args.hidden or width_for_budget(x_dims, y_dims, args.blocks, args.param_budget)
    model = FMPE(x_dims, y_dims, hidden, args.blocks, args.activation, args.sigma_min, args.alpha)
    n_params = count_parameters(model)

    run_name = args.run_name or f'{args.problem}_fmpe_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), 'hidden_resolved': hidden, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: hidden={hidden}, {n_params:,} trainable parameters, device={device}')

    # Same seeds per split as train_mdn.py, so every model sees the same data.
    t = time.time()
    x_tr, y_tr = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_train, seed=10_000 + args.seed))
    x_va, y_va = (torch.from_numpy(a).to(device) for a in make_dataset(args.problem, args.n_val, seed=20_000 + args.seed))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr)
    model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    # Fixed (x0, t) for the validation loss, so epochs are comparable.
    g = torch.Generator().manual_seed(1)
    val_noise = (torch.randn(args.n_val, x_dims, generator=g).to(device),
                 (torch.rand(args.n_val, 1, generator=g) ** (1 / (1 + args.alpha))).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'seconds'])
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = model.loss(x_tr[idx], y_tr[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running += loss.item(); step += 1
        model.eval()
        val = evaluate_loss(model, x_va, y_va, val_noise, 10_000)
        train = running / steps_per_epoch
        log.writerow([epoch, step, train, val, sched.get_last_lr()[0], time.time() - t0])
        print(f'epoch {epoch:3d}  train {train:.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))

    resim, failed = provisional_resim(model, args.problem, y_va)
    summary = {
        'run': run_name, 'problem': args.problem, 'n_params': n_params, 'hidden': hidden, 'blocks': args.blocks,
        'train_loss': train, 'val_loss': val,
        'err_resim_provisional': resim, 'resim_failed_fraction': failed,
        'provisional_setup': '200 val y x 100 samples, rk4 16 steps (64 NFE)',
        'train_seconds': time.time() - t0,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
