"""Train the standard autoencoder on one benchmark problem.

    .venv/bin/python experiments/train_autoencoder.py --problem kinematics
    .venv/bin/python experiments/train_autoencoder.py --problem ballistics

Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (hidden width, parameter count)
    log.csv         per-epoch train / val loss terms + provisional val re-simulation error
    summary.json    final losses, parameter count, provisional re-simulation error
    model.pt        state dict

Every default marked TODO(berker) is a placeholder: the paper gives no value
and the team has not agreed on one yet. See docs/autoencoder.md.
Final numbers come from metrics.evaluate, not from this script.
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
from models.autoencoder import Autoencoder, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
TERMS = ('recon', 'y', 'z')


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate_losses(model, x, y, batch_size):
    totals = dict.fromkeys(TERMS, 0.0)
    for s in range(0, len(x), batch_size):
        for k, v in model.losses(x[s:s + batch_size], y[s:s + batch_size]).items():
            totals[k] += v.item() * len(x[s:s + batch_size])
    return {k: v / len(x) for k, v in totals.items()}


@torch.no_grad()
def provisional_resim(model, problem, y, n_conditions=200, n_samples=100):
    """Mean ||f(D(y, z)) - y||^2 over a few validation y, z ~ N(0, I): a cheap
    progress signal. Report metrics.evaluate numbers, not this."""
    y = y[:n_conditions]
    x = model.sample(y, n_samples).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn(problem)(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    # Architecture. TODO(berker): depth / activation untuned.
    p.add_argument('--layers', type=int, default=4, help='hidden layers per network')
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000, help='encoder + decoder together')
    p.add_argument('--activation', default='relu', choices=['relu', 'silu', 'tanh', 'leaky_relu'])
    # Loss weights: L = L_recon + a L_y + b L_z. TODO(berker): sweep a, b.
    p.add_argument('--w-y', type=float, default=1.0, help='a, weight of the supervised y loss')
    p.add_argument('--w-z', type=float, default=100.0, help='b, weight of the latent MMD loss')
    p.add_argument('--latent-mmd', choices=['joint', 'marginal'], default='joint',
                   help='MMD on [y_hat, z] vs [y, eps] (joint) or z vs eps (marginal)')
    # Schedule. TODO(berker): placeholders until the team agrees on shared values (same as MDN).
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    # Optimizer. TODO(berker): untuned, copied from the MDN.
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
    x_dims, y_dims = problem.n_parameters, problem.n_observations

    hidden = args.hidden or width_for_budget(x_dims, args.layers, args.param_budget)
    model = Autoencoder(x_dims, y_dims, hidden, args.layers, args.activation, args.latent_mmd)
    n_params = count_parameters(model)

    run_name = args.run_name or f'{args.problem}_ae_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), 'model': 'autoencoder', 'hidden_resolved': hidden, 'z_dims': model.z_dims,
              'n_params': n_params, 'device_resolved': str(device), 'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: hidden={hidden}, dim z={model.z_dims}, {n_params:,} trainable parameters, device={device}')

    # Same split seeds as the MDN, so both models see identical data.
    t = time.time()
    x_tr, y_tr = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_train, seed=10_000 + args.seed))
    x_va, y_va = (torch.from_numpy(a).to(device) for a in make_dataset(args.problem, args.n_val, seed=20_000 + args.seed))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr)
    model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    weights = {'recon': 1.0, 'y': args.w_y, 'z': args.w_z}

    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss'] + [f'train_{k}' for k in TERMS] + [f'val_{k}' for k in TERMS]
                 + ['val_resim', 'val_resim_failed', 'lr', 'seconds'])
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = dict.fromkeys(TERMS, 0.0)
        running_total = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            terms = model.losses(x_tr[idx], y_tr[idx])
            loss = sum(weights[k] * terms[k] for k in TERMS)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}: { {k: v.item() for k, v in terms.items()} }')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running_total += loss.item(); step += 1
            for k in TERMS:
                running[k] += terms[k].item()
        model.eval()
        train = {k: v / steps_per_epoch for k, v in running.items()}
        val = evaluate_losses(model, x_va, y_va, args.batch_size)
        resim, failed = provisional_resim(model, args.problem, y_va)
        log.writerow([epoch, step, running_total / steps_per_epoch] + [train[k] for k in TERMS]
                     + [val[k] for k in TERMS] + [resim, failed, sched.get_last_lr()[0], time.time() - t0])
        print(f'epoch {epoch:3d}  val recon {val["recon"]:.5f}  y {val["y"]:.5f}  z {val["z"]:+.5f}  '
              f'resim {resim:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    summary = {'run': run_name, 'problem': args.problem, 'model': 'autoencoder', 'n_params': n_params,
               'hidden': hidden, 'layers': args.layers, 'z_dims': model.z_dims,
               'w_y': args.w_y, 'w_z': args.w_z, 'latent_mmd': args.latent_mmd,
               **{f'val_{k}': v for k, v in val.items()},
               'val_resim_provisional': resim, 'val_resim_failed_fraction': failed,
               'train_seconds': time.time() - t0}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
