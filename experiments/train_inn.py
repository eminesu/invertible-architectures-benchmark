"""Train the INN or the conditional INN on one benchmark problem.

    .venv/bin/python experiments/train_inn.py --model inn   --loss ml    --problem kinematics
    .venv/bin/python experiments/train_inn.py --model inn   --loss l2mmd --problem kinematics
    .venv/bin/python experiments/train_inn.py --model cinn               --problem ballistics

These are the three paper rows "INN", "INN (L2 + MMD)" and "CINN" (Tables 1-2).
Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (subnet width, parameter count)
    log.csv         per-epoch train / val loss terms + provisional val re-simulation error
    summary.json    final losses, parameter count, provisional re-simulation error
    model.pt        state dict

Same data, schedule and optimizer defaults as train_mdn.py / train_autoencoder.py, so
all models see identical training data and the same number of steps. Every default
marked TODO is a placeholder: the paper gives no value. See docs/inn.md.
Final numbers come from experiments/evaluate_inn.py (metrics.evaluate), not from the
provisional re-simulation error. Reused from the repo: data.toy_data, metrics.benchmarks,
metrics.mmd.mmd_torch, models.mdn.count_parameters, train_mdn.pick_device.
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
from experiments.train_mdn import pick_device
from models.inn import CINN, INN, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
V0 = 3  # index of the integer-valued launch speed x4 on ballistics


def build_model(cfg, x_dims, y_dims):
    """Model for a config dict (command-line arguments, or a saved config.json)."""
    common = dict(n_blocks=cfg['blocks'], hidden=cfg['hidden_resolved'], n_layers=cfg['layers'],
                  activation=cfg['activation'], clamp=cfg['clamp'], perm_seed=cfg['seed'])
    if cfg['model'] == 'cinn':
        return CINN(x_dims, y_dims, **common)
    return INN(x_dims, y_dims, loss=cfg['loss'], sigma_y=cfg['sigma_y'], latent_mmd=cfg['latent_mmd'], **common)


def dequantize(x, gen=None):
    """Add U(-1/2, 1/2) to the integer-valued v0 (ballistics). Opt-in, see docs/inn.md."""
    x = x.clone()
    x[:, V0] += torch.rand(len(x), device=x.device, generator=gen) - 0.5
    return x


@torch.no_grad()
def evaluate_losses(model, x, y, batch_size, weights):
    """Mean loss terms on held-out data (the MMD term is a per-batch estimate)."""
    totals, total = {}, 0.0
    for s in range(0, len(x), batch_size):
        n = len(x[s:s + batch_size])
        terms = model.losses(x[s:s + batch_size], y[s:s + batch_size])
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.0) + v.item() * n
        total += sum(weights[k] * terms[k] for k in model.loss_terms).item() * n
    return {k: v / len(x) for k, v in totals.items()}, total / len(x)


@torch.no_grad()
def provisional_resim(model, problem, y, n_conditions=200, n_samples=100):
    """Mean ||f(x) - y||^2 over a few validation y and x ~ model: a cheap progress
    signal. Report metrics.evaluate numbers, not this."""
    y = y[:n_conditions]
    x = model.sample(y, n_samples).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn(problem)(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    p.add_argument('--model', choices=['inn', 'cinn'], required=True)
    p.add_argument('--loss', choices=['ml', 'l2mmd'], default='ml',
                   help='INN only: Eq. (4) maximum likelihood or Eq. (3) L2 + MMD; the cINN always uses Eq. (5)')
    # Architecture. TODO: the paper only fixes ~3M trainable parameters.
    p.add_argument('--blocks', type=int, default=8, help='coupling blocks')
    p.add_argument('--layers', type=int, default=3, help='hidden layers per coupling subnet')
    p.add_argument('--hidden', type=int, default=None, help='subnet width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--activation', default='relu', choices=['relu', 'silu', 'tanh', 'leaky_relu'])
    p.add_argument('--clamp', type=float, default=2.0, help='soft clamp of the coupling log-scales')
    # Loss. TODO: sigma, w_y, w_z are not given in the paper.
    p.add_argument('--sigma-y', type=float, default=0.05,
                   help='Eq. (4): std of y around y_gt, in standardized y units')
    p.add_argument('--w-y', type=float, default=1.0, help='Eq. (3): weight of L2(y)')
    p.add_argument('--w-z', type=float, default=100.0, help='Eq. (3): alpha, weight of the latent MMD')
    p.add_argument('--latent-mmd', choices=['joint', 'marginal'], default='joint',
                   help="MMD on [y_hat, z] vs [y, eps] ('joint', as models/autoencoder.py) or on z vs eps alone ('marginal', literal Eq. 3)")
    p.add_argument('--dequantize-v0', action='store_true',
                   help='ballistics: add U(-.5,.5) to the integer launch speed in training/validation data')
    # Schedule: same placeholders as the MDN and the autoencoder.
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
    if args.dequantize_v0 and args.problem != 'ballistics':
        p.error('--dequantize-v0 only applies to ballistics')

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    x_dims, y_dims = problem.n_parameters, problem.n_observations
    cond_dim = y_dims if args.model == 'cinn' else 0

    hidden = args.hidden or width_for_budget(x_dims, cond_dim, args.blocks, args.layers, args.param_budget)
    cfg = {**vars(args), 'hidden_resolved': hidden}
    model = build_model(cfg, x_dims, y_dims)
    n_params = count_parameters(model)
    weights = {'nll': 1.0, 'y': args.w_y, 'z': args.w_z}

    tag = 'cinn' if args.model == 'cinn' else f'inn_{args.loss}'
    run_name = args.run_name or f'{args.problem}_{tag}_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**cfg, 'z_dims': model.z_dims, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: subnet width={hidden}, dim z={model.z_dims}, {n_params:,} trainable parameters, device={device}')

    # Same split seeds as the MDN and the autoencoder, so all models see identical data.
    t = time.time()
    x_tr, y_tr = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_train, seed=10_000 + args.seed))
    x_va, y_va = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_val, seed=20_000 + args.seed))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr)  # statistics of the raw (not dequantized) training data
    model.to(device)
    x_tr, y_tr, x_va, y_va = x_tr.to(device), y_tr.to(device), x_va.to(device), y_va.to(device)
    if args.dequantize_v0:  # fixed noise for validation so the curves are comparable across epochs
        x_va = dequantize(x_va, torch.Generator(device=device).manual_seed(args.seed))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    terms_logged = sorted({'nll', 'y', 'z'} & set(model.losses(x_va[:8], y_va[:8]).keys()))
    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss', 'val_loss'] + [f'val_{k}' for k in terms_logged]
                 + ['val_resim', 'val_resim_failed', 'lr', 'seconds'])
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x_b = dequantize(x_tr[idx]) if args.dequantize_v0 else x_tr[idx]
            terms = model.losses(x_b, y_tr[idx])
            loss = sum(weights[k] * terms[k] for k in model.loss_terms)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}: { {k: v.item() for k, v in terms.items()} }')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running += loss.item(); step += 1
        model.eval()
        val, val_loss = evaluate_losses(model, x_va, y_va, args.batch_size, weights)
        resim, failed = provisional_resim(model, args.problem, y_va)
        log.writerow([epoch, step, running / steps_per_epoch, val_loss] + [val[k] for k in terms_logged]
                     + [resim, failed, sched.get_last_lr()[0], time.time() - t0])
        print(f'epoch {epoch:3d}  train {running / steps_per_epoch:+.4f}  val {val_loss:+.4f}  '
              + '  '.join(f'{k} {val[k]:+.4f}' for k in terms_logged)
              + f'  resim {resim:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    summary = {'run': run_name, 'problem': args.problem, 'model': tag, 'n_params': n_params,
               'hidden': hidden, 'blocks': args.blocks, 'layers': args.layers, 'z_dims': model.z_dims,
               'val_loss': val_loss, **{f'val_{k}': v for k, v in val.items()},
               'err_resim_provisional': resim, 'resim_failed_fraction': failed,
               'train_seconds': time.time() - t0}
    if args.model == 'cinn':  # raw-space NLL of x | y, comparable with the MDN's test NLL
        x_te, y_te = (torch.from_numpy(a).to(device) for a in make_dataset(args.problem, 20_000, seed=30_000 + args.seed))
        summary['test_nll'] = model.nll_raw(x_te, y_te).mean().item()

    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
