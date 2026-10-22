"""Train the prior-conditioned FMPE (models/conditional_fmpe.py) on ballistics.

    .venv/bin/python experiments/train_conditional_fmpe.py --device mps --run-name bal_cfmpe_s0

Training data are (x, y, lambda) triples with lambda ~ U[--lam-min, --lam-max],
v0 ~ Poisson(lambda) (data/toy_data.py make_conditional_dataset). Everything
else matches experiments/train_fmpe.py and the plain-FMPE headline run
bal_fmpe_s0: 1M training samples, 50 epochs, batch 1000, Adam 1e-3 + cosine,
weight decay 1e-5, grad clip 10, 6 gated blocks at ~3M parameters, time prior
p(t) ∝ t (alpha = 1). Outputs in experiments/runs/<run-name>/ as train_fmpe.py
(config.json, log.csv, summary.json, model.pt), plus checkpoint.pt after every
epoch: rerunning the same command after an interruption resumes from the last
finished epoch (same data, regenerated from the seeds). Final numbers come from
experiments/evaluate_conditional_fmpe.py.
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
from data.toy_data import LAM_RANGE, InverseBallisticsModel, make_conditional_dataset
from experiments.train_fmpe import RUNS, pick_device
from metrics.benchmarks import forward_fn
from models.conditional_fmpe import ConditionalFMPE
from models.fmpe import ACTIVATIONS, count_parameters, width_for_budget


@torch.no_grad()
def evaluate_loss(model, x, y, lam, noise, batch_size):
    total = 0.0
    for s in range(0, len(x), batch_size):
        sl = slice(s, s + batch_size)
        total += model.loss(x[sl], y[sl], lam[sl], noise=(noise[0][sl], noise[1][sl])).item() * len(x[sl])
    return total / len(x)


@torch.no_grad()
def provisional_resim(model, y, lam, n_conditions=200, n_samples=100, steps=16):
    """Mean ||f(x) - y||^2 over a few validation (y, lambda): a cheap progress
    signal. Report metrics.evaluate numbers, not this."""
    y, lam = y[:n_conditions], lam[:n_conditions]
    x = model.sample(y, lam, n_samples, steps=steps).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn('ballistics')(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lam-min', type=float, default=LAM_RANGE[0])
    p.add_argument('--lam-max', type=float, default=LAM_RANGE[1])
    p.add_argument('--blocks', type=int, default=6, help='gated residual blocks (2 linear layers each)')
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--activation', default='silu', choices=list(ACTIVATIONS))
    p.add_argument('--sigma-min', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='time prior p(t) ∝ t^alpha; 1 = the plain-FMPE headline bal_fmpe_s0 (docs/fmpe.md)')
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
    assert args.lam_min <= InverseBallisticsModel.default_lam <= args.lam_max, 'lambda = 15 must be in the range'

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = InverseBallisticsModel()
    y_dims, x_dims = problem.n_observations, problem.n_parameters

    hidden = args.hidden or width_for_budget(x_dims, y_dims + 1, args.blocks, args.param_budget)
    model = ConditionalFMPE(x_dims, y_dims, hidden, args.blocks, args.activation, args.sigma_min, args.alpha)
    n_params = count_parameters(model)

    run_name = args.run_name or f'bal_cfmpe_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    ckpt_path = os.path.join(out, 'checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False) if os.path.exists(ckpt_path) else None
    if ckpt is not None:
        assert ckpt['args'] == vars(args), 'checkpoint was written with different arguments'
    config = {**vars(args), 'problem': 'ballistics', 'hidden_resolved': hidden, 'n_params': n_params,
              'device_resolved': str(device), 'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: hidden={hidden}, {n_params:,} trainable parameters, device={device}')

    # Same seeds per split as train_fmpe.py (the draws differ: lambda comes first).
    lam_range = (args.lam_min, args.lam_max)
    t = time.time()
    x_tr, y_tr, l_tr = (torch.from_numpy(a) for a in make_conditional_dataset(args.n_train, 10_000 + args.seed, lam_range))
    x_va, y_va, l_va = (torch.from_numpy(a).to(device)
                        for a in make_conditional_dataset(args.n_val, 20_000 + args.seed, lam_range))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr, l_tr)
    model.to(device)
    x_tr, y_tr, l_tr = x_tr.to(device), y_tr.to(device), l_tr.to(device)
    # Fixed (x0, t) for the validation loss, so epochs are comparable.
    g = torch.Generator().manual_seed(1)
    val_noise = (torch.randn(args.n_val, x_dims, generator=g).to(device),
                 (torch.rand(args.n_val, 1, generator=g) ** (1 / (1 + args.alpha))).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    log_file = open(os.path.join(out, 'log.csv'), 'a' if ckpt else 'w', newline='')
    log = csv.writer(log_file)
    step, start, t0 = 0, 0, time.time()
    if ckpt is None:
        log.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'lr', 'seconds'])
    else:
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt']); sched.load_state_dict(ckpt['sched'])
        torch.set_rng_state(ckpt['rng_cpu'])
        if device.type == 'mps':
            torch.mps.set_rng_state(ckpt['rng_device'])
        elif device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng_device'])
        step, start, t0 = ckpt['step'], ckpt['epoch'] + 1, time.time() - ckpt['seconds']
        train, val = ckpt['train_loss'], ckpt['val_loss']
        print(f'resumed from {ckpt_path} after epoch {ckpt["epoch"]}')
    for epoch in range(start, args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = model.loss(x_tr[idx], y_tr[idx], l_tr[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running += loss.item(); step += 1
        model.eval()
        val = evaluate_loss(model, x_va, y_va, l_va, val_noise, 10_000)
        train = running / steps_per_epoch
        log.writerow([epoch, step, train, val, sched.get_last_lr()[0], time.time() - t0])
        log_file.flush()
        rng_device = {'mps': lambda: torch.mps.get_rng_state(),
                      'cuda': lambda: torch.cuda.get_rng_state()}.get(device.type, lambda: None)()
        torch.save({'args': vars(args), 'epoch': epoch, 'step': step, 'seconds': time.time() - t0,
                    'train_loss': train, 'val_loss': val,
                    'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng_cpu': torch.get_rng_state(), 'rng_device': rng_device}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)  # atomic: a crash mid-save keeps the previous checkpoint
        print(f'epoch {epoch:3d}  train {train:.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))

    resim, failed = provisional_resim(model, y_va, l_va)
    summary = {
        'run': run_name, 'problem': 'ballistics', 'n_params': n_params, 'hidden': hidden, 'blocks': args.blocks,
        'lam_range': list(lam_range), 'train_loss': train, 'val_loss': val,
        'err_resim_provisional': resim, 'resim_failed_fraction': failed,
        'provisional_setup': '200 val (y, lambda) x 100 samples, rk4 16 steps (64 NFE)',
        'train_seconds': time.time() - t0,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
