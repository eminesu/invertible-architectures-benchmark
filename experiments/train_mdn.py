"""Train the full-covariance MDN on one benchmark problem.

    .venv/bin/python experiments/train_mdn.py --problem kinematics --components 16
    .venv/bin/python experiments/train_mdn.py --problem ballistics --components 16

Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (hidden width, parameter count)
    log.csv         per-epoch train / val NLL
    summary.json    final NLL, parameter count, inference timing,
                    provisional re-simulation error
    model.pt        state dict

Defaults for the schedule (n_train, epochs, batch size) are placeholders until
the team agrees on the shared values; see docs/mdn.md.
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
from models.mdn import MDN, count_parameters, n_tril_entries

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def width_for_budget(y_dims, x_dims, n_components, n_layers, target):
    """Largest hidden width whose MDN stays at or under `target` parameters."""
    head_out = n_components * (1 + x_dims + n_tril_entries(x_dims))

    def n_params(h):
        return (y_dims * h + h) + (n_layers - 1) * (h * h + h) + h * head_out + head_out

    h = 1
    while n_params(h + 1) <= target:
        h += 1
    return h


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')  # MPS lacks some ops we use (solve_triangular); CPU is fine at this size


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_nll(model, x, y, batch_size):
    total = 0.0
    for s in range(0, len(x), batch_size):
        total += model.loss(x[s:s + batch_size], y[s:s + batch_size]).item() * len(x[s:s + batch_size])
    # report NLL in raw x-space
    return total / len(x) + model.x_std.log().sum().item()


@torch.no_grad()
def time_inference(model, y_star, n_samples, device, repeats=5):
    """Wall time to draw n_samples posterior samples for each y* (median of repeats)."""
    times = []
    for _ in range(repeats + 1):
        sync(device); t = time.perf_counter()
        for s in range(0, len(y_star), 100):
            model.sample(y_star[s:s + 100], n_samples)
        sync(device); times.append(time.perf_counter() - t)
    return float(np.median(times[1:]))  # first run is warm-up


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    p.add_argument('--components', type=int, default=16, help='number of mixture components K')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--activation', default='relu', choices=['relu', 'silu', 'tanh'])
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--warmup-loss', choices=['jensen', 'eq15', 'none'], default='jensen',
                   help='loss for the first --warmup-steps steps before switching to the exact NLL')
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--n-eval-conditions', type=int, default=1000)
    p.add_argument('--n-eval-samples', type=int, default=4000, help='posterior samples per y* for timing / re-sim')
    p.add_argument('--run-name', default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    y_dims, x_dims = problem.n_observations, problem.n_parameters

    hidden = args.hidden or width_for_budget(y_dims, x_dims, args.components, args.layers, args.param_budget)
    model = MDN(y_dims, x_dims, args.components, hidden, args.layers, args.activation)
    n_params = count_parameters(model)

    run_name = args.run_name or f'{args.problem}_K{args.components}_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), 'hidden_resolved': hidden, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: hidden={hidden}, {n_params:,} trainable parameters, device={device}')

    # Separate seeds per split so train / val / test never overlap.
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

    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss', 'val_nll', 'lr', 'seconds'])
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            kind = args.warmup_loss if (step < args.warmup_steps and args.warmup_loss != 'none') else 'exact'
            loss = model.loss(x_tr[idx], y_tr[idx], kind=kind)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step} ({kind})')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running += loss.item(); step += 1
        model.eval()
        val = evaluate_nll(model, x_va, y_va, 10_000)
        train = running / steps_per_epoch + model.x_std.log().sum().item()  # raw x-space, like val
        log.writerow([epoch, step, train, val, sched.get_last_lr()[0], time.time() - t0])
        print(f'epoch {epoch:3d}  train {train:+.4f}  val NLL {val:+.4f}  '
              f'({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))

    # Held-out conditions y* drawn via prior + forward process, as in the paper.
    model.eval()
    x_te, y_te = make_dataset(args.problem, args.n_eval_conditions, seed=30_000 + args.seed)
    y_star = torch.from_numpy(y_te).to(device)
    infer_s = time_inference(model, y_star, args.n_eval_samples, device)

    # Provisional re-simulation error (paper Eq. 11) so runs can be compared now;
    # the shared evaluation code is the one to report.
    samples = model.sample(y_star, args.n_eval_samples).cpu().numpy()
    x_flat = samples.reshape(-1, x_dims).astype(np.float64)
    y_resim = problem.forward_process(x_flat, strict=False).reshape(len(y_te), args.n_eval_samples, y_dims)
    sq_err = np.sum((y_resim - y_te[:, None, :]) ** 2, axis=-1)
    resim_failed = float(np.mean(np.isnan(sq_err)))  # ballistics: sample with no ground impact
    err_resim = float(np.nanmean(sq_err))

    summary = {
        'run': run_name, 'problem': args.problem, 'components': args.components,
        'n_params': n_params, 'hidden': hidden, 'layers': args.layers,
        'val_nll': val, 'test_nll': evaluate_nll(model, torch.from_numpy(x_te).to(device), y_star, 10_000),
        'err_resim_provisional': err_resim,
        'resim_failed_fraction': resim_failed,
        'inference_seconds': infer_s,
        'inference_setup': f'{args.n_eval_samples} samples for each of {len(y_te)} y*, device={device}',
        'train_seconds': time.time() - t0,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
