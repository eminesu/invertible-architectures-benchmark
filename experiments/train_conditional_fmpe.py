"""Train the prior-conditioned FMPE (models/conditional_fmpe.py).

    .venv/bin/python experiments/train_conditional_fmpe.py --problem ballistics --device mps --run-name bal_cfmpe_s0
    .venv/bin/python experiments/train_conditional_fmpe.py --problem kinematics --device mps --run-name kin_cfmpe_s0

Training data are (x, y, c) triples, c the prior parameters of
experiments.fr3_fif_cfmpe_adapters.PRIOR_SPECS, each ~ U[--prior-min, --prior-max] (default: the
problem's range): ballistics lambda in [10, 20] with v0 ~ Poisson(lambda),
kinematics the joint sigmas 1..3, each in [0.25, 0.75]. Everything else matches
experiments/train_fmpe.py and the plain-FMPE headline runs
bal_fmpe_s0 / kin_fmpe_s0: 1M training samples, 50 epochs, batch 1000, Adam 1e-3 + cosine,
weight decay 1e-5, grad clip 10, 6 gated blocks at ~3M parameters, time prior
p(t) ∝ t (alpha = 1). Outputs in experiments/runs/<run-name>/ as train_fmpe.py
(config.json, log.csv, summary.json, model.pt), plus checkpoint.pt after every
epoch: rerunning the same command after an interruption resumes from the newest
complete main/temporary checkpoint (same data, regenerated from the seeds).
Checkpoint replacement retries transient Windows file locks. Final numbers come from
experiments/evaluate_conditional_fmpe.py.

FR3: seven independent width scales in [0.5, 1], contracting the URDF joint
intervals about their midpoints. By default 25% of rows use all widths = 1,
anchoring the benchmark prior. Score only that nominal prior against the
professor's exact reference; see docs/fr3_fif_cfmpe.md.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from experiments.cfmpe_checkpoint_io import align_resume_log, atomic_save, load_latest_checkpoint
from experiments.fr3_fif_cfmpe_adapters import PRIOR_SPECS, conditional_model_class, make_conditional_dataset
from experiments.train_fmpe import RUNS, pick_device
from metrics.benchmarks import forward_fn
from models.fmpe import ACTIVATIONS, count_parameters, width_for_budget


@torch.no_grad()
def evaluate_loss(model, x, y, c, noise, batch_size):
    total = 0.0
    for s in range(0, len(x), batch_size):
        sl = slice(s, s + batch_size)
        total += model.loss(x[sl], y[sl], c[sl], noise=(noise[0][sl], noise[1][sl])).item() * len(x[sl])
    return total / len(x)


@torch.no_grad()
def provisional_resim(model, problem, y, c, n_conditions=200, n_samples=100, steps=16):
    """Mean ||f(x) - y||^2 over a few validation (y, c): a cheap progress
    signal. Report metrics.evaluate numbers, not this."""
    y, c = y[:n_conditions], c[:n_conditions]
    x = model.sample(y, c, n_samples, steps=steps).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn(problem)(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(PRIOR_SPECS), default='ballistics')
    p.add_argument('--prior-min', type=float, default=None, help="default: the experiment's PRIOR_SPECS range")
    p.add_argument('--prior-max', type=float, default=None)
    p.add_argument('--nominal-fraction', type=float, default=None,
                   help='Fraction of training/validation rows at the benchmark prior; default: 0.25 for FR3, 0 for toys')
    p.add_argument('--blocks', type=int, default=6, help='gated residual blocks (2 linear layers each)')
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--activation', default='silu', choices=list(ACTIVATIONS))
    p.add_argument('--sigma-min', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='time prior p(t) ∝ t^alpha; 1 = the plain-FMPE headlines bal/kin_fmpe_s0 (docs/fmpe.md)')
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--run-name', default=None)
    args = p.parse_args(argv)
    spec = PRIOR_SPECS[args.problem]
    prior_range = (spec['range'][0] if args.prior_min is None else args.prior_min,
                   spec['range'][1] if args.prior_max is None else args.prior_max)
    if args.nominal_fraction is None:
        args.nominal_fraction = 0.25 if args.problem.startswith('fr3_') else 0.0
    if (not np.isfinite(prior_range).all() or prior_range[0] >= prior_range[1]
            or not all(prior_range[0] <= d <= prior_range[1] for d in spec['default'])):
        p.error('The finite, increasing prior range must contain the benchmark prior')
    if args.problem.startswith('fr3_') and not 0 < prior_range[0] < prior_range[1] <= 1:
        p.error('FR3 width scales must stay inside (0, 1]')
    if not np.isfinite(args.nominal_fraction) or not 0 <= args.nominal_fraction < 1:
        p.error('--nominal-fraction must be in [0, 1)')
    if min(args.n_train, args.n_val, args.batch_size) < 2 or min(args.epochs, args.threads, args.blocks, args.param_budget) < 1:
        p.error('Need at least two rows/batch and positive epochs, threads, blocks and budget')
    if args.hidden is not None and args.hidden < 1:
        p.error('--hidden must be positive')
    if not np.isfinite(args.alpha) or args.alpha <= -1:
        p.error('--alpha must be finite and greater than -1')
    if not np.isfinite(args.lr) or args.lr <= 0:
        p.error('--lr must be finite and positive')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    y_dims, x_dims, prior_dims = problem.n_observations, problem.n_parameters, len(spec['names'])

    hidden = args.hidden or width_for_budget(x_dims, y_dims + prior_dims, args.blocks, args.param_budget)
    model = conditional_model_class(args.problem)(
        x_dims, y_dims, hidden, args.blocks, args.activation, args.sigma_min, args.alpha, prior_dims)
    n_params = count_parameters(model)
    if n_params > args.param_budget:
        p.error(f'Model has {n_params:,} parameters, above --param-budget')

    prefix = {'kinematics': 'kin', 'ballistics': 'bal'}.get(args.problem, args.problem)
    run_name = args.run_name or f'{prefix}_cfmpe_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    if run_name in ('.', '..') or Path(run_name).name != run_name or '/' in run_name or '\\' in run_name:
        p.error('--run-name must be a folder name without path components')
    out = os.path.join(RUNS, run_name)
    ckpt_path = os.path.join(out, 'checkpoint.pt')
    if os.path.exists(out) and not any(os.path.isfile(path) for path in (ckpt_path, ckpt_path + '.tmp')):
        p.error(f'Output already exists without a resumable checkpoint: {out}; choose a new run name')
    os.makedirs(out, exist_ok=True)
    ckpt, ckpt_source = load_latest_checkpoint(ckpt_path)
    if ckpt is not None:
        # Existing toy checkpoints predate these two optional arguments.
        previous = {'nominal_fraction': 0.0, 'threads': 4, **ckpt['args']}
        if previous != vars(args):
            p.error('Checkpoint was written with different arguments')
        align_resume_log(os.path.join(out, 'log.csv'), ckpt)
    config = {**vars(args), 'prior_names': spec['names'], 'prior_range': prior_range, 'hidden_resolved': hidden,
              'n_params': n_params, 'device_resolved': str(device), 'torch': torch.__version__,
              'method': 'CFMPE', 'run': run_name, 'prior_mode': 'family',
              'prior_default': list(spec['default']),
              'data_seeds': [10_000 + args.seed, 20_000 + args.seed],
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: hidden={hidden}, {n_params:,} trainable parameters, device={device}')

    # Same seeds per split as train_fmpe.py (the draws differ: c comes first).
    t = time.time()
    x_tr, y_tr, c_tr = (torch.from_numpy(a)
                        for a in make_conditional_dataset(args.problem, args.n_train, 10_000 + args.seed,
                                                         prior_range, args.nominal_fraction))
    x_va, y_va, c_va = (torch.from_numpy(a).to(device)
                        for a in make_conditional_dataset(args.problem, args.n_val, 20_000 + args.seed,
                                                         prior_range, args.nominal_fraction))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr, c_tr)
    model.to(device)
    x_tr, y_tr, c_tr = x_tr.to(device), y_tr.to(device), c_tr.to(device)
    # Fixed (x0, t) for the validation loss, so epochs are comparable.
    g = torch.Generator().manual_seed(1)
    val_noise = (torch.randn(args.n_val, x_dims, generator=g).to(device),
                 (torch.rand(args.n_val, 1, generator=g) ** (1 / (1 + args.alpha))).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (args.n_train + args.batch_size - 1) // args.batch_size
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
        print(f'resumed from {ckpt_source} after epoch {ckpt["epoch"]}')
    for epoch in range(start, args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = model.loss(x_tr[idx], y_tr[idx], c_tr[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running += loss.item() * len(idx); step += 1
        model.eval()
        val = evaluate_loss(model, x_va, y_va, c_va, val_noise, 10_000)
        train = running / args.n_train
        log.writerow([epoch, step, train, val, sched.get_last_lr()[0], time.time() - t0])
        log_file.flush()
        rng_device = {'mps': lambda: torch.mps.get_rng_state(),
                      'cuda': lambda: torch.cuda.get_rng_state()}.get(device.type, lambda: None)()
        atomic_save({'args': vars(args), 'epoch': epoch, 'step': step, 'seconds': time.time() - t0,
                    'train_loss': train, 'val_loss': val,
                    'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng_cpu': torch.get_rng_state(), 'rng_device': rng_device}, ckpt_path)
        print(f'epoch {epoch:3d}  train {train:.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    atomic_save(model.state_dict(), os.path.join(out, 'model.pt'))

    resim, failed = provisional_resim(model, args.problem, y_va, c_va)
    summary = {
        'run': run_name, 'problem': args.problem, 'n_params': n_params, 'hidden': hidden, 'blocks': args.blocks,
        'prior_names': spec['names'], 'prior_range': list(prior_range),
        'nominal_fraction': args.nominal_fraction, 'train_loss': train, 'val_loss': val,
        'err_resim_provisional': resim, 'resim_failed_fraction': failed,
        'provisional_setup': '200 val (y, c) x 100 samples, rk4 16 steps (64 NFE)',
        'train_seconds': time.time() - t0,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    return Path(out)


if __name__ == '__main__':
    main()
