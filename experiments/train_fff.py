"""Train a conditional Free-Form Flow (models/fff.py) on one benchmark problem,
Euclidean or with the periodic coordinates on a torus (M-FFF).

    .venv/bin/python experiments/train_fff.py --problem kinematics --device mps --run-name kin_fff_s0
    .venv/bin/python experiments/train_fff.py --problem kinematics --manifold torus --device mps --run-name kin_mfff_s0
    .venv/bin/python experiments/train_fff.py --problem ballistics --device mps --run-name bal_fff_s0

Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (hidden width, parameter count)
    log.csv         per epoch: train / val training objective (train_loss, val_loss),
                    its surrogate-NLL and reconstruction parts, and the exact
                    NLL (encoder change of variables, raw x units) on the val
                    set and on a fixed train subset of the same size (train_nll)
    checkpoint.pt   after every epoch; rerunning the same command resumes
    summary.json    final losses, parameter count, provisional re-simulation error
    model.pt        state dict

Data, schedule and optimizer match train_fmpe.py / train_mdn.py (1M train
samples, 50 epochs, batch 1000, Adam + cosine, ~3M parameters for encoder and
decoder together), so the comparison differs only in the model. One FFF-specific
data change: y gets fresh Gaussian noise of the ground-truth likelihood's sigma
(metrics/ground_truth.py) in every batch, so FFF fits exactly the posterior the
ground truth samples. With noiseless y = f(x), p(x | y) is concentrated on a
lower-dimensional set and maximum likelihood is unbounded; FFF then trades
reconstruction for likelihood (docs/fff.md). Final numbers come from
experiments/evaluate_fff.py (metrics.evaluate), not from this script.
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
from metrics.ground_truth import GT_SIGMA
from models.fff import ACTIVATIONS, ANGLE_DIMS, ConditionalFFF, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
TERMS = ('loss', 'nll_surrogate', 'recon')


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(cfg, problem):
    angle_dims = ANGLE_DIMS[cfg['problem']] if cfg['manifold'] == 'torus' else ()
    return ConditionalFFF(problem.n_parameters, problem.n_observations, cfg['hidden_resolved'], cfg['blocks'],
                          cfg['activation'], cfg['beta'], angle_dims, cfg['beta_z'], cfg['beta_p'],
                          cfg.get('latent', 'uniform'))


def evaluate_terms(model, x, y, v, batch_size):
    """Batch-size-weighted means of the loss terms with the fixed Hutchinson
    draws v, plus the exact NLL. Needs autograd (the loss uses a VJP)."""
    sums = dict.fromkeys(TERMS + ('nll',), 0.0)
    for s in range(0, len(x), batch_size):
        sl = slice(s, s + batch_size)
        t = model.loss_terms(x[sl], y[sl], v=v[sl])
        for k in TERMS:
            sums[k] += t[k].item() * len(x[sl])
        sums['nll'] += -model.log_prob(x[sl], y[sl]).detach().sum().item()
    return {k: v / len(x) for k, v in sums.items()}


@torch.no_grad()
def provisional_resim(model, problem, y, n_conditions=200, n_samples=100):
    """Mean ||f(x) - y||^2 over a few validation y: a cheap progress signal.
    Report metrics.evaluate numbers, not this."""
    y = y[:n_conditions]
    x = model.sample(y, n_samples).cpu().numpy().reshape(-1, model.x_dims)
    y_resim = forward_fn(problem)(x).reshape(len(y), n_samples, -1)
    sq = ((y_resim - y.cpu().numpy()[:, None, :]) ** 2).sum(-1)
    return float(np.nanmean(sq)), float(np.isnan(sq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    p.add_argument('--manifold', choices=['euclidean', 'torus'], default='euclidean',
                   help=f'torus: periodic coordinates {ANGLE_DIMS} embedded on circles (M-FFF)')
    p.add_argument('--latent', choices=['uniform', 'wrapped'], default='uniform',
                   help='torus latent on the circles: uniform (M-FFF) or wrapped normal with the angle std')
    p.add_argument('--beta', type=float, default=100.0, help='x-reconstruction weight')
    p.add_argument('--y-noise', type=float, default=None,
                   help='std of Gaussian noise added to y in training: y = f(x) + noise. Default: the '
                        f'ground-truth likelihood sigma {GT_SIGMA}, so FFF fits the same posterior the GT samples '
                        '(with noiseless y, p(x | y) lives on a lower-dimensional set and its likelihood is unbounded)')
    p.add_argument('--beta-z', type=float, default=0.0, help='latent reconstruction weight (M-FFF)')
    p.add_argument('--beta-p', type=float, default=0.0, help='projection-distance weight (M-FFF)')
    p.add_argument('--blocks', type=int, default=6, help='residual blocks per network (2 linear layers each)')
    p.add_argument('--hidden', type=int, default=None, help='hidden width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000, help='encoder + decoder together')
    p.add_argument('--activation', default='silu', choices=list(ACTIVATIONS))
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

    if args.y_noise is None:
        args.y_noise = GT_SIGMA[args.problem]
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    y_dims, x_dims = problem.n_observations, problem.n_parameters
    ambient = x_dims + (len(ANGLE_DIMS[args.problem]) if args.manifold == 'torus' else 0)

    hidden = args.hidden or width_for_budget(ambient, y_dims, args.blocks, args.param_budget)
    config = {**vars(args), 'hidden_resolved': hidden}
    model = build_model(config, problem)
    n_params = count_parameters(model)

    run_name = args.run_name or f'{args.problem}_fff_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    ckpt_path = os.path.join(out, 'checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False) if os.path.exists(ckpt_path) else None
    if ckpt is not None:
        assert ckpt['args'] == vars(args), 'checkpoint was written with different arguments'
    config.update(n_params=n_params, ambient_dims=ambient, angle_dims=list(model.angle_dims),
                  device_resolved=str(device), torch=torch.__version__)
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: {args.manifold}, beta={args.beta:g}, hidden={hidden}, {n_params:,} trainable parameters, '
          f'device={device}')

    # Same seeds per split as train_mdn.py / train_fmpe.py, so every model sees the same data.
    t = time.time()
    x_tr, y_tr = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_train, seed=10_000 + args.seed))
    x_va, y_va = (torch.from_numpy(a).to(device) for a in make_dataset(args.problem, args.n_val, seed=20_000 + args.seed))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr)
    model.to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    # Fixed Hutchinson draws for the validation loss, so epochs are comparable.
    g = torch.Generator().manual_seed(1)
    val_v = torch.randn(args.n_val, ambient, generator=g).to(device)
    y_va_noisy = y_va + args.y_noise * torch.randn(y_va.shape, generator=g).to(device)
    # Fixed train subset (with fixed noise) for the exact train NLL: same quantity as val_nll.
    sub = torch.randperm(args.n_train, generator=g)[:args.n_val].to(device)
    x_sub, y_sub = x_tr[sub], y_tr[sub] + args.y_noise * torch.randn(args.n_val, y_dims, generator=g).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    log_file = open(os.path.join(out, 'log.csv'), 'a' if ckpt else 'w', newline='')
    log = csv.writer(log_file)
    step, start, t0 = 0, 0, time.time()
    if ckpt is None:
        log.writerow(['epoch', 'step'] + [f'{s}_{k}' for k in TERMS for s in ('train', 'val')]
                     + ['train_nll', 'val_nll', 'lr', 'seconds'])
    else:
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt']); sched.load_state_dict(ckpt['sched'])
        torch.set_rng_state(ckpt['rng_cpu'])
        if device.type == 'mps':
            torch.mps.set_rng_state(ckpt['rng_device'])
        elif device.type == 'cuda':
            torch.cuda.set_rng_state(ckpt['rng_device'])
        step, start, t0 = ckpt['step'], ckpt['epoch'] + 1, time.time() - ckpt['seconds']
        train, val = ckpt['train'], ckpt['val']
        print(f'resumed from {ckpt_path} after epoch {ckpt["epoch"]}')
    for epoch in range(start, args.epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        running = dict.fromkeys(TERMS, 0.0)
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            y_b = y_tr[idx]
            terms = model.loss_terms(x_tr[idx], y_b + args.y_noise * torch.randn_like(y_b))
            loss = terms['loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            for k in TERMS:
                running[k] += terms[k].item()
            step += 1
        model.eval()
        val = evaluate_terms(model, x_va, y_va_noisy, val_v, 5_000)
        train = {k: v / steps_per_epoch for k, v in running.items()}
        train['nll'] = evaluate_terms(model, x_sub, y_sub, val_v, 5_000)['nll']
        log.writerow([epoch, step] + [d[k] for k in TERMS for d in (train, val)]
                     + [train['nll'], val['nll'], sched.get_last_lr()[0], time.time() - t0])
        log_file.flush()
        rng_device = {'mps': lambda: torch.mps.get_rng_state(),
                      'cuda': lambda: torch.cuda.get_rng_state()}.get(device.type, lambda: None)()
        torch.save({'args': vars(args), 'epoch': epoch, 'step': step, 'seconds': time.time() - t0,
                    'train': train, 'val': val,
                    'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng_cpu': torch.get_rng_state(), 'rng_device': rng_device}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)  # atomic: a crash mid-save keeps the previous checkpoint
        print(f"epoch {epoch:3d}  loss train {train['loss']:.4f} val {val['loss']:.4f}  "
              f"recon train {train['recon']:.2e} val {val['recon']:.2e}  NLL train {train['nll']:.4f} val {val['nll']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))

    resim, failed = provisional_resim(model, args.problem, y_va)
    summary = {
        'run': run_name, 'problem': args.problem, 'manifold': args.manifold, 'beta': args.beta,
        'y_noise': args.y_noise,
        'n_params': n_params, 'hidden': hidden, 'blocks': args.blocks,
        'train_loss': train['loss'], 'val_loss': val['loss'], 'train_recon': train['recon'],
        'val_recon': val['recon'], 'train_nll': train['nll'], 'val_nll': val['nll'],
        'err_resim_provisional': resim, 'resim_failed_fraction': failed,
        'provisional_setup': '200 val y x 100 samples, one decoder pass',
        'train_seconds': time.time() - t0,
    }
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
