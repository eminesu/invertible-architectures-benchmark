"""Train the Invertible Liquid Neural Network (ILNN) on one benchmark problem.

    .venv/bin/python experiments/train_ilnn.py --problem kinematics
    .venv/bin/python experiments/train_ilnn.py --problem ballistics --dequantize-v0

ILNN: Zhang et al. (2025), Sci. Rep. 15:42311, doi:10.1038/s41598-025-26422-1; loss Eq. (16).
Every run writes to experiments/runs/<run-name>/:
    config.json     all arguments + derived sizes (liquid width, parameter count)
    log.csv         per-epoch train / val loss terms + provisional val re-simulation error
    summary.json    final losses, parameter count, provisional re-simulation error
    model.pt        state dict

Same data, schedule and optimizer defaults as train_inn.py / train_mdn.py /
train_autoencoder.py, so all models see identical training data and the same number of
steps; the helpers (dequantize, evaluate_losses, provisional_resim) are reused from
train_inn.py. Every default marked TODO is a placeholder: the paper gives no value.
Final numbers come from experiments/evaluate_ilnn.py (metrics.evaluate), not from the
provisional re-simulation error.
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS, make_dataset
from experiments.train_mdn import pick_device
from experiments.train_inn import dequantize, evaluate_losses, provisional_resim
from models.ilnn import ILNN, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
TERMS = ('y', 'z', 'x')


def build_model(cfg, x_dims, y_dims):
    """Model for a config dict (command-line arguments, or a saved config.json)."""
    return ILNN(x_dims, y_dims, n_blocks=cfg['blocks'], hidden=cfg['hidden_resolved'],
                unfolds=cfg['unfolds'], dt=cfg['dt'], clamp=cfg['clamp'],
                latent_mmd=cfg['latent_mmd'], perm_seed=cfg['seed'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    # Architecture. TODO: the paper only fixes ~3M trainable parameters (benchmark protocol).
    p.add_argument('--blocks', type=int, default=8, help='liquid coupling blocks')
    p.add_argument('--hidden', type=int, default=None, help='liquid cell width; default: fit --param-budget')
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--unfolds', type=int, default=4, help='fused Euler steps per liquid cell (Eq. 11)')
    p.add_argument('--dt', type=float, default=0.25, help='step size of the fused Euler step')
    p.add_argument('--clamp', type=float, default=2.0, help='soft clamp of the coupling log-scales')
    # Loss, Eq. (16). TODO: w_y, w_z, w_x are not given in the paper.
    p.add_argument('--w-y', type=float, default=1.0, help='Eq. (16): weight of L_y')
    p.add_argument('--w-z', type=float, default=100.0, help='Eq. (16): lambda_z, weight of the latent MMD')
    p.add_argument('--w-x', type=float, default=100.0, help='Eq. (16): lambda_x, weight of the backward MMD')
    p.add_argument('--latent-mmd', choices=['joint', 'marginal'], default='joint',
                   help="MMD on [y_hat, z] vs [y, eps] ('joint', as train_inn.py) or on z vs eps alone ('marginal')")
    p.add_argument('--dequantize-v0', action='store_true',
                   help='ballistics: add U(-.5,.5) to the integer launch speed in training/validation data')
    # Schedule: same placeholders as the other models.
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

    hidden = args.hidden or width_for_budget(x_dims, args.blocks, args.param_budget)
    cfg = {**vars(args), 'model': 'ilnn', 'hidden_resolved': hidden}
    model = build_model(cfg, x_dims, y_dims)
    n_params = count_parameters(model)
    weights = {'y': args.w_y, 'z': args.w_z, 'x': args.w_x}

    run_name = args.run_name or f'{args.problem}_ilnn_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**cfg, 'z_dims': model.z_dims, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: liquid width={hidden}, dim z={model.z_dims}, {n_params:,} trainable parameters, device={device}')

    # Same split seeds as the other models, so all models see identical data.
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

    terms_logged = sorted(TERMS)
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
    summary = {'run': run_name, 'problem': args.problem, 'model': 'ilnn', 'n_params': n_params,
               'hidden': hidden, 'blocks': args.blocks, 'unfolds': args.unfolds, 'dt': args.dt,
               'z_dims': model.z_dims, 'val_loss': val_loss, **{f'val_{k}': v for k, v in val.items()},
               'err_resim_provisional': resim, 'resim_failed_fraction': failed,
               'train_seconds': time.time() - t0}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()