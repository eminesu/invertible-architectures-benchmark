"""Train the Invertible Liquid Neural Network (ILNN) on one benchmark problem.

    .venv/bin/python experiments/train_ilnn.py --problem kinematics
    .venv/bin/python experiments/train_ilnn.py --problem ballistics --dequantize-v0
    .venv/bin/python experiments/train_ilnn.py --problem fr3_6d --conditional
    .venv/bin/python experiments/train_ilnn.py --problem fr3_6d --pad-x 3

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
from models.ilnn import CILNN, ILNN, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
TERMS = ('y', 'z', 'x')


def build_model(cfg, x_dims, y_dims):
    """Model for a config dict (command-line arguments, or a saved config.json)."""
    if cfg.get('conditional', False):   # old configs: unconditional
        return CILNN(x_dims, y_dims, n_blocks=cfg['blocks'], hidden=cfg['hidden_resolved'],
                     unfolds=cfg['unfolds'], dt=cfg['dt'], clamp=cfg['clamp'], perm_seed=cfg['seed'],
                     lam_init_std=cfg.get('lam_init_std', 0.0), perm_mode=cfg.get('perm_mode', 'random'))
    return ILNN(x_dims, y_dims, n_blocks=cfg['blocks'], hidden=cfg['hidden_resolved'],
                unfolds=cfg['unfolds'], dt=cfg['dt'], clamp=cfg['clamp'],
                latent_mmd=cfg['latent_mmd'], perm_seed=cfg['seed'],
                loss=cfg.get('loss', 'mmd'), sigma_y=cfg.get('sigma_y', 0.05),  # old configs: paper loss
                lam_init_std=cfg.get('lam_init_std', 0.0), perm_mode=cfg.get('perm_mode', 'random'),
                pad_dims=cfg.get('pad_x', 0), pad_noise=cfg.get('pad_noise', 0.02))


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
    p.add_argument('--conditional', action='store_true',
                   help='CILNN: condition every liquid cell on y, dim z = dim x, cINN likelihood loss (Eq. 5); '
                        'needed when dim y >= dim x (fr3_6d). Not in the paper; --loss, --w-* and --latent-mmd are ignored')
    p.add_argument('--pad-x', type=int, default=0,
                   help='unconditional ILNN: pad x with this many zero channels, dim z = dim x + pad - dim y '
                        '(fr3_6d needs >= 3); the padding is dropped when sampling. Not in the paper')
    p.add_argument('--pad-noise', type=float, default=0.02, help='--pad-x: std of the noise added to the padding while training')
    p.add_argument('--loss', choices=['mmd', 'ml'], default='mmd',
                   help="'mmd': the paper's Eq. (16); 'ml': maximum likelihood like the INN (a deviation from the paper)")
    p.add_argument('--sigma-y', type=float, default=0.05, help="--loss ml: std of y_hat around y, as the INN's sigma_y")
    p.add_argument('--latent-mmd', choices=['joint', 'marginal'], default='joint',
                   help="MMD on [y_hat, z] vs [y, eps] ('joint', as train_inn.py) or on z vs eps alone ('marginal')")
    p.add_argument('--dequantize-v0', action='store_true',
                   help='ballistics: add U(-.5,.5) to the integer launch speed in training/validation data')
    # Extras (all default to the old behaviour; old config.json files still load).
    p.add_argument('--lam-init-std', type=float, default=0.0,
                   help='std of the N(0, s) init of the cell drive Lambda; 0 (default) = zeros as before')
    p.add_argument('--perm-mode', choices=['random', 'mixing'], default='random',
                   help="'random': as before; 'mixing': redraw each inter-block permutation until it mixes the two halves")
    p.add_argument('--no-wd-cell-scalars', action='store_true',
                   help='no weight decay on Lambda and raw_tau (default: decay on everything, as before)')
    p.add_argument('--log-cell-stats', action='store_true',
                   help='append mean |Lambda| and mean tau to log.csv and the epoch line')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the flow forward/inverse (state dict keys unchanged); untested on GPU')
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
    if args.conditional and args.pad_x:
        p.error('--pad-x is for the unconditional ILNN; the conditional model does not need padding')
    if not args.conditional and x_dims + args.pad_x <= y_dims:
        p.error(f'{args.problem}: dim x = {x_dims} + pad {args.pad_x} <= dim y = {y_dims}, so the unconditional ILNN '
                f'has no latent dimension (z = {x_dims + args.pad_x - y_dims}); use --pad-x {y_dims - x_dims + 1} '
                'or more, or --conditional')
    if args.conditional:
        args.loss = 'ml'   # recorded in config.json: the CILNN always trains by likelihood

    hidden = args.hidden or width_for_budget(x_dims + args.pad_x, args.blocks, args.param_budget,
                                             cond_dim=y_dims if args.conditional else 0)
    cfg = {**vars(args), 'model': 'ilnn', 'hidden_resolved': hidden}
    model = build_model(cfg, x_dims, y_dims)
    n_params = count_parameters(model)
    weights = {'nll': 1.0, 'y': args.w_y, 'z': args.w_z, 'x': args.w_x}
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')  # TF32 matmuls: the 427x427 recurrent products dominate

    run_name = args.run_name or f'{args.problem}_ilnn_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**cfg, 'z_dims': model.z_dims, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: liquid width={hidden}, dim z={model.z_dims}, {n_params:,} trainable parameters, device={device}')
    report = model.flow.permutation_report()
    print('permutations (mixing the halves: ' + f'{sum(r[2] for r in report)}/{len(report)}): '
          + ' '.join(f'{r[1]}{"" if r[2] else "!"}' for r in report))

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

    if args.no_wd_cell_scalars:
        scalars = [p_ for n, p_ in model.named_parameters() if n.endswith(('.lam', '.raw_tau'))]
        rest = [p_ for n, p_ in model.named_parameters() if not n.endswith(('.lam', '.raw_tau'))]
        opt = torch.optim.Adam([{'params': rest, 'weight_decay': args.weight_decay},
                                {'params': scalars, 'weight_decay': 0.0}], lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.compile:   # assigned on the instance, so state_dict keys and model.pt stay the same
        model.flow.forward = torch.compile(model.flow.forward)
        model.flow.inverse = torch.compile(model.flow.inverse)
    steps_per_epoch = args.n_train // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    terms_logged = sorted(set(TERMS + ('nll',)) & set(model.losses(x_va[:8], y_va[:8]).keys()))
    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss', 'val_loss'] + [f'val_{k}' for k in terms_logged]
                 + ['val_resim', 'val_resim_failed', 'lr', 'seconds']
                 + (['lam_abs_mean', 'tau_mean'] if args.log_cell_stats else []))
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
            if step % 100 == 0 and not torch.isfinite(loss):  # .item() syncs the GPU, so not every step
                raise RuntimeError(f'non-finite loss at step {step}: { {k: v.item() for k, v in terms.items()} }')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            running = running + loss.detach(); step += 1   # stays on the GPU until the epoch ends
        running = float(running)
        if running != running or running in (float('inf'), float('-inf')):
            raise RuntimeError(f'non-finite training loss in epoch {epoch}')
        model.eval()
        val, val_loss = evaluate_losses(model, x_va, y_va, args.batch_size, weights)
        resim, failed = provisional_resim(model, args.problem, y_va)
        stats = model.cell_stats() if args.log_cell_stats else {}
        log.writerow([epoch, step, running / steps_per_epoch, val_loss] + [val[k] for k in terms_logged]
                     + [resim, failed, sched.get_last_lr()[0], time.time() - t0] + list(stats.values()))
        print(f'epoch {epoch:3d}  train {running / steps_per_epoch:+.4f}  val {val_loss:+.4f}  '
              + '  '.join(f'{k} {val[k]:+.4f}' for k in terms_logged)
              + f'  resim {resim:.4f}'
              + ''.join(f'  {k} {v:.4f}' for k, v in stats.items())
              + f'  ({time.time() - t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    summary = {'run': run_name, 'problem': args.problem, 'model': 'ilnn', 'n_params': n_params,
               'hidden': hidden, 'blocks': args.blocks, 'unfolds': args.unfolds, 'dt': args.dt,
               'z_dims': model.z_dims, 'val_loss': val_loss, **{f'val_{k}': v for k, v in val.items()},
               'err_resim_provisional': resim, 'resim_failed_fraction': failed,
               'train_seconds': time.time() - t0,
               'lam_init_std': args.lam_init_std, 'perm_mode': args.perm_mode,
               'no_wd_cell_scalars': args.no_wd_cell_scalars, **model.cell_stats()}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()