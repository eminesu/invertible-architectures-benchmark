"""Train IKFlow (Ames et al., arXiv:2111.08933) on one benchmark problem.

    .venv/bin/python experiments/train_ikflow.py --problem fr3_6d                  # 3M parameters
    .venv/bin/python experiments/train_ikflow.py --problem fr3_6d --paper-size     # 1024-wide subnets, 2.5M points
    .venv/bin/python experiments/train_ikflow.py --problem fr3_6d --param-budget 300000 --n-train 100000   # small, for scaling curves

Model: models/ikflow.py (conditional GLOW flow with zero-padded width, SoftFlow noise, latent
sub-sampling). Training recipe from the paper's appendix: maximum likelihood with SoftFlow noise
1e-3, batch size 128, learning rate 5e-4 decayed by 0.979 after every 39000 batches, Ranger
optimizer (RAdam + Lookahead; Lookahead k = 6, alpha = 0.5 are Ranger's defaults, not stated in the
paper), 2.5M training points "until convergence" (the number of epochs is not stated).

Extras for tuning (off by default, so the defaults are the paper's recipe): --steps (a fixed number
of optimizer steps instead of epochs, so runs with different batch sizes cost the same number of
steps), --schedule cosine, --ema DECAY (an exponential moving average of the weights, which is what
gets evaluated and saved as model.pt when it is on). experiments/run_ikflow_baseline.sh searches them.

Differences from the paper, all flags: the default size is fitted to --param-budget (3M, the budget
of the other models in this repository) instead of the paper's ~5e7 parameters (--paper-size gives
1024-wide subnets and 2.5M points); the default is 1M points x 50 epochs; gradient-norm clipping at
10 (the paper does not mention clipping; --grad-clip 0 disables it).

Every run writes to experiments/runs/<run-name>/ (config.json, log.csv, summary.json, model.pt),
like train_ilnn.py / train_inn.py, and uses the same data seeds, so all models see identical data.
Final numbers come from experiments/evaluate_ikflow.py (metrics.evaluate); the training-time
re-simulation error uses the model's latent_scale (0.25, the paper's setting).
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
from experiments.train_inn import evaluate_losses, provisional_resim
from models.ikflow import IKFlow, count_parameters, width_for_budget

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


class Lookahead:
    """Lookahead (Zhang et al. 2019) around a torch optimizer: every k steps the slow weights move a
    fraction alpha towards the fast weights and the fast weights restart from them. With RAdam this is
    Ranger, the optimizer of the IKFlow paper. Only what the training loop needs is implemented."""

    def __init__(self, base, k=6, alpha=0.5):
        self.base, self.k, self.alpha, self.n = base, k, alpha, 0
        self.slow = [[q.detach().clone() for q in group['params']] for group in base.param_groups]

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.base.step()
        self.n += 1
        if self.n % self.k == 0:
            for group, slows in zip(self.base.param_groups, self.slow):
                for q, s in zip(group['params'], slows):
                    s.add_(q - s, alpha=self.alpha)
                    q.copy_(s)


def make_optimizer(params, name, lr, k, alpha):
    """(optimizer used for stepping, torch optimizer the LR scheduler attaches to)."""
    if name == 'adam':
        opt = torch.optim.Adam(params, lr=lr)
        return opt, opt
    base = torch.optim.RAdam(params, lr=lr)
    return (Lookahead(base, k, alpha) if name == 'ranger' else base), base


class EMA:
    """Exponential moving average of the parameters, shadow = d * shadow + (1 - d) * param, with a short
    warm-up of d so that the average does not start at the initial weights. swap_in() puts the average
    into the model (for validation and saving) and swap_out() restores the training weights."""

    def __init__(self, params, decay):
        self.decay, self.params = decay, list(params)
        self.shadow = [q.detach().clone() for q in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, [q.detach() for q in self.params], alpha=1 - d)

    @torch.no_grad()
    def swap_in(self):
        self.backup = [q.detach().clone() for q in self.params]
        for sh, q in zip(self.shadow, self.params):
            q.copy_(sh)

    @torch.no_grad()
    def swap_out(self):
        for b, q in zip(self.backup, self.params):
            q.copy_(b)
        self.backup = None


def resolve_epochs(epochs, steps, steps_per_epoch):
    """--steps (when given) replaces --epochs: the whole epochs that come closest to that many steps."""
    return epochs if not steps else max(1, round(steps / steps_per_epoch))


def build_model(cfg, x_dims, y_dims):
    """Model for a config dict (command-line arguments, or a saved config.json)."""
    return IKFlow(x_dims, y_dims, width=cfg['width_resolved'], n_blocks=cfg['blocks'],
                  hidden=cfg['hidden_resolved'], n_layers=cfg['layers'], activation=cfg['activation'],
                  clamp=cfg['clamp'], softflow_scale=cfg['softflow_scale'], latent_scale=cfg['latent_scale'],
                  perm_seed=cfg['seed'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    # Architecture (paper Table I: Panda 12 blocks, width 9 = joints + 2; subnets 3 x 1024, LeakyReLU).
    p.add_argument('--blocks', type=int, default=12, help='coupling blocks (paper Table I: 8-16)')
    p.add_argument('--width', type=int, default=None, help='flow width, >= dim x; default dim x + 2')
    p.add_argument('--hidden', type=int, default=None, help='subnet width; default: fit --param-budget')
    p.add_argument('--layers', type=int, default=3, help='hidden layers per subnet')
    p.add_argument('--activation', default='leaky_relu', choices=['leaky_relu', 'relu', 'silu', 'tanh'])
    p.add_argument('--clamp', type=float, default=2.0)
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--paper-size', action='store_true',
                   help='hidden 1024 and 2.5M training points (unless --hidden / --n-train are given): ~5e7 parameters')
    # SoftFlow and sampling.
    p.add_argument('--softflow-scale', type=float, default=1e-3, help='noise std at c = 1, in raw joint units (rad)')
    p.add_argument('--latent-scale', type=float, default=0.25,
                   help='z std when sampling (the paper: 0.25); used by the training-time re-simulation check')
    # Optimisation (paper appendix).
    p.add_argument('--optimizer', choices=['ranger', 'radam', 'adam'], default='ranger')
    p.add_argument('--schedule', choices=['step', 'cosine'], default='step',
                   help="'step': the paper's x --lr-decay every --lr-decay-steps batches; 'cosine': to 0 over the run")
    p.add_argument('--ema', type=float, default=0.0, help='decay of an exponential moving average of the weights; 0 = off')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr-decay', type=float, default=0.979, help='multiplicative LR decay ...')
    p.add_argument('--lr-decay-steps', type=int, default=39000, help='... after this many batches')
    p.add_argument('--lookahead-k', type=int, default=6)
    p.add_argument('--lookahead-alpha', type=float, default=0.5)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--n-train', type=int, default=None, help='default 1,000,000 (2,500,000 with --paper-size)')
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--steps', type=int, default=None, help='total optimizer steps; replaces --epochs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--run-name', default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    problem = MODELS[args.problem]()
    x_dims, y_dims = problem.n_parameters, problem.n_observations
    width = args.width or x_dims + 2
    if width < x_dims:
        p.error(f'--width {width} is smaller than dim x = {x_dims}')
    hidden = args.hidden or (1024 if args.paper_size else
                             width_for_budget(width, y_dims + 1, args.blocks, args.layers, args.param_budget))
    n_train = args.n_train or (2_500_000 if args.paper_size else 1_000_000)
    steps_per_epoch = max(1, n_train // args.batch_size)
    epochs = resolve_epochs(args.epochs, args.steps, steps_per_epoch)

    cfg = {**vars(args), 'model': 'ikflow', 'width_resolved': width, 'hidden_resolved': hidden,
           'n_train_resolved': n_train, 'epochs_resolved': epochs}
    model = build_model(cfg, x_dims, y_dims)
    n_params = count_parameters(model)
    weights = {'nll': 1.0}
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    run_name = args.run_name or f'{args.problem}_ikflow_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    out = os.path.join(RUNS, run_name)
    os.makedirs(out, exist_ok=True)
    config = {**cfg, 'z_dims': model.z_dims, 'n_params': n_params, 'device_resolved': str(device),
              'torch': torch.__version__}
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    print(f'{run_name}: width={width} (dim x {x_dims} + {width - x_dims} padding), subnets {args.layers} x {hidden}, '
          f'{args.blocks} blocks, {n_params:,} trainable parameters, {epochs} epochs x {steps_per_epoch} steps, '
          f'device={device}')

    # Same split seeds as the other models, so all models see identical data.
    t = time.time()
    x_tr, y_tr = (torch.from_numpy(a) for a in make_dataset(args.problem, n_train, seed=10_000 + args.seed))
    x_va, y_va = (torch.from_numpy(a) for a in make_dataset(args.problem, args.n_val, seed=20_000 + args.seed))
    print(f'data generated in {time.time() - t:.0f}s')

    model.set_normalization(x_tr, y_tr)
    model.to(device)
    x_tr, y_tr, x_va, y_va = x_tr.to(device), y_tr.to(device), x_va.to(device), y_va.to(device)

    opt, base = make_optimizer(model.parameters(), args.optimizer, args.lr, args.lookahead_k, args.lookahead_alpha)
    if args.schedule == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=epochs * steps_per_epoch)
    else:
        sched = torch.optim.lr_scheduler.StepLR(base, step_size=args.lr_decay_steps, gamma=args.lr_decay)
    ema = EMA(model.parameters(), args.ema) if args.ema > 0 else None

    terms_logged = ['nll']
    log = csv.writer(open(os.path.join(out, 'log.csv'), 'w', newline=''))
    log.writerow(['epoch', 'step', 'train_loss', 'val_loss'] + [f'val_{k}' for k in terms_logged]
                 + ['val_resim', 'val_resim_failed', 'lr', 'seconds'])
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = model.losses(x_tr[idx], y_tr[idx])['nll']
            if step % 100 == 0 and not torch.isfinite(loss):  # .item() syncs the GPU, so not every step
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            if ema is not None:
                ema.update(step)
            running = running + loss.detach(); step += 1   # stays on the GPU until the epoch ends
        running = float(running)
        if running != running or running in (float('inf'), float('-inf')):
            raise RuntimeError(f'non-finite training loss in epoch {epoch}')
        model.eval()
        if ema is not None:
            ema.swap_in()
        val, val_loss = evaluate_losses(model, x_va, y_va, 1000, weights)
        resim, failed = provisional_resim(model, args.problem, y_va)
        if ema is not None:
            ema.swap_out()
        log.writerow([epoch, step, running / steps_per_epoch, val_loss] + [val[k] for k in terms_logged]
                     + [resim, failed, sched.get_last_lr()[0], time.time() - t0])
        print(f'epoch {epoch:3d}  train {running / steps_per_epoch:+.4f}  val {val_loss:+.4f}  '
              f'resim {resim:.4f}  lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)', flush=True)

    if ema is not None:
        ema.swap_in()   # model.pt holds the averaged weights, the ones that were validated
    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    summary = {'run': run_name, 'problem': args.problem, 'model': 'ikflow', 'n_params': n_params,
               'hidden': hidden, 'width': width, 'blocks': args.blocks, 'z_dims': model.z_dims,
               'n_train': n_train, 'epochs': epochs, 'steps': step, 'ema': args.ema, 'schedule': args.schedule, 'val_loss': val_loss, **{f'val_{k}': v for k, v in val.items()},
               'err_resim_provisional': resim, 'resim_failed_fraction': failed,
               'latent_scale_resim': args.latent_scale, 'train_seconds': time.time() - t0}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
