"""Train, evaluate and diagnose standard AE, CVAE and tied-transpose InvAuto.

Examples (Windows):
  .venv/Scripts/python experiments/autoencoder_bench.py train --family cvae --problem kinematics --profile pilot
  .venv/Scripts/python experiments/autoencoder_bench.py evaluate RUN --scope pilot
  .venv/Scripts/python experiments/autoencoder_bench.py diagnose RUN

Pilot results are explicitly labelled and are never paper-replication results.
"""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.toy_data import make_dataset
from metrics.benchmarks import forward_fn, load_test_conditions
from metrics.evaluate import evaluate, summarize
from metrics.mmd import mmd
from models.autoencoder_family import FAMILIES, build_model

RUNS = ROOT / 'experiments' / 'runs'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def hardware(device):
    return {'platform': platform.platform(), 'torch': torch.__version__, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'threads': torch.get_num_threads()}


def load_run(run, device='cpu'):
    folder = RUNS / run
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True))
    return model.eval(), cfg, folder


@torch.no_grad()
def validation(model, x, y, batch_size, problem):
    totals = {}
    # Validation draws do not advance the training RNG stream.
    devices = [x.device.index or 0] if x.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(991)
        for s in range(0, len(x), batch_size):
            terms = model.losses(x[s:s+batch_size], y[s:s+batch_size])
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.0) + v.item()*len(x[s:s+batch_size])
        draws = model.sample(y[:64], 32).cpu().numpy()
    resim = forward_fn(problem)(draws.reshape(-1, 4)).reshape(len(draws), 32, -1)
    errors = ((resim-y[:64].cpu().numpy()[:, None])**2).sum(-1)
    finite = np.isfinite(errors)
    # All invalid draws receive a penalty for checkpoint selection only.
    score = np.where(finite, errors, 100.0).mean()
    return {k: v/len(x) for k, v in totals.items()}, float(score), float((~finite).mean())


def train(args):
    cfg = vars(args).copy()
    cfg.pop('command')
    full = args.profile == 'full'
    for k, v in {'n_train': 1_000_000 if full else 30_000,
                 'n_val': 20_000 if full else 2_000, 'epochs': 50 if full else 12,
                 'batch_size': 1000 if full else 256}.items():
        if cfg[k] is None:
            cfg[k] = v
    if not full and cfg['hidden'] is None:
        cfg['param_budget'] = min(cfg['param_budget'], 100_000)
    if min(cfg['n_train'], cfg['n_val']) < 2 or cfg['batch_size'] < 2 or cfg['epochs'] < 1:
        raise ValueError('At least 2 data rows / batch and 1 epoch are required')
    if min(args.w_y, args.w_z, args.beta, args.orth_weight, args.weight_decay) < 0 or args.lr <= 0:
        raise ValueError('Loss weights must be nonnegative and learning rate positive')
    if args.threads < 1 or args.grad_clip <= 0:
        raise ValueError('threads and grad_clip must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(cfg)
    cfg['hidden_resolved'] = model.encoder[0].out_features
    cfg['z_dims'] = model.z_dims
    cfg['n_params'] = sum(p.numel() for p in model.parameters())
    cfg['hardware'] = hardware(device)
    cfg['data_seeds'] = {'train': 10_000+args.seed, 'validation': 20_000+args.seed}
    cfg['ballistics_handling'] = 'original prior; no rounding or dequantization'
    cfg['effective_activation'] = 'modified_leaky_relu' if args.family == 'invertible' else args.activation
    cfg['checkpoint_selection'] = {'split': 'validation', 'conditions': 64, 'samples': 32,
                                   'invalid_penalty': 100.0, 'draw_seed': 991}
    run = args.run_name or f'{args.problem}_{args.family}_{args.profile}_s{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
    folder = RUNS/run
    folder.mkdir(parents=True, exist_ok=False)  # never overwrite an existing experiment
    write_json(folder/'config.json', cfg)
    x, y = (torch.from_numpy(a) for a in make_dataset(args.problem, cfg['n_train'], 10_000+args.seed))
    xv, yv = (torch.from_numpy(a) for a in make_dataset(args.problem, cfg['n_val'], 20_000+args.seed))
    model.set_normalization(x, y)
    model.to(device)
    xv, yv = xv.to(device), yv.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Avoid singleton tails: unbiased MMD requires >=2 rows.
    steps = (len(x)+cfg['batch_size']-1)//cfg['batch_size']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs']*steps)
    weights = {'recon': 1.0, 'y': args.w_y, 'z': args.w_z, 'kl': args.beta}
    best, t0 = float('inf'), time.perf_counter()
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(len(x))
        total, batches = 0.0, 0
        for idx in perm.split(cfg['batch_size']):
            if len(idx) < 2:
                continue
            terms = model.losses(x[idx].to(device), y[idx].to(device))
            loss = sum(weights[k]*v for k, v in terms.items())
            if args.family == 'invertible' and args.orth_weight:
                loss = loss + args.orth_weight*model.orthogonality_terms().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in epoch {epoch}: {terms}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            opt.step(); sched.step()
            total += loss.item(); batches += 1
        model.eval()
        val, score, failed = validation(model, xv, yv, cfg['batch_size'], args.problem)
        if not np.isfinite(score) or any(not np.isfinite(v) for v in val.values()):
            raise RuntimeError(f'Nonfinite validation result in epoch {epoch}')
        if score < best:
            best = score
            torch.save(model.state_dict(), folder/'model.pt')
            best_epoch = epoch+1
        row = {'epoch': epoch+1, 'train_loss': total/batches, **{f'val_{k}': v for k, v in val.items()},
               'val_resim_penalized': score, 'val_failed': failed, 'lr': sched.get_last_lr()[0]}
        history.append(row)
        print(f'{run}: epoch {epoch+1}/{cfg["epochs"]}, val resim {score:.5f}, failed {failed:.3%}', flush=True)
    train_seconds = time.perf_counter()-t0
    torch.save(model.state_dict(), folder/'last_model.pt')
    with (folder/'log.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, history[0].keys()); writer.writeheader(); writer.writerows(history)
    write_json(folder/'summary.json', {'run': run, 'profile': args.profile, 'best_epoch': best_epoch,
                                     'validation_selection_score': best, 'train_seconds': train_seconds,
                                     'n_params': cfg['n_params'], 'hardware': cfg['hardware']})
    print(f'Saved {folder}', flush=True)
    return run


def score_run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, cfg, folder = load_run(args.run, args.device)
    options = {} if args.scope == 'full' else dict(n_conditions=args.conditions, n_samples=args.samples,
                    gt_options={'n_gt': args.samples, 'max_proposals': args.max_proposals})
    result = evaluate(model.sample, cfg['problem'], args.run, out_dir=folder/'evaluation', **options)
    result['training_profile'] = cfg['profile']
    result['hardware'] = hardware(torch.device(args.device))
    result['evaluation_seed'] = args.seed
    result['metric_settings'] = {'kernel': 'IMQ', 'bandwidths': [0.05, 0.2, 0.9],
                                 'units': 'raw', 'ballistics_clamp': 10,
                                 'sigma': 0.01 if cfg['problem'] == 'kinematics' else 0.02}
    write_json(folder/'evaluation'/'summary.json', result)


@torch.no_grad()
def diagnose(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, cfg, folder = load_run(args.run, args.device)
    x, y = (torch.from_numpy(a).to(args.device) for a in make_dataset(cfg['problem'], 2000, 40_000+args.seed))
    xn, yn = (x-model.x_mean)/model.x_std, (y-model.y_mean)/model.y_std
    report = {'run': args.run, 'profile': cfg['profile'], 'diagnostic_seed': args.seed}
    if cfg['family'] == 'cvae':
        mu, logvar = model.posterior(xn, yn)
        z = mu + (logvar*0.5).exp()*torch.randn_like(mu)
        rec = model.decode(yn, mu)
        kl_dim = (0.5*(mu.square()+logvar.exp()-1-logvar)).mean(0)
        report['kl_per_dimension'] = kl_dim.cpu().tolist()
        report['inactive_latent_dimensions_kl_below_0_01'] = int((kl_dim < 0.01).sum())
        report['reconstruction_kind'] = 'posterior mean latent (not stochastic training loss)'
    else:
        yh, z = model.encode(xn)
        rec = model.decode(yh, z)
        report['forward_y_mse_standardized'] = float((yh-yn).square().sum(1).mean())
        # Joint distribution diagnostic detects dependence missed by marginal moments.
        report['joint_latent_mmd'] = mmd(torch.cat([yn[:512], z[:512]], 1).cpu().numpy(),
                                       torch.cat([yn[:512], torch.randn_like(z[:512])], 1).cpu().numpy())
    residual = ((rec-xn)*model.x_std).cpu().numpy()
    zn = z.cpu().numpy()
    report['reconstruction_rmse_raw'] = float(np.sqrt((residual**2).mean()))
    report['reconstruction_median_raw'] = np.median(residual, axis=0).tolist()
    report['latent_mean'] = zn.mean(0).tolist(); report['latent_std'] = zn.std(0).tolist()
    corr = np.corrcoef(np.concatenate([zn, yn.cpu().numpy()], axis=1), rowvar=False)
    report['max_abs_latent_y_correlation'] = float(np.nanmax(np.abs(corr[:model.z_dims, model.z_dims:])))
    if cfg['family'] == 'invertible':
        report['orthogonality_error_per_layer'] = model.orthogonality_terms().cpu().tolist()
    target, _ = load_test_conditions(cfg['problem'])
    target = target[:64]
    if cfg['problem'] == 'kinematics':
        target = np.concatenate([target, [[1.5, 0.0]]])  # fixed difficult observation
    samples = model.sample(target, 256).cpu().numpy().astype(np.float64)
    fy = forward_fn(cfg['problem'])(samples.reshape(-1, 4)).reshape(len(target), 256, -1)
    error = ((fy-target[:, None])**2).sum(-1)
    valid_counts = np.isfinite(error).sum(1)
    condition_errors = np.divide(np.where(np.isfinite(error), error, 0).sum(1), valid_counts,
                                 out=np.full(len(error), np.nan), where=valid_counts > 0)
    report['resim'] = summarize(condition_errors)
    report['failed_fraction'] = float((~np.isfinite(error)).mean())
    report['posterior_variance_raw_mean'] = samples.var(1).mean(0).tolist()
    if cfg['problem'] == 'ballistics':
        report['mean_distance_v0_to_integer'] = float(np.abs(samples[..., 3]-np.rint(samples[..., 3])).mean())
        report['negative_launch_height_fraction'] = float((samples[..., 1] < 0).mean())
        # Supplemental only: never substitutes for raw-unit shared Err_post.
        from metrics.ground_truth import ground_truth_posterior
        gt = ground_truth_posterior('ballistics', target[:8], n_gt=128, max_proposals=2_000_000, verbose=False)
        raw, standardized, rounded = [], [], []
        prior_std = np.array([0.5, 0.5, (0.4*np.pi)/np.sqrt(12), np.sqrt(15)])
        for a, b in zip(samples[:8], gt):
            if len(b) < 2:
                continue
            rounded_a = a.copy(); rounded_a[:, 3] = np.rint(rounded_a[:, 3])
            raw.append(mmd(a, b)); standardized.append(mmd(a/prior_std, b/prior_std)); rounded.append(mmd(rounded_a, b))
        report['supplemental_v0_sensitivity'] = {'n_conditions': len(raw),
            'raw': float(np.mean(raw)) if raw else None,
            'standardized': float(np.mean(standardized)) if raw else None,
            'rounded_v0': float(np.mean(rounded)) if raw else None}
    else:
        report['hard_target_resim'] = float(np.mean(error[-1]))
    out = folder/'diagnostics'; out.mkdir(exist_ok=True)
    write_json(out/'summary.json', report)
    np.savez(out/'samples.npz', samples=samples, y_star=target, residual=residual, z=zn)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(zn, bins=40, density=True, histtype='step'); axes[0].set_title('Encoded latent distribution')
    axes[1].boxplot(residual, showfliers=False); axes[1].set_title('Reconstruction residuals (raw units)')
    means = condition_errors
    axes[2].scatter(np.arange(len(means)), means); axes[2].set_yscale('symlog', linthresh=1e-5)
    axes[2].set_title('Per-condition re-simulation error')
    fig.tight_layout(); fig.savefig(out/'overview.png', dpi=150); plt.close(fig)
    from data.toy_data import MODELS
    simulator = MODELS[cfg['problem']]()
    selected = [0, 1, 2, len(target)-1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, i in zip(axes, selected):
        draw = samples[i, :100]
        if cfg['problem'] == 'kinematics':
            arms = np.stack(simulator.joint_positions(draw), axis=1)
            for arm in arms:
                ax.plot(arm[:, 0], arm[:, 1], alpha=0.08, color='tab:blue')
            ax.scatter(*target[i], marker='+', color='red', s=100)
            ax.set_aspect('equal')
        else:
            xs, ys = simulator.trajectories_from_parameters(draw)
            for xx, yy in zip(xs, ys):
                keep = yy > -0.5
                ax.plot(xx[keep], yy[keep], alpha=0.08, color='tab:blue')
            ax.axvline(target[i, 0], color='red'); ax.axhline(0, color='gray')
        ax.set_title(f'y* #{i}; resim {condition_errors[i]:.3g}')
    fig.tight_layout(); fig.savefig(out/'posterior.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--family', choices=FAMILIES, required=True)
    t.add_argument('--problem', choices=['kinematics', 'ballistics'], required=True)
    t.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    t.add_argument('--run-name'); t.add_argument('--device', default='auto')
    t.add_argument('--param-budget', type=int, default=3_000_000)
    t.add_argument('--hidden', type=int); t.add_argument('--layers', type=int, default=4)
    t.add_argument('--activation', choices=['relu', 'silu', 'tanh', 'leaky_relu'], default='relu')
    t.add_argument('--z-dims', type=int)
    t.add_argument('--latent-mmd', choices=['joint', 'marginal'], default='joint')
    t.add_argument('--recon-condition', choices=['predicted', 'true'], default='predicted')
    t.add_argument('--mmd-scale', type=float, default=1.0)
    t.add_argument('--w-y', type=float, default=1.0); t.add_argument('--w-z', type=float, default=100.0)
    t.add_argument('--beta', type=float, default=0.01)
    t.add_argument('--slope', type=float, default=2.0); t.add_argument('--orth-weight', type=float, default=0.0)
    for key in ['n-train', 'n-val', 'epochs', 'batch-size']:
        t.add_argument('--'+key, type=int)
    t.add_argument('--lr', type=float, default=1e-3); t.add_argument('--weight-decay', type=float, default=1e-5)
    t.add_argument('--grad-clip', type=float, default=10.0)
    for action in ['evaluate', 'diagnose']:
        a = sub.add_parser(action); a.add_argument('run'); a.add_argument('--device', default='cpu')
        if action == 'evaluate':
            a.add_argument('--scope', choices=['pilot', 'full'], default='pilot')
            a.add_argument('--conditions', type=int, default=16); a.add_argument('--samples', type=int, default=256)
            a.add_argument('--max-proposals', type=int, default=20_000_000)
    for a in [t, *[sub.choices[k] for k in ['evaluate', 'diagnose']]]:
        a.add_argument('--seed', type=int, default=0); a.add_argument('--threads', type=int, default=4)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    {'train': train, 'evaluate': score_run, 'diagnose': diagnose}[args.command](args)
