"""FR3 FIF/CFMPE learning curves or seven-joint posterior marginals.

Learning curves read saved files only. --posterior samples the trained model
and loads the exact benchmark reference (generating it if missing).

    python experiments/plot_fr3_fif_cfmpe.py fr3_6d_fif_s0 fr3_6d_cfmpe_s0 --out curves.png
    python experiments/plot_fr3_fif_cfmpe.py fr3_6d_fif_s0 --posterior --split val --out posterior.png
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_run(run):
    folder = Path(run)
    if not folder.is_dir():
        folder = ROOT / 'experiments' / 'runs' / run
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if not cfg['problem'].startswith('fr3_'):
        raise ValueError('This plotter supports FR3 tasks only')
    family = 'FIF' if cfg.get('method') == 'FIF' else 'CFMPE' if 'prior_range' in cfg else None
    if family is None:
        raise ValueError(f'Expected a FIF or CFMPE run at {folder}')
    return folder, cfg, family


def learning_curves(runs, plt):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        folder, cfg, family = read_run(run)
        with (folder / 'log.csv').open(newline='', encoding='utf-8') as source:
            rows = list(csv.DictReader(source))
        epoch = np.array([int(row['epoch']) for row in rows])
        if epoch[0] == 0:
            epoch += 1  # CFMPE logs zero-based epochs; FIF logs one-based epochs
        train = np.array([float(row['train_loss']) for row in rows])
        valid = np.array([float(row['val_loss']) for row in rows])
        ax.plot(epoch, train, label='train')
        ax.plot(epoch, valid, '--', label='validation')
        ax.set_xscale('log')
        if np.all(train > 0) and np.all(valid > 0):
            ax.set_yscale('log')
        ax.set(title=f'{cfg["problem"]}: {family}\n{folder.name}', xlabel='Epoch',
               ylabel='FIF surrogate (not log density)' if family == 'FIF' else 'Flow-matching loss')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    return fig


def posterior(args, plt):
    import torch
    from experiments.fr3_fif_cfmpe_adapters import PRIOR_SPECS
    from metrics.benchmarks import load_test_conditions
    from metrics.ground_truth import ground_truth_posterior
    from robot_ik.kinematics import joint_limits

    folder, cfg, family = read_run(args.runs[0])
    torch.manual_seed(args.seed)
    if family == 'FIF':
        from experiments.evaluate_fif import load_run
        model, _, _ = load_run(folder, args.device)
        sample = model.sample
    else:
        from experiments.evaluate_conditional_fmpe import load_run
        model, _ = load_run(folder, args.device)
        method, step_text = args.solver.split(':')
        sample = lambda y, n: model.sample(y, PRIOR_SPECS[cfg['problem']]['default'], n,
                                           method=method, steps=int(step_text), chunk=args.chunk)
    y, _ = load_test_conditions(cfg['problem'], args.split)
    if any(index < 0 or index >= len(y) for index in args.index):
        raise ValueError('Posterior indices must belong to the selected split')
    gt = ground_truth_posterior(cfg['problem'], y, verbose=False)
    draws = sample(y[args.index], args.samples).detach().cpu().numpy()
    lo_lim, hi_lim = joint_limits()
    fig, axes = plt.subplots(len(args.index), 7, figsize=(22, 3 * len(args.index)), squeeze=False)
    for row, index, x in zip(axes, args.index, draws):
        violation = ((x < lo_lim) | (x > hi_lim)).any(1).mean()
        for d, ax in enumerate(row):
            bounds = np.quantile(np.concatenate([x[:, d], gt[index][:, d]]), [0.002, 0.998])
            bins = np.linspace(min(bounds[0], lo_lim[d]), max(bounds[1], hi_lim[d]), 60)
            ax.hist(gt[index][:, d], bins, density=True, alpha=0.4, color='gray', label='Exact ground truth')
            ax.hist(x[:, d], bins, density=True, histtype='step', color='tab:blue', label=family)
            ax.axvline(lo_lim[d], color='tab:red', ls='--', lw=1)
            ax.axvline(hi_lim[d], color='tab:red', ls='--', lw=1)
            if cfg['problem'] == 'fr3_5d' and d == 6:
                ax.axhline(1 / (hi_lim[d] - lo_lim[d]), color='black', ls=':', label='Uniform prior')
            ax.set(xlabel=f'Joint {d + 1} (rad)', yticks=[])
        row[0].set_ylabel(f'{args.split} pose #{index}\nOutside limits: {violation:.1%}')
        row[0].legend(fontsize=8)
    fig.suptitle(f'{cfg["problem"]}: {family} joint marginals, {folder.name}')
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--posterior', action='store_true')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--index', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--solver', default='euler:64')
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args(argv)
    if args.out.exists():
        p.error('Output already exists; choose a new --out path')
    if args.posterior and len(args.runs) != 1:
        p.error('--posterior takes exactly one run')
    if min(args.samples, args.chunk) < 1:
        p.error('--samples and --chunk must be positive')
    method, separator, steps = args.solver.partition(':')
    if method not in ('euler', 'midpoint', 'rk4') or separator != ':' or not steps.isdigit() or int(steps) < 1:
        p.error('Invalid --solver; use euler:N, midpoint:N or rk4:N')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = posterior(args, plt) if args.posterior else learning_curves(args.runs, plt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
