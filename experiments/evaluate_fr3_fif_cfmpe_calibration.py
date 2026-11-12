"""Calibrate FR3 FIF/CFMPE through an experiment-local sampler adapter.

Consumes metrics.calibration unchanged. The CFMPE prior is supplied by the
local FR3 registry, without extending the shared toy prior registry.

    python experiments/evaluate_fr3_fif_cfmpe_calibration.py fif fr3_6d_fif_s0
    python experiments/evaluate_fr3_fif_cfmpe_calibration.py cfmpe fr3_6d_cfmpe_s0 --solver euler:64
"""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fr3_fif_cfmpe_adapters import FR3_CONSTRAINT_DIMS, PRIOR_SPECS
from metrics.calibration import calibration, save
from metrics.evaluate import RESULTS_DIR
from models.fmpe import nfe


def build_sampler(method, run, device, solver='euler:64', chunk=8192):
    if method == 'fif':
        from experiments.evaluate_fif import load_run
        model, cfg, _ = load_run(run, device)
        sample = model.sample
    else:
        from experiments.evaluate_conditional_fmpe import load_run
        model, cfg = load_run(run, device)
        ode_method, steps = solver.split(':')
        c = PRIOR_SPECS[cfg['problem']]['default']
        sample = lambda y, n: model.sample(y, c, n, steps=int(steps), method=ode_method, chunk=chunk)
    if cfg['problem'] not in FR3_CONSTRAINT_DIMS:
        raise ValueError('This calibration adapter supports FR3 tasks only')
    return sample, cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('method', choices=('fif', 'cfmpe'))
    p.add_argument('run')
    p.add_argument('--solver', default='euler:64', help='Use the CFMPE solver selected on validation')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default=None)
    p.add_argument('--out', type=Path, default=RESULTS_DIR)
    p.add_argument('--no-gt', action='store_true', help='Skip the reference spread comparison')
    args = p.parse_args(argv)
    if min(args.chunk, args.threads) < 1 or args.samples < 2:
        p.error('Need positive chunk/threads and at least two samples')
    try:
        method, steps = args.solver.split(':')
        if int(steps) < 1:
            raise ValueError
        nfe(method, int(steps))
    except (ValueError, KeyError):
        p.error('Invalid --solver; use euler:N, midpoint:N or rk4:N')
    name = args.name or Path(args.run).name
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        p.error('--name must be a filename component')
    torch.set_num_threads(args.threads)
    sample, cfg = build_sampler(args.method, args.run, args.device, args.solver, args.chunk)
    stem = f'{cfg["problem"]}_{name}_calibration'
    if any((args.out / f'{stem}{suffix}').exists() for suffix in ('.json', '.npz')):
        p.error('Calibration output already exists; choose --name or --out')
    torch.manual_seed(args.seed)
    report = calibration(sample, cfg['problem'], name=name, n_samples=args.samples,
                         with_ground_truth=not args.no_gt)
    report.update(method_family=args.method.upper(), evaluation_seed=args.seed, device=args.device,
                  solver=args.solver if args.method == 'cfmpe' else None)
    save(report, args.out, cfg['problem'], name)
    return report


if __name__ == '__main__':
    main()
