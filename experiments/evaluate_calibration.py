"""Is a trained model over-confident, under-confident or calibrated?

    .venv/bin/python experiments/evaluate_calibration.py inn  kin_cinn_s0
    .venv/bin/python experiments/evaluate_calibration.py fmpe kin_fmpe_s0 --solver rk4:8
    .venv/bin/python experiments/evaluate_calibration.py mdn  kin_K16_s0
    .venv/bin/python experiments/evaluate_calibration.py fmpe kin_fmpe_s0 --no-gt   # skip rejection sampling

Loads a run (same loaders as evaluate_<model>.py), samples 1000 posterior draws
for each of the 1000 shared test y*, and writes
metrics/results/<benchmark>_<name>_calibration.{json,npz}. See
metrics/calibration.py for what is measured and how to read it.

    ... --plot fig.png   additionally compares several already-scored runs:
    .venv/bin/python experiments/evaluate_calibration.py plot kinematics fig.png kin_cinn_s0 kin_fmpe_s0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from metrics.calibration import calibration, plot, save
from metrics.evaluate import RESULTS_DIR

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_mdn(run, device):
    from models.mdn import MDN
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    problem = MODELS[cfg['problem']]()
    model = MDN(problem.n_observations, problem.n_parameters, cfg['components'], cfg['hidden_resolved'],
                cfg['layers'], cfg['activation'])
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def sampler(kind, run, device, solver):
    if kind == 'inn':
        from experiments.evaluate_inn import load_run
        model, cfg = load_run(run, device)
        return (lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32, device=device), n)), cfg
    if kind == 'fmpe':
        from experiments.evaluate_fmpe import load_run
        model, cfg = load_run(run, device)
        method, steps = solver.split(':')
        return (lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32, device=device), n,
                                          steps=int(steps), method=method)), cfg
    if kind == 'mdn':
        model, cfg = load_mdn(run, device)
        return (lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32, device=device), n)), cfg
    raise SystemExit(f'unknown model kind {kind}; for NPSE wrap NPSE.sample with metrics.calibration.calibration()')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        _, _, benchmark, out, *runs = sys.argv
        reps = {}
        for r in runs:
            d = np.load(RESULTS_DIR / f'{benchmark}_{r}_calibration.npz')
            rep = json.load(open(RESULTS_DIR / f'{benchmark}_{r}_calibration.json'))
            rep['_arrays'] = {k: d[k] for k in d.files}
            reps[r] = rep
        plot(reps, out)
        return

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=['inn', 'fmpe', 'mdn'])
    p.add_argument('run')
    p.add_argument('--solver', default='rk4:8', help='fmpe only, method:steps')
    p.add_argument('--name', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-gt', action='store_true', help='skip the spread-vs-ground-truth check')
    args = p.parse_args()

    fn, cfg = sampler(args.kind, args.run, args.device, args.solver)
    torch.manual_seed(args.seed)
    name = args.name or args.run
    rep = calibration(fn, cfg['problem'], name=name, with_ground_truth=not args.no_gt)
    save(rep, RESULTS_DIR, cfg['problem'], name)


if __name__ == '__main__':
    main()