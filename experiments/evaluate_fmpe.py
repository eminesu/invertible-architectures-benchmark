"""Score a trained FMPE run with the shared metrics (metrics.evaluate), at one or
more ODE solver settings (the NFE sweep).

    .venv/bin/python experiments/evaluate_fmpe.py kin_fmpe_s0 --solver rk4:8
    .venv/bin/python experiments/evaluate_fmpe.py kin_fmpe_s0 --solver euler:8 midpoint:8 rk4:8 rk4:32

Each --solver entry is method:steps and uses nfe(method, steps) network
evaluations per sample. Writes metrics/results/<benchmark>_<name>_<method>x<steps>.{json,npz};
the json gets the solver, NFE and parameter count added. Inference time is
measured on --device; use cpu to compare with the other models' numbers.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from metrics.evaluate import RESULTS_DIR, evaluate
from models.fmpe import FMPE, count_parameters, nfe

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_fmpe.py run from its config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    problem = MODELS[cfg['problem']]()
    model = FMPE(problem.n_parameters, problem.n_observations, cfg['hidden_resolved'], cfg['blocks'],
                 cfg['activation'], cfg['sigma_min'], cfg['alpha'])
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--solver', nargs='+', default=['rk4:8'], help='method:steps, e.g. rk4:8 (32 NFE)')
    p.add_argument('--name', default=None, help='result name prefix; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the x(0) draws')
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    for spec in args.solver:
        method, steps = spec.split(':')
        steps = int(steps)
        torch.manual_seed(args.seed)
        name = f'{args.name or args.run}_{method}x{steps}'
        print(f'--- {name}: {nfe(method, steps)} NFE per sample', flush=True)
        summary = evaluate(lambda y, n: model.sample(y, n, steps=steps, method=method), cfg['problem'], name=name)
        summary.update(method=method, steps=steps, nfe=nfe(method, steps), n_params=count_parameters(model),
                       device=args.device)
        (RESULTS_DIR / f"{cfg['problem']}_{name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
