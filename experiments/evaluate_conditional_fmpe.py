"""Score a trained prior-conditioned FMPE run with the shared metrics, once per
prior lambda, against the ground truth under that prior.

    .venv/bin/python experiments/evaluate_conditional_fmpe.py bal_cfmpe_s0 --lam 12 15 18 --solver midpoint:16

For every lambda the model samples p(x | y*, lambda) on the committed 1000 test
y*, and metrics.evaluate compares with rejection samples from the prior
v0 ~ Poisson(lambda) (metrics/ground_truth.py; lambda = 15 is the benchmark
ground truth). Writes metrics/results/ballistics_<name>_lam<lambda>_<method>x<steps>.{json,npz}.
The y* are the benchmark's, drawn under lambda = 15, so under lambda = 12 the
far impacts are tail events of the prior (fewer ground-truth samples; n_gt_min).

--plain-baseline bal_fmpe_s0 also scores that plain FMPE run (trained on lambda
= 15 only) against each lambda's ground truth: the error of ignoring the prior
change, which the conditional model should beat at lambda != 15. Written as
ballistics_<plain run>_vs_lam<lambda>_<method>x<steps>.{json,npz}.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics.evaluate import RESULTS_DIR, evaluate
from experiments.evaluate_fmpe import load_run as load_plain_run
from models.conditional_fmpe import ConditionalFMPE
from models.fmpe import count_parameters, nfe

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_conditional_fmpe.py run from config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    model = ConditionalFMPE(4, 1, cfg['hidden_resolved'], cfg['blocks'], cfg['activation'], cfg['sigma_min'],
                            cfg['alpha'])
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--lam', type=float, nargs='+', default=[12, 15, 18])
    p.add_argument('--solver', nargs='+', default=['midpoint:16'], help='method:steps, e.g. midpoint:16 (32 NFE)')
    p.add_argument('--name', default=None, help='result name prefix; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the x(0) draws')
    p.add_argument('--plain-baseline', default=None, help='plain FMPE run to score against every lambda')
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    plain = load_plain_run(args.plain_baseline, args.device)[0] if args.plain_baseline else None
    for lam in args.lam:
        for spec in args.solver:
            if plain is not None:
                method, steps = spec.split(':')
                steps = int(steps)
                torch.manual_seed(args.seed)
                name = f'{args.plain_baseline}_vs_lam{lam:g}_{method}x{steps}'
                print(f'--- {name}: plain FMPE (lambda = 15 prior) vs. lambda = {lam:g} ground truth', flush=True)
                summary = evaluate(lambda y, n: plain.sample(y, n, steps=steps, method=method), 'ballistics',
                                   name=name, lam=lam)
                summary.update(method=method, steps=steps, nfe=nfe(method, steps), n_params=count_parameters(plain),
                               device=args.device)
                (RESULTS_DIR / f'ballistics_{name}.json').write_text(json.dumps(summary, indent=2))
            method, steps = spec.split(':')
            steps = int(steps)
            torch.manual_seed(args.seed)
            name = f'{args.name or args.run}_lam{lam:g}_{method}x{steps}'
            print(f'--- {name}: lambda = {lam:g}, {nfe(method, steps)} NFE per sample', flush=True)
            summary = evaluate(lambda y, n: model.sample(y, lam, n, steps=steps, method=method), 'ballistics',
                               name=name, lam=lam)
            summary.update(method=method, steps=steps, nfe=nfe(method, steps), n_params=count_parameters(model),
                           device=args.device, lam_range=[cfg['lam_min'], cfg['lam_max']])
            (RESULTS_DIR / f'ballistics_{name}.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
