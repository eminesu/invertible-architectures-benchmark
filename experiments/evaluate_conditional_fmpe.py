"""Score a trained prior-conditioned FMPE run with the shared metrics, once per
prior setting, against the ground truth under that prior.

    .venv/bin/python experiments/evaluate_conditional_fmpe.py bal_cfmpe_s0 --prior 12 15 18 --plain-baseline bal_fmpe_s0
    .venv/bin/python experiments/evaluate_conditional_fmpe.py kin_cfmpe_s0 \
        --prior 0.5,0.5,0.5 0.35,0.35,0.35 0.65,0.65,0.65 0.35,0.65,0.5 --plain-baseline kin_fmpe_s0

Each --prior entry is one setting of data.toy_data.PRIOR_PARAMS, comma-separated
(ballistics: lambda; kinematics: sigma1,sigma2,sigma3). For every setting the
model samples p(x | y*, c) on the committed 1000 test y*, and metrics.evaluate
compares with rejection samples under that prior (metrics/ground_truth.py; the
benchmark values give the benchmark ground truth). Writes
metrics/results/<benchmark>_<name>_<tag>_<method>x<steps>.{json,npz}, tag e.g.
lam12 or sig0.35-0.35-0.35. The y* are the benchmark's, drawn under the
benchmark prior, so under a narrower prior some y* are tail events (fewer
ground-truth samples; n_gt_min).

--plain-baseline also scores that plain FMPE run (trained on the benchmark prior
only) against each setting's ground truth: the error of ignoring the prior
change, which the conditional model should beat off the benchmark prior. Written
as <benchmark>_<plain run>_vs_<tag>_<method>x<steps>.{json,npz}.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS, PRIOR_PARAMS, prior_tag
from experiments.evaluate_fmpe import load_run as load_plain_run
from metrics.evaluate import RESULTS_DIR, evaluate
from models.conditional_fmpe import ConditionalFMPE
from models.fmpe import count_parameters, nfe

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_conditional_fmpe.py run from config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    if 'prior_range' not in cfg:  # runs from before kinematics support (ballistics, lambda only)
        cfg['prior_range'] = [cfg['lam_min'], cfg['lam_max']]
    problem = MODELS[cfg['problem']]()
    model = ConditionalFMPE(problem.n_parameters, problem.n_observations, cfg['hidden_resolved'], cfg['blocks'],
                            cfg['activation'], cfg['sigma_min'], cfg['alpha'],
                            len(PRIOR_PARAMS[cfg['problem']]['names']))
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def score(sample_fn, problem, name, c, n_params, method, steps, device, **extra):
    summary = evaluate(sample_fn, problem, name=name, prior_params=c)
    summary.update(method=method, steps=steps, nfe=nfe(method, steps), n_params=n_params, device=device, **extra)
    (RESULTS_DIR / f'{problem}_{name}.json').write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--prior', nargs='+', required=True, help='prior settings, e.g. 12 or 0.35,0.35,0.35')
    p.add_argument('--solver', nargs='+', default=['midpoint:16'], help='method:steps, e.g. midpoint:16 (32 NFE)')
    p.add_argument('--name', default=None, help='result name prefix; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the x(0) draws')
    p.add_argument('--plain-baseline', default=None, help='plain FMPE run to score against every prior setting')
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    problem = cfg['problem']
    plain = load_plain_run(args.plain_baseline, args.device)[0] if args.plain_baseline else None
    for setting in args.prior:
        c = [float(v) for v in setting.split(',')]
        assert len(c) == model.prior_dims, f'{problem} takes {model.prior_dims} prior parameters'
        tag = prior_tag(problem, c)
        for spec in args.solver:
            method, steps = spec.split(':')
            steps = int(steps)
            if plain is not None:
                torch.manual_seed(args.seed)
                name = f'{args.plain_baseline}_vs_{tag}_{method}x{steps}'
                print(f'--- {name}: plain FMPE (benchmark prior) vs. the {tag} ground truth', flush=True)
                score(lambda y, n: plain.sample(y, n, steps=steps, method=method), problem, name, c,
                      count_parameters(plain), method, steps, args.device)
            torch.manual_seed(args.seed)
            name = f'{args.name or args.run}_{tag}_{method}x{steps}'
            print(f'--- {name}: {nfe(method, steps)} NFE per sample', flush=True)
            score(lambda y, n: model.sample(y, c, n, steps=steps, method=method), problem, name, c,
                  count_parameters(model), method, steps, args.device, prior_range=cfg['prior_range'])


if __name__ == '__main__':
    main()
