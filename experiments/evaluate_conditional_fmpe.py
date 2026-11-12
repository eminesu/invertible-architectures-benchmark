"""Score a trained prior-conditioned FMPE run with the shared metrics, once per
prior setting, against the ground truth under that prior.

    .venv/bin/python experiments/evaluate_conditional_fmpe.py bal_cfmpe_s0 --prior 12 15 18 --plain-baseline bal_fmpe_s0
    .venv/bin/python experiments/evaluate_conditional_fmpe.py kin_cfmpe_s0 \
        --prior 0.5,0.5,0.5 0.35,0.35,0.35 0.65,0.65,0.65 0.35,0.65,0.5 --plain-baseline kin_fmpe_s0

Each --prior entry is one setting of the experiment-local PRIOR_SPECS, comma-separated
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

FR3: --prior may be omitted to use all seven widths = 1. Only this prior has
an exact reference. Choose samplers with --split val before reporting on test.
--chunk bounds ODE memory; outputs are never overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from experiments.fr3_fif_cfmpe_adapters import (
    PRIOR_SPECS, conditional_model_class, evaluation_prior_params, prior_tag,
)
from experiments.evaluate_fmpe import load_run as load_plain_run
from experiments.evaluate_fmpe import redraw_nonfinite
from metrics.evaluate import RESULTS_DIR, evaluate
from models.fmpe import count_parameters, nfe

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_conditional_fmpe.py run from config.json + model.pt."""
    folder = Path(run)
    if not folder.is_dir():
        folder = Path(RUNS) / run
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if 'prior_range' not in cfg:  # runs from before kinematics support (ballistics, lambda only)
        cfg['prior_range'] = [cfg['lam_min'], cfg['lam_max']]
    problem = MODELS[cfg['problem']]()
    model = conditional_model_class(cfg['problem'])(
        problem.n_parameters, problem.n_observations, cfg['hidden_resolved'], cfg['blocks'],
        cfg['activation'], cfg['sigma_min'], cfg['alpha'], len(PRIOR_SPECS[cfg['problem']]['names']))
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=device, weights_only=True))
    return model.to(device).eval(), cfg


def score(sample_fn, problem, name, c, n_params, method, steps, device, *, split='test',
          n_samples=1000, n_conditions=None, out_dir=RESULTS_DIR, gt_options=None, counter=None, **extra):
    # The exact FR3 posterior is currently available only at the nominal prior.
    # Do not relabel this as ground truth under narrower joint limits.
    prior = evaluation_prior_params(problem, c)
    out_dir = Path(out_dir)
    if (out_dir / f'{problem}_{name}.json').exists() or (out_dir / f'{problem}_{name}.npz').exists():
        raise FileExistsError(f'Evaluation already exists: {problem}_{name}; choose --name or --out')
    summary = evaluate(sample_fn, problem, name=name, prior_params=prior, split=split,
                       n_samples=n_samples, n_conditions=n_conditions, out_dir=out_dir, gt_options=gt_options)
    summary.update(method=method, steps=steps, nfe=nfe(method, steps), n_params=n_params,
                   device=device, split=split, **extra)
    if counter is not None:
        summary['n_redrawn_nonfinite'] = counter['n']
    (out_dir / f'{problem}_{name}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--prior', nargs='+', help='prior settings; omitted = benchmark prior. FR3 accepts only 1,1,1,1,1,1,1')
    p.add_argument('--solver', nargs='+', default=['midpoint:16'], help='method:steps, e.g. midpoint:16 (32 NFE)')
    p.add_argument('--name', default=None, help='result name prefix; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the x(0) draws')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--conditions', type=int, default=None, help='First M shared poses; omitted = full split')
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--gt-samples', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=8192, help='ODE states per chunk; bounds GPU memory')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=RESULTS_DIR)
    p.add_argument('--redraw-nonfinite', action='store_true', help='Replace diverged samples and record the count')
    p.add_argument('--plain-baseline', default=None, help='plain FMPE run to score against every prior setting')
    args = p.parse_args(argv)
    if min(args.samples, args.gt_samples) < 2 or min(args.chunk, args.threads) < 1:
        p.error('Need at least two model/reference samples, and positive chunk/threads')
    torch.set_num_threads(args.threads)

    model, cfg = load_run(args.run, args.device)
    problem = cfg['problem']
    plain = load_plain_run(args.plain_baseline, args.device)[0] if args.plain_baseline else None
    if plain is not None and (plain.x_dims != model.x_dims or plain.y_dims != model.obs_dims):
        p.error('--plain-baseline must use the same benchmark dimensions')
    settings = ([list(PRIOR_SPECS[problem]['default'])] if args.prior is None
                else [[float(v) for v in setting.split(',')] for setting in args.prior])
    if any(len(c) != model.prior_dims or not np.isfinite(c).all() for c in settings):
        p.error(f'{problem} takes {model.prior_dims} finite prior parameters')
    if problem.startswith('fr3_') and any(not np.array_equal(c, PRIOR_SPECS[problem]['default']) for c in settings):
        p.error('FR3 ground truth supports only seven width scales equal to 1')
    solvers = []
    for spec in args.solver:
        try:
            method, step_text = spec.split(':')
            steps = int(step_text)
            if steps < 1:
                raise ValueError
            nfe(method, steps)
        except (ValueError, KeyError):
            p.error(f'Invalid solver {spec!r}; use euler:N, midpoint:N or rk4:N with positive N')
        solvers.append((method, steps))
    summaries = []
    for c in settings:
        tag = prior_tag(problem, c)
        for method, steps in solvers:
            suffix = f'_{method}x{steps}' + ('_val' if args.split == 'val' else '')
            options = dict(split=args.split, n_samples=args.samples, n_conditions=args.conditions,
                           out_dir=args.out, gt_options={'n_gt': args.gt_samples} if args.gt_samples != 1000 else None,
                           evaluation_seed=args.seed, prior_params_evaluated=c)
            if plain is not None:
                torch.manual_seed(args.seed)
                name = f'{Path(args.plain_baseline).name}_vs_{tag}{suffix}'
                print(f'--- {name}: plain FMPE (benchmark prior) vs. the {tag} ground truth', flush=True)
                counter = {'n': 0}
                bound_plain = SimpleNamespace(sample=lambda y, n, **kw: plain.sample(y, n, chunk=args.chunk, **kw))
                sample = (redraw_nonfinite(bound_plain, steps, method, counter) if args.redraw_nonfinite
                          else lambda y, n: plain.sample(y, n, steps=steps, method=method, chunk=args.chunk))
                summaries.append(score(sample, problem, name, c, count_parameters(plain), method, steps,
                                       args.device, counter=counter if args.redraw_nonfinite else None, **options))
            torch.manual_seed(args.seed)
            name = f'{args.name or Path(args.run).name}_{tag}{suffix}'
            print(f'--- {name}: {nfe(method, steps)} NFE per sample', flush=True)
            counter = {'n': 0}
            bound = SimpleNamespace(sample=lambda y, n, **kw: model.sample(y, c, n, chunk=args.chunk, **kw))
            sample = (redraw_nonfinite(bound, steps, method, counter) if args.redraw_nonfinite
                      else lambda y, n: model.sample(y, c, n, steps=steps, method=method, chunk=args.chunk))
            summaries.append(score(sample, problem, name, c, count_parameters(model), method, steps, args.device,
                                   counter=counter if args.redraw_nonfinite else None,
                                   family='CFMPE', prior_range=cfg['prior_range'],
                                   nominal_fraction=cfg.get('nominal_fraction', 0.0),
                                   training={key: cfg.get(key) for key in
                                             ('seed', 'n_train', 'n_val', 'epochs', 'batch_size', 'param_budget',
                                              'lr', 'weight_decay', 'blocks', 'alpha', 'sigma_min')},
                                   **options))
    return summaries


if __name__ == '__main__':
    main()
