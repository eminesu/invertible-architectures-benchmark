"""Is a trained model over-confident, under-confident or calibrated?

    python experiments/evaluate_calibration.py list                       # runs with weights and the model kind found
    python experiments/evaluate_calibration.py auto kin_fif_s0            # kind detected from config.json / run name
    python experiments/evaluate_calibration.py auto kin_fmpe_s0 --solver midpoint:16
    python experiments/evaluate_calibration.py auto kin_cinn_s0 --no-gt   # skip the ground-truth spread check
    python experiments/evaluate_calibration.py plots                      # plots of ALL finished runs
    python experiments/evaluate_calibration.py plot kinematics fig.png kin_cinn_s0 kin_fmpe_s0_midpointx16

Kinds: inn (INN / cINN), ilnn (invertible liquid NN), fmpe, cfmpe (prior-conditional FMPE, scored at the
default benchmark prior), mdn, ae (autoencoder / cVAE / InvAuto), fff (FFF / mFFF), fif. BayesFlow NPSE
runs are not handled here (call metrics.calibration.calibration() with their sample function).

Samples 1000 posterior draws for each of the 1000 shared test y* and writes
metrics/results/<benchmark>_<name>_calibration.{json,npz}. See metrics/calibration.py for what is
measured and how to read it.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
KINDS = ['inn', 'ilnn', 'fmpe', 'cfmpe', 'mdn', 'ae', 'fff', 'fif']
SOLVER_KINDS = ('fmpe', 'cfmpe')
DEFAULT_SOLVER = 'midpoint:16'  # 32 NFE, same cost as rk4:8


def read_config(run):
    return json.load(open(os.path.join(RUNS, run, 'config.json')))


def detect_kind(run, cfg=None):
    """Model kind of a run from its config.json keys, then from tokens of the run name.
    Returns None for runs this script cannot score (e.g. BayesFlow NPSE)."""
    cfg = read_config(run) if cfg is None else cfg
    tokens = set(run.split('_'))
    if 'bf' in tokens or 'npse' in tokens:
        return None
    if 'ilnn' in tokens or 'ilnn' in str(cfg.get('model', '')).lower():
        return 'ilnn'
    if 'cfmpe' in tokens or 'prior_names' in cfg or 'prior_range' in cfg or 'lam_min' in cfg:
        return 'cfmpe'  # older ballistics runs only have lam_min / lam_max
    if 'sigma_min' in cfg:
        return 'fmpe'
    if 'family' in cfg or cfg.get('model') == 'autoencoder':
        return 'ae'
    if 'components' in cfg:
        return 'mdn'
    if 'manifold' in cfg:
        return 'fff'
    if cfg.get('method') == 'FIF':
        return 'fif'
    if cfg.get('model') in ('inn', 'cinn'):
        return 'inn'
    for tok in tokens:  # name fallbacks
        if tok in ('inn', 'cinn'):
            return 'inn'
        if re.fullmatch(r'K\d+', tok):
            return 'mdn'
        if tok in ('fff', 'mfff', 'fif', 'fmpe', 'cfmpe'):
            return {'mfff': 'fff'}.get(tok, tok)
        if tok in ('standard', 'cvae', 'invertible'):
            return 'ae'
    return None


def list_runs():
    out = []
    for run in sorted(os.listdir(RUNS)):
        folder = os.path.join(RUNS, run)
        if not (os.path.isfile(os.path.join(folder, 'model.pt')) and os.path.isfile(os.path.join(folder, 'config.json'))):
            continue
        try:
            cfg = read_config(run)
        except Exception as e:  # noqa: BLE001
            print(f'# skip {run}: unreadable config ({e})', file=sys.stderr)
            continue
        kind = detect_kind(run, cfg)
        if kind is None:
            print(f'# skip {run}: no calibration loader (BayesFlow or unknown kind)', file=sys.stderr)
            continue
        out.append((run, kind, cfg.get('problem')))
    return out


# ---- loaders: each returns (sample_fn(y_np_float32, n) -> (M, n, 4), cfg) --------------------------

def _t(y, device):
    import torch
    return torch.as_tensor(y, dtype=torch.float32, device=device)


def build_sampler(kind, run, device, solver):
    import torch
    if kind in ('inn', 'ilnn'):
        if kind == 'ilnn':
            try:
                from experiments.evaluate_ilnn import load_run  # add this file with the ILNN code (same API as evaluate_inn)
            except ImportError:
                from experiments.evaluate_inn import load_run  # works if train_inn.build_model knows the ILNN
        else:
            from experiments.evaluate_inn import load_run
        model, cfg = load_run(run, device)
        return (lambda y, n: model.sample(_t(y, device), n)), cfg
    if kind == 'fmpe':
        from experiments.evaluate_fmpe import load_run
        model, cfg = load_run(run, device)
        method, steps = solver.split(':')
        return (lambda y, n: model.sample(_t(y, device), n, steps=int(steps), method=method)), cfg
    if kind == 'cfmpe':
        from data.toy_data import PRIOR_PARAMS
        from experiments.evaluate_conditional_fmpe import load_run
        model, cfg = load_run(run, device)
        c = np.asarray(PRIOR_PARAMS[cfg['problem']]['default'], dtype=np.float32)  # benchmark prior
        method, steps = solver.split(':')
        return (lambda y, n: model.sample(_t(y, device), c, n, steps=int(steps), method=method)), cfg
    if kind == 'mdn':
        from data.toy_data import MODELS
        from models.mdn import MDN
        cfg = read_config(run)
        problem = MODELS[cfg['problem']]()
        model = MDN(problem.n_observations, problem.n_parameters, cfg['components'], cfg['hidden_resolved'],
                    cfg['layers'], cfg['activation'])
        model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
        model = model.to(device).eval()
        return (lambda y, n: model.sample(_t(y, device), n)), cfg
    if kind == 'ae':
        from models.autoencoder_family import build_model
        cfg = read_config(run)
        cfg.setdefault('family', 'standard')  # older kin_ae_s0 configs have no family
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device, weights_only=True))
        model.eval()
        return (lambda y, n: model.sample(y, n)), cfg
    if kind == 'fff':
        from experiments.evaluate_fff import load_run
        model, cfg = load_run(run, device)
        return (lambda y, n: model.sample(y, n)), cfg
    if kind == 'fif':
        from experiments.evaluate_fif import load_run
        model, cfg, _ = load_run(run, device)
        return (lambda y, n: model.sample(y, n)), cfg
    raise SystemExit(f'unknown kind {kind}')


def main():
    from_cli = sys.argv[1:]
    if from_cli and from_cli[0] == 'list':
        for run, kind, prob in list_runs():
            print(f'{run}\t{kind}\t{prob}')
        return
    if from_cli and from_cli[0] == 'plots':
        from metrics.calibration import plot_summary
        from metrics.evaluate import RESULTS_DIR
        files = plot_summary(RESULTS_DIR)
        print('\n'.join(files) if files else 'no calibration results found')
        return
    if from_cli and from_cli[0] == 'plot':
        import metrics.calibration as C
        from metrics.evaluate import RESULTS_DIR
        _, benchmark, out, *runs = from_cli
        reps = {}
        for r in runs:
            d = np.load(RESULTS_DIR / f'{benchmark}_{r}_calibration.npz')
            rep = json.load(open(RESULTS_DIR / f'{benchmark}_{r}_calibration.json'))
            rep['_arrays'] = {k: d[k] for k in d.files}
            reps[r] = rep
        C.plot(reps, out)
        return

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=['auto'] + KINDS)
    p.add_argument('run')
    p.add_argument('--solver', default=DEFAULT_SOLVER, help='fmpe / cfmpe only, method:steps')
    p.add_argument('--name', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-gt', action='store_true', help='skip the spread-vs-ground-truth check')
    args = p.parse_args()

    import torch
    from metrics.calibration import calibration, save
    from metrics.evaluate import RESULTS_DIR

    kind = detect_kind(args.run) if args.kind == 'auto' else args.kind
    if kind is None:
        raise SystemExit(f'{args.run}: cannot detect a supported model kind')
    fn, cfg = build_sampler(kind, args.run, args.device, args.solver)
    torch.manual_seed(args.seed)
    name = args.name or args.run
    rep = calibration(fn, cfg['problem'], name=name, with_ground_truth=not args.no_gt)
    save(rep, RESULTS_DIR, cfg['problem'], name)


if __name__ == '__main__':
    main()