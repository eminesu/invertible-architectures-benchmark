"""Evaluate FIF; FR3 results use metrics/results like the FMPE experiments."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.evaluate import RESULTS_DIR, evaluate
from metrics.ground_truth import GT_SIGMA
from models.fif import FIF


def load_run(run, device='cpu'):
    folder = Path(run)
    if not folder.is_dir():
        folder = ROOT / 'experiments' / 'runs' / run
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('method') != 'FIF':
        raise ValueError('Expected a FIF run configuration')
    if config.get('prior_mode') != 'fixed':
        raise ValueError('This FIF evaluator supports fixed benchmark priors only')
    model = FIF(**config['model']).to(device)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=device, weights_only=True))
    return model.eval(), config, folder


def new_evaluation_dir(folder):
    """Reserve a fresh directory without replacing any previous evaluation."""
    index = 1
    while True:
        out = folder / ('evaluation' if index == 1 else f'evaluation_{index}')
        try:
            out.mkdir()
        except FileExistsError:
            index += 1
        else:
            return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Run name or path, e.g. kin_fif_s0')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=('test', 'val'), default='test',
                        help='Select settings on val; report only the chosen setting on test')
    parser.add_argument('--name', help='Result name; default: the run name, with _val for validation')
    parser.add_argument('--out', type=Path,
                        help='Output directory; FR3 default: metrics/results; toy default: evaluation[_N] inside the run')
    parser.add_argument('--conditions', type=int, help='First M shared conditions; default: all 1000')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--gt-samples', type=int, default=1000)
    parser.add_argument('--max-proposals', type=int, help='Optional rejection-reference proposal cap')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.gt_samples < 2:
        parser.error('Need positive threads and at least two reference samples')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, config, folder = load_run(args.run, args.device)
    options = {}
    if args.gt_samples != 1000:
        options['n_gt'] = args.gt_samples
    if args.max_proposals is not None:
        options['max_proposals'] = args.max_proposals
    fr3 = config['problem'].startswith('fr3_')
    name = (args.name or config['run']) + ('_val' if args.split == 'val' else '')
    out = args.out
    if out is None:
        out = RESULTS_DIR if fr3 else new_evaluation_dir(folder)
    result_path = out / f'{config["problem"]}_{name}.json'
    if result_path.exists() or result_path.with_suffix('.npz').exists():
        raise FileExistsError(f'Evaluation already exists: {config["problem"]}_{name}; choose --name or --out')
    out.mkdir(parents=True, exist_ok=fr3 or args.out is None)
    print(f'Saving evaluation to {out}', flush=True)
    summary = evaluate(model.sample, config['problem'], name, n_samples=args.samples,
                       n_conditions=args.conditions, gt_options=options, out_dir=out, split=args.split)
    summary.update(prior_mode='fixed', family='FIF', method='conditional FIF',
                   arch=model.arch,
                   sampling_sigma=model.sigma, reference_sigma=GT_SIGMA.get(config['problem'], 0.0),
                   discrete_speed=model.discrete_speed, evaluation_seed=args.seed,
                   device=args.device, split=args.split, n_params=config['n_params'],
                   latent_dims=model.z_dims, conditioning_dims=model.y_dims)
    summary['training'] = {key: config.get(key) for key in
                           ('seed', 'n_train', 'n_val', 'epochs', 'batch_size', 'param_budget',
                            'lr', 'beta', 'layers', 'blocks')}
    # The evaluator's metric JSON is extended with FIF-specific provenance only.
    text = json.dumps(summary, indent=2)
    result_path.write_text(text, encoding='utf-8')
    if not fr3:
        (out / 'summary.json').write_text(text, encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
