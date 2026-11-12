"""Descriptive parameter-scaling fits from saved FR3 FIF/CFMPE evaluations.

No training, sampling, or cache generation. At least three distinct model sizes
are required. Fit E = A (N / 1e6)^b to the mean metric at each size, grouping
compatible protocols and averaging independent training seeds at each size.
These are fixed-data/epoch fits, not compute-optimal Kaplan exponents.

    python experiments/analyze_fr3_scaling.py experiments/runs metrics/results \
        --split val --metric pos_err_mm.median --out experiments/figures/shared/comparison/fr3_scaling.json

Writes JSON provenance and a PNG with a separate panel for each protocol.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def metric_value(record, dotted_name):
    value = record
    for part in dotted_name.split('.'):
        value = value[part]
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{dotted_name} must be finite and positive for a log-log fit')
    return float(value)


def protocol(record):
    training = {k: v for k, v in record['training'].items() if k not in ('seed', 'param_budget')}
    return {key: record.get(key) for key in
            ('benchmark', 'family', 'arch', 'split', 'n_conditions', 'n_samples', 'n_gt_min',
             'gt_options', 'evaluation_seed', 'prior_range', 'nominal_fraction',
             'sampling_sigma', 'reference_sigma', 'prior_params_evaluated')} | {
                 'training': training,
                 'sampler': [record['method'], record.get('steps')]}


def fit_group(records, metric):
    sizes = defaultdict(list)
    seen = set()
    for source, record in records:
        size, seed = int(record['n_params']), record['training']['seed']
        if size < 1:
            raise ValueError('Parameter counts must be positive')
        if (size, seed) in seen:
            raise ValueError(f'Duplicate size/seed evaluation at {source}; select one result per run')
        seen.add((size, seed))
        sizes[size].append((metric_value(record, metric), seed, str(source)))
    if len(sizes) < 3:
        raise ValueError('A scaling fit needs at least three distinct parameter counts')
    points = [{'n_params': size, 'metric_mean': float(np.mean([row[0] for row in rows])),
               'seeds': [row[1] for row in rows], 'sources': [row[2] for row in rows]}
              for size, rows in sorted(sizes.items())]
    if len({tuple(sorted(point['seeds'])) for point in points}) != 1:
        raise ValueError('Every model size must have the same set of training seeds')
    x = np.log([point['n_params'] / 1e6 for point in points])
    y = np.log([point['metric_mean'] for point in points])
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.square(y - (intercept + slope * x)).sum())
    total = float(np.square(y - y.mean()).sum())
    return {'protocol': protocol(records[0][1]), 'metric': metric,
            'form': 'E = A * (N / 1e6)^b', 'A': float(np.exp(intercept)), 'b': float(slope),
            'alpha': float(-slope), 'log_space_r2': 1 - residual / total if total > 0 else None,
            'points': points, 'interpretation': 'descriptive fixed-data/epoch parameter scaling; no compute-optimal claim',
            'single_seed': len(points[0]['seeds']) == 1}


def collect(paths, split):
    groups = defaultdict(list)
    files = set()
    for path in map(Path, paths):
        files.update(path.rglob('*.json') if path.is_dir() else [path])
    for path in sorted(files):
        if path.name == 'summary.json':
            continue  # FIF writes the identical record under its metric filename
        record = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(record, dict) or record.get('family') not in ('FIF', 'CFMPE')
                or not record.get('benchmark', '').startswith('fr3_')
                or record.get('split', 'test') != split):
            continue
        if not all(record.get('training', {}).get(key) is not None
                   for key in ('seed', 'n_train', 'n_val', 'epochs', 'batch_size')):
            raise ValueError(f'Missing training protocol in {path}; re-evaluate with the updated evaluator')
        key = json.dumps(protocol(record), sort_keys=True)
        groups[key].append((path, record))
    return groups


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', help='Result JSON files or directories searched recursively')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--metric', default='pos_err_mm.median',
                   choices=('pos_err_mm.median', 'pos_err_mm.mean', 'rot_err_deg.median',
                            'axis_err_deg.median', 'err_post.mean', 'err_resim.mean'))
    p.add_argument('--out', type=Path, required=True, help='New JSON file; the plot uses the same stem')
    args = p.parse_args(argv)
    if args.out.suffix.lower() != '.json':
        p.error('--out must have a .json extension')
    if args.out.exists() or args.out.with_suffix('.png').exists():
        p.error('Output already exists; choose a new --out path')
    fits, skipped = [], []
    for records in collect(args.paths, args.split).values():
        try:
            fits.append(fit_group(records, args.metric))
        except (KeyError, ValueError) as error:
            skipped.append({'sources': [str(path) for path, _ in records], 'reason': str(error)})
    if not fits:
        p.error('No compatible groups with three sizes: ' + json.dumps(skipped))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 4), squeeze=False)
    for ax, fit in zip(axes[0], fits):
        x = np.array([point['n_params'] for point in fit['points']])
        y = np.array([point['metric_mean'] for point in fit['points']])
        ax.loglog(x, y, 'o', label='mean over training seeds')
        ax.loglog(x, fit['A'] * (x / 1e6) ** fit['b'], '--', label=f'b = {fit["b"]:.3f}')
        label = fit['protocol']['family'] + (f' ({fit["protocol"]["arch"]})' if fit['protocol']['arch'] else '')
        ax.set(title=f'{fit["protocol"]["benchmark"]}: {label}', xlabel='Trainable parameters', ylabel=args.metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Descriptive parameter scaling at fixed data and epochs', fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    args.out.write_text(json.dumps({'fits': fits, 'skipped': skipped}, indent=2), encoding='utf-8')
    print(args.out)
    return fits


if __name__ == '__main__':
    main()
