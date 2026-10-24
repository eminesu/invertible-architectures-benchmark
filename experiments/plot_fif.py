"""Plot saved FIF errors for both benchmarks; never runs training or evaluation."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=('kinematics', 'ballistics', 'both'), default='both')
    parser.add_argument('--out', type=Path, help='Output PNG; defaults to docs/fif-results/fif.png for both problems')
    args = parser.parse_args()
    problems = ('kinematics', 'ballistics') if args.problem == 'both' else (args.problem,)
    name = {'both': 'fif', 'kinematics': 'kin', 'ballistics': 'bal'}[args.problem]
    out = ROOT / (args.out or Path(f'docs/fif-results/{name}.png'))
    records, protocol = [], None
    for problem in problems:
        prefix = 'kin' if problem == 'kinematics' else 'bal'
        run = f'{prefix}_fif_s0'
        source = ROOT / 'experiments' / 'runs' / run / 'evaluation' / 'summary.json'
        result = json.loads(source.read_text(encoding='utf-8'))
        current = {key: result[key] for key in ('n_conditions', 'n_samples', 'evaluation_seed')}
        if (result['benchmark'] != problem or result['model'] != run
                or result.get('prior_mode') != 'fixed'
                or (protocol is not None and current != protocol)):
            raise ValueError(f'Incompatible evaluation settings in {source}')
        protocol = current
        values = [result[key]['mean'] for key in ('err_post', 'err_resim')]
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError(f'Mean errors must be finite in {source}')
        records.append({'problem': problem, 'run': run,
                        'source': source.relative_to(ROOT).as_posix(), 'evaluation': result})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(records), 2, figsize=(10, 3.2 * len(records) + 0.6),
                             squeeze=False, sharey='col')
    for col, metric in enumerate(('err_post', 'err_resim')):
        values = [record['evaluation'][metric]['mean'] for record in records]
        positive = min(values) > 0
        for row, (record, value) in enumerate(zip(records, values)):
            ax = axes[row, col]
            ax.scatter([0], [value], c='#1f77b4', s=65, zorder=3)
            if positive:
                ax.set_yscale('log')
                ax.set_ylim(min(values) * 0.65, max(values) * 2.5)
            else:
                ax.set_yscale('symlog', linthresh=1e-4)
            ax.annotate(f'{value:.6g}', (0, value), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=10)
            ax.set_xticks([0], [record['run']])
            ax.set_xlim(-0.5, 0.5)
            ax.set_title(f'{record["problem"]}: {metric} (mean)')
            ax.set_ylabel('Mean error (log scale)' if positive else 'Mean error (symmetric-log scale)')
            ax.grid(axis='y', which='both', alpha=0.15)
    fig.suptitle('FIF evaluation results', fontsize=13)
    fig.text(0.5, 0.015, f'Saved evaluations | {protocol["n_conditions"]:,} conditions x '
             f'{protocol["n_samples"]:,} samples | seed {protocol["evaluation_seed"]} | lower is better',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    snapshot = {'protocol': protocol, 'err_post_estimator': 'unbiased MMD squared',
                'statistic': 'mean over test conditions', 'records': records}
    out.with_suffix('.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(out)
    for record in records:
        result = record['evaluation']
        print(f'{record["run"]}: Err_post={result["err_post"]["mean"]:.9g}, '
              f'Err_resim={result["err_resim"]["mean"]:.9g}')


if __name__ == '__main__':
    main()
