"""Plot saved FIF errors for both benchmarks; never runs training or evaluation."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=('kinematics', 'ballistics', 'both'), default='both')
    parser.add_argument('--kin-run', default='kin_fif_s0', help='Saved kinematics run to plot')
    parser.add_argument('--bal-run', default='bal_fif_s0', help='Saved ballistics run to plot')
    parser.add_argument('--compare-kin-run', help='Optional second kinematics run to plot alongside --kin-run')
    parser.add_argument('--compare-bal-run', help='Optional second ballistics run to plot alongside --bal-run')
    parser.add_argument('--title', default='FIF evaluation results', help='Figure title')
    parser.add_argument('--out', type=Path, help='Output PNG; comparison plots add -comparison to the default name')
    args = parser.parse_args()
    problems = ('kinematics', 'ballistics') if args.problem == 'both' else (args.problem,)
    name = {'both': 'fif', 'kinematics': 'kin', 'ballistics': 'bal'}[args.problem]
    compare_runs = {'kinematics': args.compare_kin_run, 'ballistics': args.compare_bal_run}
    comparing = any(compare_runs.values())
    if comparing and not all(compare_runs[problem] for problem in problems):
        parser.error('Provide a comparison run for each selected problem')
    if comparing:
        name += '-comparison'
    out = ROOT / (args.out or Path(f'docs/fif-results/{name}.png'))
    records, protocol = [], None
    runs = {'kinematics': args.kin_run, 'ballistics': args.bal_run}
    for problem in problems:
        selected_runs = [runs[problem]]
        if comparing:
            selected_runs.append(compare_runs[problem])
        for run in selected_runs:
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

    fig, axes = plt.subplots(len(problems), 2, figsize=(10, 3.2 * len(problems) + 0.6),
                             squeeze=False, sharey='col')
    styles = {'mlp': ('MLP-FIF', '#1f77b4', 'o'), 'paper': ('FIF-paper', '#ff7f0e', 's')}
    for col, metric in enumerate(('err_post', 'err_resim')):
        values = [record['evaluation'][metric]['mean'] for record in records]
        positive = min(values) > 0
        for row, problem in enumerate(problems):
            ax = axes[row, col]
            group = [record for record in records if record['problem'] == problem]
            labels = []
            for position, record in enumerate(group):
                value = record['evaluation'][metric]['mean']
                label, color, marker = styles.get(record['evaluation'].get('arch', 'mlp'),
                                                  (record['run'], '#1f77b4', 'o'))
                labels.append(label if comparing else record['run'])
                ax.scatter([position], [value], c=color if comparing else '#1f77b4',
                           marker=marker if comparing else 'o', s=65, zorder=3)
                ax.annotate(f'{value:.6g}', (position, value), xytext=(0, 10),
                            textcoords='offset points', ha='center', fontsize=10)
            if positive:
                ax.set_yscale('log')
                ax.set_ylim(min(values) * 0.65, max(values) * 2.5)
            else:
                ax.set_yscale('symlog', linthresh=1e-4)
            ax.set_xticks(range(len(group)), labels)
            ax.set_xlim(-0.5, len(group) - 0.5)
            ax.set_title(f'{problem}: {metric} (mean)')
            ax.set_ylabel('Mean error (log scale)' if positive else 'Mean error (symmetric-log scale)')
            ax.grid(axis='y', which='both', alpha=0.15)
    fig.suptitle(args.title, fontsize=13)
    fig.text(0.5, 0.015, f'Saved evaluations | {protocol["n_conditions"]:,} conditions x '
             f'{protocol["n_samples"]:,} samples | seed {protocol["evaluation_seed"]} | lower is better',
             ha='center', fontsize=9)
    fig.canvas.draw()  # Finalize shared log-axis ticks before measuring label margins.
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
