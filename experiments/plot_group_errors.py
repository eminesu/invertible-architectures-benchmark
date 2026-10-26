"""Plot saved learning curves and final test errors, without training models.

Run from any directory:
    python experiments/plot_group_errors.py

Learning curves use the recorded training objective and held-out validation
objective. Final test plots use the benchmark's Err_post and Err_resim means.
These are different metrics and are never substituted for each other.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'experiments' / 'runs'
RESULTS = ROOT / 'metrics' / 'results'
COLORS = ('#2563a5', '#d17b18', '#22836b', '#8b5da7', '#bd4b53', '#168b98')
PROBLEMS = {'kinematics': ('kin', 'Kinematics'), 'balistics': ('bal', 'Ballistics')}


@dataclass(frozen=True)
class Model:
    label: str
    run_suffix: str
    objective: str
    result_suffix: str = ''
    validation: str = 'val_loss'


GROUPS = {
    'fmpe': (
        Model('FMPE (baseline)', 'fmpe_s0', 'Flow-matching loss', '_midpointx16'),
        Model('FMPE (alpha = 0)', 'fmpe_a0_s0', 'Flow-matching loss', '_midpointx16'),
        Model('FMPE (alpha = 4)', 'fmpe_a4_s0', 'Flow-matching loss', '_midpointx16'),
        Model('FMPE (300k parameters)', 'fmpe_p300000_s0', 'Flow-matching loss', '_midpointx16'),
        Model('FMPE (1M parameters)', 'fmpe_p1000000_s0', 'Flow-matching loss', '_midpointx16'),
    ),
    'inn': (
        Model('INN (ML, seed=0)', 'inn_ml_s0', 'Maximum-likelihood objective'),
        Model('INN (L2 + MMD, seed=0)', 'inn_l2mmd_s0', 'L2 + MMD objective'),
        Model('cINN (seed=0)', 'cinn_s0', 'Negative log-likelihood'),
        Model('cINN (seed=2)', 'cinn_s2', 'Negative log-likelihood'),
        Model('iLNN (ML, seed=0)', 'ilnn_ml_s0', 'Maximum-likelihood objective'),
        Model('iLNN (MMD, seed=0)', 'ilnn_mmd_s0', 'MMD objective'),
    ),
    'fff': (
        Model('M-FFF (torus, uniform)', 'mfff_s0', 'M-FFF surrogate objective'),
        Model('FFF (Euclidean, fff_s0)', 'fff_s0', 'FFF surrogate objective'),
        Model('FIF (MLP)', 'fif_s0', 'FIF objective'),
    ),
}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def load_model(problem, model):
    prefix, _ = PROBLEMS[problem]
    benchmark = 'ballistics' if prefix == 'bal' else 'kinematics'
    run = f'{prefix}_{model.run_suffix}'
    log_path = RUNS / run / 'log.csv'
    if problem == 'balistics' and model.run_suffix == 'mfff_s0' and not log_path.exists():
        return dict(model=model, run=run, missing='M-FFF was not run for ballistics')
    with log_path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'Empty training log: {log_path}')
    epochs = np.array([int(row['epoch']) for row in rows])
    if epochs[0] == 0:
        epochs += 1
    if not np.array_equal(epochs, np.arange(1, len(rows) + 1)):
        raise ValueError(f'Unexpected epoch sequence: {log_path}')
    train = np.array([float(row['train_loss']) for row in rows])
    validation = np.array([float(row[model.validation]) for row in rows])
    if not np.isfinite([train, validation]).all():
        raise ValueError(f'Non-finite learning-curve data: {log_path}')
    config = json.loads((RUNS / run / 'config.json').read_text(encoding='utf-8'))
    seed = config['seed']
    if not run.endswith(f'_s{seed}'):
        raise ValueError(f'Run name and configured seed disagree: {run}, seed={seed}')

    if model.run_suffix.startswith('fif_'):
        result_path = RUNS / run / 'evaluation' / 'summary.json'
    else:
        result_path = RESULTS / f'{benchmark}_{run}{model.result_suffix}.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result['benchmark'] != benchmark:
        raise ValueError(f'Wrong benchmark: {result_path}')
    if result['n_conditions'] != 1000 or result['n_samples'] != 1000:
        raise ValueError(f'Unexpected test evaluation protocol: {result_path}')
    for key in ('err_post', 'err_resim'):
        if not np.isfinite(result[key]['mean']):
            raise ValueError(f'Non-finite test metric: {result_path}, {key}')
    return dict(model=model, run=run, epochs=epochs, train=train,
                validation=validation, result=result, seed=seed,
                log_path=log_path, result_path=result_path)


def seed_caption(records):
    seeds = sorted({record['seed'] for record in records if 'missing' not in record})
    return ', '.join(f'seed={seed}' for seed in seeds)


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dce2e8', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=200, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    print(f'Saved {path.with_suffix(".png")}')


def plot_learning(problem, group, records, path):
    columns = min(3, len(records))
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.1 * rows), squeeze=False)
    for index, (record, ax) in enumerate(zip(records, axes.flat)):
        color = COLORS[index]
        model = record['model']
        ax.set_title(model.label, loc='left', fontweight='bold', fontsize=12)
        if 'missing' in record:
            ax.text(0.5, 0.5, 'Not run for ballistics\nNo training / validation data',
                    transform=ax.transAxes, ha='center', va='center', color='#697583', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            continue
        epoch = record['epochs']
        ax.plot(epoch, record['train'], color=color, linewidth=1.9, label='Training')
        ax.plot(epoch, record['validation'], color='#273441', linewidth=1.7,
                linestyle='--', label='Validation')
        ax.set_ylabel(model.objective, fontsize=10)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_xlim(1, epoch[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.text(0.98, 0.97,
                f"Final train: {record['train'][-1]:.5g}\n"
                f"Final val: {record['validation'][-1]:.5g}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.82))
        style_axis(ax)
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    fig.suptitle(f'{PROBLEMS[problem][1]}: {group.upper()} group',
                 x=0.055, y=0.98, ha='left', fontsize=19, fontweight='bold')
    fig.text(0.055, 0.84 if rows == 1 else 0.915,
             f'Training and validation error over 50 epochs ({seed_caption(records)})',
             fontsize=12, color='#4c5967')
    fig.legend(handles=[Line2D([], [], color='#2563a5', lw=2, label='Training'),
                        Line2D([], [], color='#273441', lw=2, ls='--', label='Validation')],
               loc='upper right', bbox_to_anchor=(0.97, 0.98), ncol=2, frameon=False)
    note = 'Objectives have different definitions and independent y-axes. Values are plotted as logged, without smoothing.'
    if group == 'fff':
        note = 'FFF / M-FFF use surrogate objectives, not exact NLL. FIF uses a different objective. Each panel has its own y-axis.'
    fig.text(0.055, 0.025, note, fontsize=10, color='#4c5967')
    fig.tight_layout(rect=(0.015, 0.09 if rows == 1 else 0.06, 0.99,
                          0.85 if rows == 1 else 0.91), h_pad=2, w_pad=2.5)
    save_figure(fig, path)


def plot_testing(problem, group, records, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3))
    labels = [record['model'].label for record in records]
    positions = np.arange(len(records))
    for ax, key, title, xlabel in zip(
            axes, ('err_post', 'err_resim'),
            ('Posterior error', 'Re-simulation error'),
            ('Mean Err_post (unbiased MMD squared)',
             'Mean Err_resim (m squared)' if problem == 'balistics'
             else 'Mean Err_resim (squared simulator units)')):
        values = np.array([record['result'][key]['mean'] if 'missing' not in record
                           else np.nan for record in records])
        finite = values[np.isfinite(values)]
        # A dot plot keeps the logarithmic comparison honest (no arbitrary bar baseline).
        use_log = finite.min() > 0 and finite.max() / finite.min() >= 20
        if use_log:
            ax.set_xscale('log')
            ax.set_xlim(finite.min() / 2.2, finite.max() * 12)
        else:
            low = min(0, finite.min() * 1.2)
            ax.set_xlim(low, max(finite.max() * 1.45, low + 1e-8))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                ax.text(0.03, index, 'Not run for ballistics', transform=ax.get_yaxis_transform(),
                        va='center', fontsize=10, color='#697583')
                continue
            ax.scatter(value, index, s=95, color=COLORS[index], zorder=3)
            ax.annotate(f'{value:.4g}', (value, index), xytext=(10, 0),
                        textcoords='offset points', va='center', fontsize=10)
        ax.set_yticks(positions, labels, fontsize=10)
        ax.set_ylim(len(records) - 0.55, -0.6)
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=13)
        ax.set_xlabel(xlabel + (' (log scale)' if use_log else ''), fontsize=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#dce2e8', linewidth=0.65)
        ax.set_axisbelow(True)
    fig.suptitle(f'{PROBLEMS[problem][1]}: {group.upper()} testing error',
                 x=0.035, y=0.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(0.035, 0.90,
             'Saved checkpoint evaluation: 1,000 test conditions x 1,000 samples per condition. Lower is better.',
             fontsize=11, color='#4c5967')
    note = f'{seed_caption(records)}. Test metrics were saved once per evaluated checkpoint; no epoch-wise test curve is available.'
    if group == 'fmpe':
        note += '\nAll five FMPE runs use midpoint x16 (32 NFE).'
    if group == 'inn' and problem == 'kinematics':
        note += '\nThe INN (L2 + MMD) re-simulation outlier is retained at its recorded value.'
    fig.text(0.035, 0.025, note, fontsize=9, color='#4c5967', linespacing=1.5)
    fig.tight_layout(rect=(0, 0.12, 0.99, 0.86), w_pad=3.5)
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'plots')
    parser.add_argument('--groups', nargs='+', choices=tuple(GROUPS), default=tuple(GROUPS),
                        help='Regenerate only selected groups; source manifest always covers all groups')
    args = parser.parse_args()
    # Read and validate every source before writing any plots.
    loaded = {(problem, group): [load_model(problem, model) for model in models]
              for problem in PROBLEMS for group, models in GROUPS.items()}
    manifest = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.labelcolor': '#273441',
                         'text.color': '#273441', 'axes.titlecolor': '#273441'})
    for (problem, group), records in loaded.items():
        folder = args.out / problem
        if group in args.groups:
            plot_learning(problem, group, records, folder / 'training_error' / group)
            plot_testing(problem, group, records, folder / 'testing_error' / group)
        for record in records:
            if 'missing' in record:
                manifest.append({'problem': problem, 'group': group,
                                 'model': record['model'].label, 'run': record['run'],
                                 'status': 'unavailable', 'reason': record['missing']})
                continue
            manifest.append({
                'problem': problem, 'group': group, 'model': record['model'].label,
                'status': 'available',
                'seed': record['seed'],
                'run': record['run'], 'epochs': len(record['epochs']),
                'training_column': 'train_loss',
                'validation_column': record['model'].validation,
                'final_train': float(record['train'][-1]),
                'final_validation': float(record['validation'][-1]),
                'err_post_mean': record['result']['err_post']['mean'],
                'err_resim_mean': record['result']['err_resim']['mean'],
                'n_conditions': record['result']['n_conditions'],
                'n_samples': record['result']['n_samples'],
                'resim_failed_fraction': record['result'].get('resim_failed_fraction'),
                'training_source': relative(record['log_path']),
                'testing_source': relative(record['result_path']),
            })
    (args.out / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
