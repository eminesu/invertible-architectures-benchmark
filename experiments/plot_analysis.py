"""Diagnostic figures for the report, from committed results only (no sampling, no training).

Run from any directory:
    python experiments/plot_analysis.py
    python experiments/plot_analysis.py --fr3   # only loglog_learning_curve_fmpe_fr3_* (nothing else is rewritten)

Writes under experiments/figures/:
    <model>/learning_curves/loglog_learning_curve_<family>_<problem>   training objective, log-log
    shared/comparison/error_vs_size_<problem>                          final Err_post against parameter count
    fmpe/sweeps/fmpe_alpha_sweep_<problem>                             FMPE time prior against solver and NFE
    <model>/ode_progression/ode_progression_<model>_<problem>          final error against solver and NFE
    shared/comparison/analysis_plots_sources.json                      sources, skipped runs and derived numbers

Training objectives come from experiments/runs/<run>/log.csv and final test errors from
metrics/results/*.json. The two are never mixed, and objectives of different families
never share a y-axis. Only files tracked by git are used. Markdown tables of the sampler
results, and the medians and tail shares quoted in docs/analysis_plots.md, are printed to stdout.
"""

import csv
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator
import numpy as np

from plot_group_errors import COLORS, RESULTS, ROOT, RUNS, relative, save_figure, style_axis


FIGURES = ROOT / 'experiments' / 'figures'
PROBLEMS = {'kinematics': ('kin', 'Kinematics'), 'ballistics': ('bal', 'Ballistics')}
# FR3 runs are named after the full problem (fr3_6d_fmpe_s0); only the FMPE curve is drawn (--fr3).
FR3_PROBLEMS = {'fr3_6d': ('fr3_6d', 'FR3 6-D pose'), 'fr3_5d': ('fr3_5d', 'FR3 5-D pose')}
INK_MUTED = '#4c5967'
VALIDATION_INK = '#273441'

# family -> (title, objective label, run suffixes, validation column)
FAMILIES = {
    'fmpe': ('FMPE', 'Flow-matching loss', ('fmpe_s0',), 'val_loss'),
    'npse': ('NPSE (BayesFlow diffusion)', 'Score-matching loss', ('bf_npse_s0',), 'val_loss'),
    'cfmpe': ('Conditional FMPE', 'Flow-matching loss', ('cfmpe_s0',), 'val_loss'),
    'fff': ('FFF / M-FFF', 'Surrogate objective',
            ('fff_s0', 'mfff_s0', 'mfff_wrapped_b400_s0'), 'val_loss'),
    'mdn': ('MDN (K = 16)', 'Negative log-likelihood', ('K16_s0',), 'val_nll'),
    'cinn': ('cINN', 'Negative log-likelihood', ('cinn_s0', 'cinn_s1', 'cinn_s2'), 'val_loss'),
}
PANEL_LABELS = {'fff_s0': 'FFF (Euclidean)', 'mfff_s0': 'M-FFF (torus, uniform)',
                'mfff_wrapped_b400_s0': 'M-FFF (wrapped, batch 400)'}
SIZE_MODELS = (('FMPE', 'fmpe', '_midpointx16', 'midpoint x16, 32 NFE', COLORS[0], 'o'),
               ('NPSE', 'bf_npse', '_eulerx32', 'Euler x32, 32 NFE', COLORS[1], 's'))
SIZE_BUDGETS = ('p300000_', 'p1000000_', '')
ALPHAS = ((0, 'fmpe_a0_s0', COLORS[0], 'o'), (1, 'fmpe_s0', COLORS[1], 's'),
          (4, 'fmpe_a4_s0', COLORS[2], '^'))
ALPHA_SCRIPT_GRID = (('euler', 8), ('midpoint', 4), ('midpoint', 16), ('rk4', 8), ('rk4', 16))
SOLVERS = {'euler': ('Euler', COLORS[0], 'o'), 'midpoint': ('Midpoint', COLORS[1], 's'),
           'rk4': ('RK4', COLORS[2], '^'), 'rk45': ('RK45 (fixed step)', COLORS[3], 'D'),
           'euler_maruyama': ('Euler-Maruyama (reverse SDE)', COLORS[4], 'v')}
ODE_MODELS = (('fmpe', 'FMPE (hand-rolled)', 'fmpe_s0'),
              ('bf_fmpe', 'FMPE (BayesFlow)', 'bf_fmpe_s0'),
              ('npse', 'NPSE (BayesFlow diffusion)', 'bf_npse_s0'))
RESULT_NAME = re.compile(r'(?P<mps>mps_)?(?P<solver>euler_maruyama|euler|midpoint|rk45|rk4)x(?P<steps>\d+)$')


def tracked_files():
    output = subprocess.check_output(['git', 'ls-files', 'experiments/runs', 'metrics/results'],
                                     cwd=ROOT, text=True)
    return set(output.splitlines())


TRACKED = tracked_files()


def committed(path):
    return relative(path) in TRACKED


def read_result(path, benchmark):
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['benchmark'] != benchmark:
        raise ValueError(f'Wrong benchmark: {path}')
    if result['n_conditions'] != 1000 or result['n_samples'] != 1000:
        raise ValueError(f'Unexpected evaluation protocol: {path}')
    for key in ('err_post', 'err_resim'):
        if not np.isfinite(result[key]['mean']) or result[key]['mean'] <= 0:
            raise ValueError(f'{key} mean cannot go on a log axis: {path}')
    return result


def find_result(benchmark, run, suffix):
    path = RESULTS / f'{benchmark}_{run}{suffix}.json'
    return path if committed(path) else None


def read_config(run):
    return json.loads((RUNS / run / 'config.json').read_text(encoding='utf-8'))


def read_log(run, validation):
    path = RUNS / run / 'log.csv'
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    epochs = np.array([int(row['epoch']) for row in rows]) + 1
    if not np.array_equal(epochs, np.arange(1, len(rows) + 1)):
        raise ValueError(f'Unexpected epoch sequence: {path}')
    train = np.array([float(row['train_loss']) for row in rows])
    val = np.array([float(row[validation]) for row in rows])
    if not np.isfinite([train, val]).all():
        raise ValueError(f'Non-finite learning-curve data: {path}')
    return epochs, train, val


def loglog_slope(epochs, values, first, last):
    keep = (epochs >= first) & (epochs <= last)
    return float(np.polyfit(np.log(epochs[keep]), np.log(values[keep]), 1)[0])


def flatten_epoch(epochs, train, fraction=0.05):
    """First epoch after which train loss stays within `fraction` of its total drop from the end value."""
    excess = train - train[-1]
    above = np.flatnonzero(excess > fraction * (train[0] - train[-1]))
    return int(epochs[above[-1]] + 1) if len(above) else int(epochs[0])


def curve_stats(epochs, train, val):
    gap = val - train
    return {'final_train': float(train[-1]), 'final_val': float(val[-1]),
            'final_gap_val_minus_train': float(gap[-1]),
            'final_gap_percent_of_train': float(100 * gap[-1] / train[-1]),
            'max_abs_gap_last_10_epochs': float(np.abs(gap[-10:]).max()),
            'slope_epochs_1_5': loglog_slope(epochs, train, 1, 5),
            'slope_epochs_5_25': loglog_slope(epochs, train, 5, 25),
            'slope_epochs_25_50': loglog_slope(epochs, train, 25, 50),
            'flatten_epoch_95pct_of_drop': flatten_epoch(epochs, train)}


def log_ticks(axis, low, high, count=5):
    """Readable log-axis ticks: decades for wide ranges, 1-2-5 for medium, plain numbers for narrow."""
    if high / low >= 30:
        axis.set_minor_formatter(NullFormatter())
        return
    decades = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    ticks = [m * 10.0 ** d for d in decades for m in (1, 2, 5) if low <= m * 10.0 ** d <= high]
    if high / low < 3 or len(ticks) < 3:
        ticks = [tick for tick in MaxNLocator(nbins=count).tick_values(low, high) if low <= tick <= high]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
    axis.set_minor_locator(NullLocator())


def log_limits(values, minimum_ratio=1.0, pad=1.14):
    """Log-axis limits. A minimum ratio stops tiny differences from filling the panel."""
    low, high = float(np.min(values)) / pad, float(np.max(values)) * pad
    if high / low < minimum_ratio:
        centre = np.sqrt(low * high)
        low, high = centre / np.sqrt(minimum_ratio), centre * np.sqrt(minimum_ratio)
    return low, high


def fixed_x_ticks(ax, ticks, labels=None):
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}') if labels is None
                                 else FuncFormatter(lambda value, _: labels.get(value, '')))


def finish(fig, title, subtitle, note, height, width_chars):
    note = textwrap.fill(note, width_chars)
    lines = note.count('\n') + 1
    bottom = (0.22 + 0.17 * lines) / height
    fig.text(0.03, 1 - 0.27 / height, title, ha='left', va='center', fontsize=16, fontweight='bold')
    fig.text(0.03, 1 - 0.58 / height, subtitle, fontsize=10.5, color=INK_MUTED, va='center')
    fig.text(0.03, 0.12 / height, note, fontsize=9, color=INK_MUTED, va='bottom', linespacing=1.4)
    fig.tight_layout(rect=(0.01, bottom, 0.99, 1 - 0.72 / height), w_pad=2.5, h_pad=2)


def plot_loglog(family, problem, report):
    prefix, name = {**PROBLEMS, **FR3_PROBLEMS}[problem]
    title, objective, suffixes, validation = FAMILIES[family]
    panels = []
    for suffix in suffixes:
        run = f'{prefix}_{suffix}'
        log_path = RUNS / run / 'log.csv'
        if not committed(log_path):
            report['loglog_missing'].append({'run': run, 'reason': 'no committed log.csv'})
            continue
        epochs, train, val = read_log(run, validation)
        bad_train, bad_val = int((train <= 0).sum()), int((val <= 0).sum())
        panels.append(dict(run=run, suffix=suffix, epochs=epochs, train=train, val=val,
                           bad_train=bad_train, bad_val=bad_val, log_path=log_path))
    eligible = [panel for panel in panels if not panel['bad_train'] and not panel['bad_val']]
    for panel in panels:
        if panel['bad_train'] or panel['bad_val']:
            report['loglog_excluded'].append({
                'run': panel['run'], 'family': family,
                'train_nonpositive_epochs': panel['bad_train'],
                'val_nonpositive_epochs': panel['bad_val'],
                'train_min': float(panel['train'].min()), 'val_min': float(panel['val'].min()),
                'val_nonpositive_at_epoch': panel['epochs'][panel['val'] <= 0].tolist()})
    if not eligible:
        return
    width = max(7.6, 5.2 * len(panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 5.1), squeeze=False)
    for panel, ax in zip(panels, axes.flat):
        ax.set_title(PANEL_LABELS.get(panel['suffix'], panel['run']), loc='left',
                     fontweight='bold', fontsize=11.5, pad=20)
        if panel not in eligible:
            ax.text(0.5, 0.5, 'Not plotted: objective is non-positive\n'
                    f"train {panel['bad_train']}/{len(panel['train'])}, "
                    f"validation {panel['bad_val']}/{len(panel['val'])} epochs\n"
                    f"(minimum {min(panel['train'].min(), panel['val'].min()):.3g})",
                    transform=ax.transAxes, ha='center', va='center', color='#697583', fontsize=10.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            continue
        epochs, train, val = panel['epochs'], panel['train'], panel['val']
        stats = curve_stats(epochs, train, val)
        report['loglog_plotted'][panel['run']] = dict(stats, source=relative(panel['log_path']))
        ax.plot(epochs, train, color=COLORS[0], linewidth=1.9, label='Training')
        ax.plot(epochs, val, color=VALIDATION_INK, linewidth=1.7, linestyle='--', label='Validation')
        ax.axvline(stats['flatten_epoch_95pct_of_drop'], color='#8c97a3', linewidth=0.9, linestyle=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, epochs[-1])
        low, high = log_limits(np.concatenate([train, val]))
        ax.set_ylim(low, high)
        fixed_x_ticks(ax, [1, 2, 5, 10, 20, 50])
        log_ticks(ax.yaxis, low, high)
        ax.set_xlabel('Epoch (log scale)', fontsize=10)
        ax.set_ylabel(f'{objective} (log scale)', fontsize=10)
        ax.text(0, 1.02, f"Final train {train[-1]:.4g}, val {val[-1]:.4g}; "
                f"95% of drop by epoch {stats['flatten_epoch_95pct_of_drop']}",
                transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, color=INK_MUTED)
        style_axis(ax)
    fig.legend(handles=[Line2D([], [], color=COLORS[0], lw=2, label='Training'),
                        Line2D([], [], color=VALIDATION_INK, lw=2, ls='--', label='Validation')],
               loc='upper right', bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=False)
    note = ('Values as logged, without smoothing or shifting. Dotted line: first epoch after which '
            'training loss stays within 5% of its total drop from the final value.')
    excluded = [panel for panel in panels if panel not in eligible]
    if excluded:
        note += (' Runs with any non-positive training or validation value are not drawn on log axes; '
                 'nothing was clipped or shifted.')
    if len(panels) > 1:
        note += ' Objectives differ between panels; each panel has its own y-axis.'
    finish(fig, f'{name}: {title} learning curves', 'Training and held-out validation objective (seed = 0)',
           note, 5.1, int(width * 15.5))
    folder = {'cinn': 'inn'}.get(family, family)
    save_figure(fig, FIGURES / folder / 'learning_curves' / f'loglog_learning_curve_{family}_{problem}')


def plot_size(problem, report):
    prefix, name = PROBLEMS[problem]
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    entry = {}
    all_values = []
    for label, stem, suffix, sampler, color, marker in SIZE_MODELS:
        xs, ys, sources = [], [], []
        for budget in SIZE_BUDGETS:
            run = f'{prefix}_{stem}_{budget}s0'
            path = find_result(problem, run, suffix)
            if path is None:
                report['size_missing'].append({'run': run, 'suffix': suffix})
                continue
            config = read_config(run)
            result = read_result(path, problem)
            if result['n_params'] != config['n_params']:
                raise ValueError(f'n_params disagree for {run}')
            xs.append(config['n_params'])
            ys.append(result['err_post']['mean'])
            sources.append(relative(path))
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        all_values.extend(ys)
        ax.plot(xs, ys, color=color, marker=marker, markersize=7.5, linewidth=1.9,
                label=f'{label} ({sampler})')
        ax.scatter(xs[-1:], ys[-1:], s=210, facecolor='none', edgecolor=color, linewidth=1.6, zorder=3)
        for x, y in zip(xs, ys):
            offset = (0, -18) if label == 'FMPE' else (0, 12)
            ax.annotate(f'{y:.5f}', (x, y), xytext=offset, textcoords='offset points',
                        ha='center', fontsize=9, color=VALIDATION_INK)
        seeds = {}
        for seed in (1, 2):
            run = f'{prefix}_{stem}_s{seed}'
            path = find_result(problem, run, suffix)
            if path is None:
                continue
            value = read_result(path, problem)['err_post']['mean']
            seeds[f's{seed}'] = value
            all_values.append(value)
            ax.scatter([read_config(run)['n_params']], [value], s=26, color=color, alpha=0.55,
                       edgecolor='white', linewidth=0.5, zorder=4)
        entry[label] = dict(sampler=sampler, n_params=xs.tolist(), err_post=ys.tolist(),
                            headline_other_seeds=seeds, sources=sources)
    ax.set_xscale('log')
    ax.set_yscale('log')
    low, high = log_limits(all_values, minimum_ratio=1.5)
    ax.set_ylim(low, high)
    ax.set_xlim(2.2e5, 4.2e6)
    fixed_x_ticks(ax, [3e5, 1e6, 3e6], {3e5: '300k', 1e6: '1M', 3e6: '3M'})
    log_ticks(ax.yaxis, low, high)
    ax.set_xlabel('Parameters, n_params from config.json (log scale)', fontsize=10)
    ax.set_ylabel('Mean Err_post, unbiased MMD squared (log scale)', fontsize=10)
    style_axis(ax)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color='#697583', marker='o', markerfacecolor='none', markersize=12,
                       linestyle='', label='3M headline point'),
                Line2D([], [], color='#697583', marker='o', markersize=4.5, alpha=0.6,
                       linestyle='', label='3M, seeds 1 and 2')]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=9.5)
    finish(fig, f'{name}: final error against model size', 'Seed 0 at every size; lower is better',
           'Same sampler at every size: FMPE midpoint x16, NPSE Euler x32 (both 32 NFE). '
           'The y-range spans at least a factor of 1.5, so differences of about 1% stay visually small. '
           '1,000 conditions x 1,000 samples.', 5.3, 118)
    save_figure(fig, FIGURES / 'shared' / 'comparison' / f'error_vs_size_{problem}')
    report['size'][problem] = entry


def alpha_grid(problem):
    prefix, _ = PROBLEMS[problem]
    grid = {}
    for alpha, suffix, _, _ in ALPHAS:
        run = f'{prefix}_{suffix}'
        for solver in ('euler', 'midpoint'):
            for steps in (2, 4, 8, 16, 32, 64):
                path = find_result(problem, run, f'_{solver}x{steps}')
                if path is not None:
                    grid[(alpha, solver, steps)] = path
    return grid


def plot_alpha(problem, report):
    prefix, name = PROBLEMS[problem]
    effective = read_config(f'{prefix}_fmpe_s0').get('alpha_effective')
    if effective != 1.0:
        raise ValueError(f'Headline run is not alpha_effective = 1: {effective}')
    grid = alpha_grid(problem)
    common = sorted(key for key in {(s, n) for _, s, n in grid}
                    if all((alpha, *key) in grid for alpha, *_ in ALPHAS))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), sharex='col')
    entry = {}
    for row, (metric, label) in enumerate((('err_post', 'Mean Err_post'), ('err_resim', 'Mean Err_resim'))):
        panel_values = {}
        for column, solver in enumerate(('midpoint', 'euler')):
            ax = axes[row, column]
            steps = [n for s, n in common if s == solver]
            nfe, gathered = None, []
            for alpha, suffix, color, marker in ALPHAS:
                results = [read_result(grid[(alpha, solver, n)], problem) for n in steps]
                nfe = [result['nfe'] for result in results]
                values = [result[metric]['mean'] for result in results]
                gathered.extend(values)
                ax.plot(nfe, values, color=color, marker=marker, markersize=7, linewidth=1.9)
                entry.setdefault(f'{prefix}_{suffix}', {})[f'{solver}_{metric}'] = dict(zip(nfe, values))
            ax.set_xscale('log')
            ax.set_yscale('log')
            panel_values[column] = gathered
            fixed_x_ticks(ax, nfe)
            ax.set_xlim(nfe[0] / 1.2, nfe[-1] * 1.2)
            style_axis(ax)
            if row == 0:
                ax.set_title({'midpoint': 'Midpoint', 'euler': 'Euler'}[solver], loc='left',
                             fontweight='bold', fontsize=12)
            else:
                ax.set_xlabel('Function evaluations, NFE (log scale)', fontsize=10)
            ax.set_ylabel(f'{label} (log scale)', fontsize=10)
        low, high = log_limits(panel_values[0] + panel_values[1], minimum_ratio=1.5)
        for ax in axes[row]:
            ax.set_ylim(low, high)
            log_ticks(ax.yaxis, low, high)
    handles = [Line2D([], [], color=color, marker=marker, markersize=7, lw=2,
                      label=f'alpha = {alpha}' + (' (headline run)' if alpha == 1 else ''))
               for alpha, _, color, marker in ALPHAS]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.985), ncol=3, frameon=False)
    finish(fig, f'{name}: FMPE time-prior sweep', 'p(t) proportional to t^alpha; 3M parameters, seed 0; lower is better',
           'alpha = 1 is the headline run (alpha_effective = 1.0 in its config.json). Only solver settings '
           'committed for all three alphas are shown. The rk4 settings listed in sweep_alpha.sh have no '
           'committed result for alpha 0 or 4. The two panels of each row share one y-range.', 8.0, 160)
    save_figure(fig, FIGURES / 'fmpe' / 'sweeps' / f'fmpe_alpha_sweep_{problem}')
    missing = [{'alpha': alpha, 'solver': solver, 'steps': steps}
               for alpha, suffix, *_ in ALPHAS[::2] for solver, steps in ALPHA_SCRIPT_GRID
               if find_result(problem, f'{prefix}_{suffix}', f'_{solver}x{steps}') is None]
    report['alpha'][problem] = dict(alpha_effective_headline=effective, values=entry,
                                    shown=[list(key) for key in common],
                                    in_sweep_script_but_missing=missing)


def ode_results(problem, run):
    found = {}
    stem = f'{problem}_{run}_'
    for path in sorted(RESULTS.glob(f'{stem}*.json')):
        match = RESULT_NAME.fullmatch(path.stem[len(stem):])
        if match is None or not committed(path):
            continue
        result = read_result(path, problem)
        if result['method'] != match['solver'] or result['steps'] != int(match['steps']):
            raise ValueError(f'File name and contents disagree: {path}')
        found.setdefault(match['solver'], []).append(
            dict(nfe=result['nfe'], steps=result['steps'], post=result['err_post']['mean'],
                 resim=result['err_resim']['mean'], mps=bool(match['mps']), source=relative(path)))
    return {solver: sorted(rows, key=lambda row: row['nfe']) for solver, rows in found.items()}


def plot_ode(model, title, suffix, problem, report):
    prefix, name = PROBLEMS[problem]
    results = ode_results(problem, f'{prefix}_{suffix}')
    if not results:
        report['ode_missing'].append({'model': model, 'problem': problem})
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5))
    grid = sorted({row['nfe'] for rows in results.values() for row in rows})
    for ax, key, label in zip(axes, ('post', 'resim'), ('Mean Err_post', 'Mean Err_resim')):
        values = []
        for solver in SOLVERS:
            if solver not in results:
                continue
            rows = results[solver]
            display, color, marker = SOLVERS[solver]
            nfe = [row['nfe'] for row in rows]
            value = [row[key] for row in rows]
            values.extend(value)
            ax.plot(nfe, value, color=color, marker=marker, markersize=7, linewidth=1.9, label=display)
            for row in rows:
                if row['mps']:
                    ax.scatter(row['nfe'], row[key], s=120, facecolor='white', edgecolor=color,
                               linewidth=1.6, marker=marker, zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ticks = [nfe for nfe in (4, 8, 16, 32, 64, 128, 256) if grid[0] <= nfe <= grid[-1]]
        fixed_x_ticks(ax, ticks)
        ax.set_xlim(grid[0] / 1.2, grid[-1] * 1.2)
        low, high = log_limits(values)
        ax.set_ylim(low, high)
        log_ticks(ax.yaxis, low, high)
        ax.set_xlabel('Function evaluations, NFE (log scale)', fontsize=10)
        ax.set_ylabel(f'{label} (log scale)', fontsize=10)
        ax.set_title('Posterior error' if key == 'post' else 'Re-simulation error', loc='left',
                     fontweight='bold', fontsize=12)
        style_axis(ax)
    axes[1].legend(loc='upper right', frameon=False, fontsize=9.5)
    note = ('Values from the committed result JSONs; no sampling was re-run. Missing solver and NFE '
            'combinations are left out, not interpolated. ')
    if any(row['mps'] for rows in results.values() for row in rows):
        note += 'Hollow marker: 256-NFE RK4 reference evaluated on MPS instead of CPU. '
    note += 'NFE is the value recorded in each file (stages x steps).'
    finish(fig, f'{name}: {title} sampling progression', 'Final error by solver and NFE; 3M parameters, seed 0; lower is better',
           note, 5.5, 175)
    folder = {'bf_fmpe': 'fmpe'}.get(model, model)
    save_figure(fig, FIGURES / folder / 'ode_progression' / f'ode_progression_{model}_{problem}')
    report['ode'][f'{model}_{problem}'] = results


def print_tables(report):
    for key, results in report['ode'].items():
        grid = sorted({row['nfe'] for rows in results.values() for row in rows})
        print(f'\n{key}: Err_post mean ({", ".join(sorted(results))})\n')
        print('| Solver | ' + ' | '.join(str(nfe) for nfe in grid) + ' |')
        print('|---|' + '---|' * len(grid))
        for solver in SOLVERS:
            if solver not in results:
                continue
            by_nfe = {row['nfe']: row['post'] for row in results[solver]}
            cells = [f'{by_nfe[nfe]:.4f}' if nfe in by_nfe else '–' for nfe in grid]
            print(f'| {SOLVERS[solver][0]} | ' + ' | '.join(cells) + ' |')


def post_summary(benchmark, run, suffix):
    path = find_result(benchmark, run, suffix)
    if path is None:
        raise FileNotFoundError(f'No committed result for {benchmark} {run}{suffix}')
    post = read_result(path, benchmark)['err_post']
    return post['mean'], post['median']


def per_condition_post(benchmark, run, suffix):
    if find_result(benchmark, run, suffix) is None:
        raise FileNotFoundError(f'No committed result for {benchmark} {run}{suffix}')
    with np.load(RESULTS / f'{benchmark}_{run}{suffix}.npz') as data:
        return data['err_post'], data['y_star']


def print_robustness():
    """Print the median, tail-share and paired-difference numbers quoted in docs/analysis_plots.md."""
    print('\nErr_post tail share: mean, median, share of the sum from the worst 1% and 5% of conditions\n')
    for benchmark, run, suffix in (('kinematics', 'kin_fmpe_s0', '_midpointx16'),
                                   ('ballistics', 'bal_fmpe_s0', '_midpointx16')):
        values = np.sort(per_condition_post(benchmark, run, suffix)[0])
        count = len(values)
        print(f'| {run}{suffix} | {values.mean():.5f} | {np.median(values):.5f} '
              f'| {values[-count // 100:].sum() / values.sum():.3f} '
              f'| {values[-count // 20:].sum() / values.sum():.3f} |')

    print('\nKinematics Err_post mean / median, midpoint by NFE\n')
    for steps in (2, 4, 8, 16, 32):
        mean, median = post_summary('kinematics', 'kin_fmpe_s0', f'_midpointx{steps}')
        print(f'| {2 * steps} | {mean:.5f} | {median:.5f} |')

    print('\nKinematics Err_post mean / median, alpha 0 / 1 / 4\n')
    for suffix in ('_midpointx4', '_midpointx8', '_midpointx16', '_eulerx8', '_eulerx16', '_eulerx32'):
        cells = [post_summary('kinematics', f'kin_{run}', suffix) for run in ('fmpe_a0_s0', 'fmpe_s0', 'fmpe_a4_s0')]
        print(f'| {suffix[1:]} | ' + ' | '.join(f'{mean:.5f} / {median:.5f}' for mean, median in cells) + ' |')

    print('\nKinematics Err_post mean / median by model size\n')
    for label, model, suffix, *_ in SIZE_MODELS:
        cells = [post_summary('kinematics', f'kin_{model}_{budget}s0', suffix) for budget in SIZE_BUDGETS]
        print(f'| {label} | ' + ' | '.join(f'{mean:.5f} / {median:.5f}' for mean, median in cells) + ' |')

    print('\nRK4 x16 against the other solvers at 64 NFE: means, paired difference and its standard error\n')
    for benchmark, prefix in (('kinematics', 'kin'), ('ballistics', 'bal')):
        rk4, y_star = per_condition_post(benchmark, f'{prefix}_fmpe_s0', '_rk4x16')
        for suffix in ('_midpointx32', '_eulerx64'):
            other, other_y = per_condition_post(benchmark, f'{prefix}_fmpe_s0', suffix)
            if not np.array_equal(y_star, other_y):
                raise ValueError(f'Conditions differ for {benchmark} {suffix}')
            difference = rk4 - other
            print(f'| {benchmark} | {suffix[1:]} | {rk4.mean():.5f} | {other.mean():.5f} '
                  f'| {difference.mean():+.5f} | {difference.std(ddof=1) / np.sqrt(len(difference)):.5f} '
                  f'| {np.median(rk4):.5f} | {np.median(other):.5f} |')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'axes.labelcolor': '#273441',
                         'text.color': '#273441', 'axes.titlecolor': '#273441'})
    report = dict(loglog_plotted={}, loglog_excluded=[], loglog_missing=[], size={}, size_missing=[],
                  alpha={}, ode={}, ode_missing=[])
    if '--fr3' in sys.argv[1:]:
        for problem in FR3_PROBLEMS:
            plot_loglog('fmpe', problem, report)
        print(json.dumps({key: report[key] for key in ('loglog_plotted', 'loglog_excluded', 'loglog_missing')},
                         indent=2))
        return
    for problem in PROBLEMS:
        for family in FAMILIES:
            plot_loglog(family, problem, report)
        plot_size(problem, report)
        plot_alpha(problem, report)
        for model, title, suffix in ODE_MODELS:
            plot_ode(model, title, suffix, problem, report)
    sources = FIGURES / 'shared' / 'comparison' / 'analysis_plots_sources.json'
    sources.parent.mkdir(parents=True, exist_ok=True)
    sources.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print_tables(report)
    print_robustness()


if __name__ == '__main__':
    main()
