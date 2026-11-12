"""Log-log analysis of the FR3 training curves of every model family (no training, no sampling).

Run from any directory:
    python experiments/fr3_loglog.py

Reads experiments/runs/fr3_{6d,5d}_<run>/log.csv (committed files only) and writes:
    metrics/analysis/fr3_loglog.csv                                   one row per (task, run, series)
    experiments/figures/<folder>/learning_curves/loglog_learning_curve_<family>_<task>.{png,svg}
                                                                      every family except FMPE
    experiments/figures/shared/comparison/fr3_loglog_<task>.{png,svg} normalised improvement rate + late-drop share
    experiments/figures/shared/comparison/fr3_loglog_resim_<task>.{png,svg}
                                                                      per-epoch validation resimulation error

Epochs are counted from 1 (row `epoch` = 0 is epoch 1). Statistics per series:
    slope_<a>_<b>   log(value) against log(epoch) over epochs a..b (plot_analysis.loglog_slope);
                    strictly positive series only
    rate_slope, k   fit of log d(t) against log t over epochs 25..50, d(t) = S(t-1) - S(t), S = 3-epoch
                    centred moving average of the series (defined for epochs 2..49, so d for 3..49);
                    only epochs with d > 0 enter the fit. If L = a + b t^-k then d ~ t^-(k+1), so
                    k = -rate_slope - 1 (model-based estimate)
    late_drop_share (L(40) - L(50)) / (L_start - L(50)), L_start = the worst (largest) value over
                    epochs 1..40. For a monotone falling series that is L(1); for a series that first
                    rises it is the peak, so the share is measured from the top of the curve
    best_epoch      epoch with the lowest value (all series are minimised)
All series are minimised. Loss levels are never compared across families.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_analysis import (FIGURES, INK_MUTED, VALIDATION_INK, committed, finish, fixed_x_ticks,  # noqa: E402
                           log_limits, log_ticks, loglog_slope)
from plot_group_errors import COLORS, ROOT, RUNS, relative, save_figure, style_axis  # noqa: E402

TASKS = {'fr3_6d': 'FR3 6-D pose', 'fr3_5d': 'FR3 5-D pose'}
OUT_CSV = ROOT / 'metrics' / 'analysis' / 'fr3_loglog.csv'
WINDOWS = ((1, 5), (5, 25), (25, 50), (30, 50))
RATE_WINDOW = (25, 50)
MIN_RATE_EPOCHS = 15
LATE_EPOCH, LAST_EPOCH = 40, 50
STILL_IMPROVING, FLAT = 0.05, 0.01

# family -> (label, figure folder, runs with tasks, series analysed, comparison series)
FAMILIES = {
    'fmpe': ('FMPE', 'fmpe', {'fmpe_s0': TASKS, 'fmpe_a0_s0': TASKS, 'fmpe_p1000000_s0': TASKS,
                              'fmpe_p300000_s0': TASKS},
             ('train_loss', 'val_loss'), 'val_loss'),
    'cfmpe': ('cFMPE', 'cfmpe', {'cfmpe_s0': TASKS}, ('train_loss', 'val_loss'), 'val_loss'),
    'fff': ('FFF / M-FFF', 'fff', {'fff_b400_s0': TASKS, 'mfff_wrapped_b400_s0': TASKS},
            ('train_nll', 'val_nll', 'train_recon', 'val_recon', 'train_loss', 'val_loss'), 'val_nll'),
    'fif': ('FIF', 'fif', {'fif_s0': TASKS},
            ('train_loss', 'val_loss', 'val_recon', 'val_latent', 'val_trace', 'val_resim'), 'val_loss'),
    'ikflow': ('IKFlow', 'ikflow', {'ikflow': TASKS}, ('train_loss', 'val_loss', 'val_resim'), 'val_loss'),
    'cinn': ('cINN', 'inn', {'cinn': TASKS}, ('train_loss', 'val_loss', 'val_resim'), 'val_loss'),
    'cilnn': ('ciLNN', 'ilnn', {'cilnn': TASKS}, ('train_loss', 'val_loss', 'val_resim'), 'val_loss'),
    'ilnn': ('iLNN', 'ilnn', {'pad3': TASKS, 'base': ('fr3_5d',)},
             ('train_loss', 'val_loss', 'val_nll', 'val_resim'), 'val_loss'),
}
RUN_LABELS = {'fmpe_s0': 'FMPE 3M (α = 1)', 'fmpe_a0_s0': 'FMPE 3M (α = 0)',
              'fmpe_p1000000_s0': 'FMPE 1M', 'fmpe_p300000_s0': 'FMPE 300k', 'cfmpe_s0': 'cFMPE',
              'fff_b400_s0': 'FFF', 'mfff_wrapped_b400_s0': 'M-FFF', 'fif_s0': 'FIF',
              'ikflow': 'IKFlow', 'cinn': 'cINN', 'cilnn': 'ciLNN', 'pad3': 'iLNN pad3', 'base': 'iLNN base'}
# Panels of the per-family figures: (title, series drawn together). Training solid, validation dashed.
PANELS = {
    'cfmpe': (('Flow-matching loss', ('train_loss', 'val_loss')),),
    'fff': (('NLL, as d(t)', ('train_nll', 'val_nll')),
            ('Reconstruction error', ('train_recon', 'val_recon')),
            ('Total loss (rises first)', ('train_loss', 'val_loss'))),
    'fif': (('FIF loss', ('train_loss', 'val_loss')), ('Reconstruction', ('val_recon',)),
            ('Latent term', ('val_latent',)), ('Trace term', ('val_trace',)),
            ('Validation resimulation', ('val_resim',))),
    'ikflow': (('NLL, as d(t)', ('train_loss', 'val_loss')),
               ('Validation resimulation', ('val_resim',))),
    'cinn': (('NLL, as d(t)', ('train_loss', 'val_loss')),
             ('Validation resimulation', ('val_resim',))),
    'cilnn': (('NLL, as d(t)', ('train_loss', 'val_loss')),
              ('Validation resimulation', ('val_resim',))),
    'ilnn': (('Loss (= val_nll on validation)', ('train_loss', 'val_loss')),
             ('Validation resimulation', ('val_resim',))),
}
RESIM_FAMILIES = ('fif', 'ikflow', 'cinn', 'cilnn', 'ilnn')
# One (colour, line style, marker) per run, so no two runs share a look in the comparison figures.
# Flow-matching runs solid with filled circles, likelihood / autoencoder runs dashed with open squares.
RUN_STYLES = {'fmpe_s0': (COLORS[0], '-', 'o'), 'fmpe_a0_s0': (COLORS[1], '-', 'o'),
              'fmpe_p1000000_s0': (COLORS[2], '-', 'o'), 'fmpe_p300000_s0': (COLORS[3], '-', 'o'),
              'cfmpe_s0': (COLORS[4], '-', 'o'), 'fff_b400_s0': (COLORS[0], '--', 's'),
              'mfff_wrapped_b400_s0': (COLORS[1], '--', 's'), 'fif_s0': (COLORS[2], '--', 's'),
              'ikflow': (COLORS[3], '--', 's'), 'cinn': (COLORS[4], '--', 's'), 'cilnn': (COLORS[5], '--', 's'),
              'pad3': ('#4c5967', ':', 'D'), 'base': ('#a0522d', ':', 'D')}


def runs_for(task):
    for family, (_, _, runs, series, comparison) in FAMILIES.items():
        for run, tasks in runs.items():
            if task in tasks:
                yield family, run, series, comparison


def read_columns(run):
    path = RUNS / run / 'log.csv'
    if not committed(path):
        raise FileNotFoundError(f'No committed log.csv: {relative(path)}')
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    columns = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    return path, columns


def check_epochs(run, columns):
    """Returns (fatal problems, deviations). 50 consecutive rows numbered 1..50 (FIF writes its loop
    counter, range(1, 51)) are a numbering deviation, not missing data; they map to epochs 1..50."""
    epochs = columns['epoch'].astype(int)
    if len(epochs) != LAST_EPOCH:
        return [f'{len(epochs)} rows, not {LAST_EPOCH}'], []
    if np.array_equal(epochs, np.arange(LAST_EPOCH)):
        return [], []
    if np.array_equal(epochs, np.arange(1, LAST_EPOCH + 1)):
        return [], [f'epoch column numbered 1..{LAST_EPOCH}, not 0..{LAST_EPOCH - 1} (50 consecutive rows)']
    return [f'epochs {epochs.min()}..{epochs.max()} not consecutive from 0'], []


def sign_of(values):
    if (values > 0).all():
        return 'positive'
    if (values < 0).all():
        return 'negative'
    return 'signed'


def improvement(values):
    """d(t) = S(t-1) - S(t) on the 3-epoch centred moving average S. Returns (epochs, d)."""
    smooth = np.convolve(values, np.ones(3) / 3, mode='valid')      # epochs 2..49
    smooth_epochs = np.arange(2, len(values))
    return smooth_epochs[1:], smooth[:-1] - smooth[1:]              # epochs 3..49


def rate_fit(values):
    t, d = improvement(values)
    window = (t >= RATE_WINDOW[0]) & (t <= RATE_WINDOW[1])
    keep = window & (d > 0)
    used, nonpositive = int(keep.sum()), int((window & (d <= 0)).sum())
    if used < 3:
        return dict(rate_slope=None, k=None, rate_epochs_used=used, rate_epochs_nonpositive=nonpositive)
    slope = float(np.polyfit(np.log(t[keep]), np.log(d[keep]), 1)[0])
    return dict(rate_slope=slope, k=-slope - 1, rate_epochs_used=used, rate_epochs_nonpositive=nonpositive)


def start_level(values):
    return float(values[:LATE_EPOCH].max())


def late_drop(values):
    start = start_level(values)
    rises_first = int(np.argmax(values[:LATE_EPOCH])) > 0
    total = start - values[LAST_EPOCH - 1]
    if total <= 0:
        return None, 'no net improvement', rises_first
    share = float((values[LATE_EPOCH - 1] - values[LAST_EPOCH - 1]) / total)
    label = ('still improving' if share > STILL_IMPROVING else 'flat' if share < FLAT else 'slowing')
    return share, label, rises_first


def reported_epoch(task, run):
    """Epoch of the weights in model.pt: FIF saves its best validation-resim epoch, the rest the last."""
    summary = json.loads((RUNS / f'{task}_{run}' / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((RUNS / f'{task}_{run}' / 'config.json').read_text(encoding='utf-8'))
    if 'best_epoch' in summary:
        return int(summary['best_epoch']), 'best validation resim (summary.json best_epoch)'
    if config.get('ema', 0):
        return LAST_EPOCH, 'last epoch, EMA weights'
    return LAST_EPOCH, 'last epoch (model.pt saved after the final epoch)'


def analyse(task, family, run, series, columns):
    epochs = np.arange(1, len(columns['epoch']) + 1)
    steps = np.diff(np.concatenate([[0], columns['step']]))
    if not (steps == steps[0]).all():
        raise ValueError(f'Steps per epoch vary: {task}_{run}')
    model_epoch, model_note = reported_epoch(task, run)
    rows = []
    for name in series:
        values = columns[name]
        sign = sign_of(values)
        row = dict(task=task, run=f'{task}_{run}', family=FAMILIES[family][0], series=name, sign=sign,
                   split='validation' if name.startswith('val_') else 'train')
        for first, last in WINDOWS:
            row[f'slope_{first}_{last}'] = (loglog_slope(epochs, values, first, last)
                                            if sign == 'positive' else None)
        row.update(rate_fit(values))
        if row['split'] == 'validation':
            share, label, rises = late_drop(values)
            row.update(late_drop_share=share, late_drop_label=label,
                       late_drop_start=('peak over epochs 1-40 (series rises first)' if rises
                                        else 'epoch 1'))
        else:
            row.update(late_drop_share=None, late_drop_label=None, late_drop_start=None)
        best = int(epochs[np.argmin(values)])
        row.update(best_epoch=best, best_is_last=best == LAST_EPOCH, last_epoch=int(epochs[-1]),
                   value_epoch_1=float(values[0]), value_epoch_40=float(values[LATE_EPOCH - 1]),
                   value_epoch_50=float(values[-1]), value_best=float(values.min()),
                   steps_per_epoch=int(steps[0]), model_epoch=model_epoch, model_epoch_source=model_note,
                   note=('identical to val_loss in this log' if name == 'val_nll'
                         and np.array_equal(values, columns.get('val_loss')) else ''))
        rows.append(row)
    return rows


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with OUT_CSV.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(keys)
        for row in rows:
            writer.writerow([fmt(row[key]) for key in keys])
    print(f'Saved {OUT_CSV}')


# ----------------------------------------------------------------------------------------- figures

def draw_series(ax, epochs, values, colour, style, label):
    """Positive series: value on log-log axes. Otherwise d(t) (only d > 0 drawn). Returns plotted y."""
    if sign_of(values) == 'positive':
        ax.plot(epochs, values, color=colour, linestyle=style, linewidth=1.8, label=label)
        return values, False
    t, d = improvement(values)
    keep = d > 0
    ax.plot(t[keep], d[keep], color=colour, linestyle=style, linewidth=1.6, marker='o', markersize=2.5,
            label=label)
    return d[keep], True


def plot_family(family, task, data, report):
    title, folder, runs, _, _ = FAMILIES[family]
    runs = [run for run in runs if (task, run) in data]
    panels = PANELS[family]
    fig, axes = plt.subplots(len(runs), len(panels), figsize=(4.3 * len(panels) + 1.2, 3.7 * len(runs) + 1.5),
                             squeeze=False)
    any_rate = False
    for row_axes, run in zip(axes, runs):
        columns = data[(task, run)]
        epochs = np.arange(1, len(columns['epoch']) + 1)
        for ax, (panel_title, names) in zip(row_axes, panels):
            plotted, rate = [], False
            for name in names:
                is_val = name.startswith('val_')
                colour = VALIDATION_INK if is_val else COLORS[0]
                y, is_rate = draw_series(ax, epochs, columns[name], colour, '--' if is_val else '-',
                                         'Validation' if is_val else 'Training')
                plotted.append(y)
                rate |= is_rate
            any_rate |= rate
            model_epoch = reported_epoch(task, run)[0]
            if model_epoch != LAST_EPOCH:
                ax.axvline(model_epoch, color='#8c97a3', linewidth=0.9, linestyle=':')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, LAST_EPOCH)
            low, high = log_limits(np.concatenate(plotted), minimum_ratio=1.5)
            ax.set_ylim(low, high)
            log_ticks(ax.yaxis, low, high)
            fixed_x_ticks(ax, [1, 2, 5, 10, 20, 50])
            ax.set_title(f'{RUN_LABELS[run]}: {panel_title}', loc='left', fontsize=10.5, fontweight='bold')
            ax.set_xlabel('Epoch (log scale)', fontsize=9.5)
            ax.set_ylabel('Improvement per epoch d(t)' if rate else 'Value (log scale)', fontsize=9.5)
            style_axis(ax)
    fig.legend(handles=[Line2D([], [], color=COLORS[0], lw=2, label='Training'),
                        Line2D([], [], color=VALIDATION_INK, lw=2, ls='--', label='Validation')],
               loc='upper right', bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=False)
    subtitle = 'Positive series as logged, on log-log axes'
    if any_rate:
        subtitle += ('; signed series (NLL) shown as improvement per epoch d(t) = S(t-1) - S(t), '
                     'S = 3-epoch moving average, only d > 0 drawn')
    note = ('Each panel has its own y-axis. No smoothing, shifting or clipping of positive series. '
            'Levels are not comparable with other families (different objectives).')
    if family == 'ikflow':
        note += ' IKFlow uses batch 128 (7,812 steps per epoch); the other families 1,000 steps per epoch.'
    if family == 'fif':
        note += (' Dotted line: epoch 27, the checkpoint the reported FIF model comes from (best validation '
                 'resim), not epoch 50.')
    if family == 'ilnn':
        note += ' val_nll is identical to val_loss in these logs, so it is not drawn twice.'
    width = 4.3 * len(panels) + 1.2
    height = 3.7 * len(runs) + 1.5
    finish(fig, f'{TASKS[task]}: {title} learning curves (log-log)', subtitle, note, height, int(width * 15))
    path = FIGURES / folder / 'learning_curves' / f'loglog_learning_curve_{family}_{task}'
    save_figure(fig, path)
    report['figures'].append(relative(path.with_suffix('.png')))


def plot_comparison(task, data, rows, report):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={'width_ratios': [1.25, 1]})
    entries = []
    for index, (family, run, _, comparison) in enumerate(runs_for(task)):
        values = data[(task, run)][comparison]
        t, d = improvement(values)
        total = start_level(values) - values[-1]
        keep = d > 0
        colour, style, marker = RUN_STYLES[run]
        left.plot(t[keep], d[keep] / total, color=colour, linestyle=style, linewidth=1.6,
                  label=f'{RUN_LABELS[run]} ({comparison})')
        row = next(r for r in rows if r['run'] == f'{task}_{run}' and r['series'] == comparison)
        entries.append((row['late_drop_share'], RUN_LABELS[run], comparison, colour, marker))
    left.set_xscale('log')
    left.set_yscale('log')
    left.set_xlim(3, LAST_EPOCH)
    fixed_x_ticks(left, [3, 5, 10, 20, 50])
    left.set_xlabel('Epoch (log scale)', fontsize=10)
    left.set_ylabel('d(t) / total improvement (log scale)', fontsize=10)
    left.set_title('Normalised improvement per epoch', loc='left', fontweight='bold', fontsize=11.5)
    left.legend(fontsize=8, frameon=False, loc='lower left', ncol=2)
    style_axis(left)

    entries.sort(key=lambda entry: -np.inf if entry[0] is None else entry[0])
    positions = np.arange(len(entries))
    for y, (share, label, _, colour, marker) in zip(positions, entries):
        filled = marker == 'o'
        right.plot([100 * share], [y], marker, color=colour, markersize=7.5,
                   markerfacecolor=colour if filled else 'white', markeredgewidth=1.6)
        right.text(100 * share, y + 0.28, f'{100 * share:.2f}%', ha='center', fontsize=8, color=INK_MUTED)
    right.set_yticks(positions, [f'{label} ({series})' for _, label, series, _, _ in entries], fontsize=9)
    for level, text in ((1, '1% (flat below)'), (5, '5% (still improving above)')):
        right.axvline(level, color='#8c97a3', linewidth=1, linestyle=':')
        right.text(level, len(entries) - 0.35, text, fontsize=8, color=INK_MUTED, ha='left', va='bottom',
                   rotation=0)
    shares = [100 * entry[0] for entry in entries]
    right.set_xlim(min(0, min(shares)) - 0.5, max(max(shares), 5) * 1.12 + 0.5)
    right.set_ylim(-0.7, len(entries) + 0.3)
    right.set_xlabel('Late-drop share: (L(40) − L(50)) / (L_start − L(50)), %', fontsize=10)
    right.set_title('Share of the total improvement made in epochs 41–50', loc='left', fontweight='bold',
                    fontsize=11.5)
    style_axis(right)
    right.grid(axis='x', color='#dce2e8', linewidth=0.65)
    note = ('Validation series per run: val_loss (FMPE, cFMPE, FIF, IKFlow, cINN, ciLNN, iLNN), val_nll '
            '(FFF, M-FFF). d(t) is the drop of the 3-epoch moving average from one epoch to the next; epochs '
            'with d ≤ 0 are not drawn. Total improvement = worst value over epochs 1–40 minus the epoch-50 '
            'value, so the curves are scale- and shift-free. IKFlow runs 7,812 steps per epoch, the others '
            '1,000. FIF\'s reported model is from epoch 27.')
    finish(fig, f'{TASKS[task]}: how far each run was from convergence at epoch 50',
           'Left: improvement rate (log-log). Right: late-drop share, sorted.', note, 6.4, 215)
    path = FIGURES / 'shared' / 'comparison' / f'fr3_loglog_{task}'
    save_figure(fig, path)
    report['figures'].append(relative(path.with_suffix('.png')))


def plot_resim(task, data, report):
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    values_all = []
    for index, (family, run, series, _) in enumerate(
            entry for entry in runs_for(task) if entry[0] in RESIM_FAMILIES):
        epochs = np.arange(1, LAST_EPOCH + 1)
        values = data[(task, run)]['val_resim']
        values_all.append(values)
        colour, style, _ = RUN_STYLES[run]
        ax.plot(epochs, values, color=colour, linestyle=style, linewidth=1.9, label=RUN_LABELS[run])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, LAST_EPOCH)
    low, high = log_limits(np.concatenate(values_all))
    ax.set_ylim(low, high)
    log_ticks(ax.yaxis, low, high)
    fixed_x_ticks(ax, [1, 2, 5, 10, 20, 50])
    ax.set_xlabel('Epoch (log scale)', fontsize=10)
    ax.set_ylabel('Validation resimulation error (log scale)', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)
    note = ('Mean squared forward-kinematics error of posterior draws on validation poses, logged each epoch. '
            'FIF: 64 poses x 32 draws. IKFlow, cINN, ciLNN, iLNN: 200 poses x 100 draws; IKFlow samples '
            'its latent at scale 0.25. FMPE, cFMPE, FFF and M-FFF do not log a per-epoch resimulation '
            'error, so they are absent; nothing is estimated for them. Training signal only, not the test '
            'metric.')
    finish(fig, f'{TASKS[task]}: validation resimulation during training',
           'Families that log val_resim per epoch, log-log', note, 5.9, 122)
    path = FIGURES / 'shared' / 'comparison' / f'fr3_loglog_resim_{task}'
    save_figure(fig, path)
    report['figures'].append(relative(path.with_suffix('.png')))


def main():
    report = {'epoch_problems': {}, 'figures': []}
    data, rows = {}, []
    for task in TASKS:
        for family, run, series, _ in runs_for(task):
            _, columns = read_columns(f'{task}_{run}')
            problems, deviations = check_epochs(f'{task}_{run}', columns)
            if problems:
                report['epoch_problems'][f'{task}_{run}'] = problems
            if deviations:
                print(f'{task}_{run}: {deviations[0]}')
            missing = [name for name in series if name not in columns]
            if missing:
                raise KeyError(f'{task}_{run} has no column(s) {missing}')
            data[(task, run)] = columns
    if report['epoch_problems']:
        print('Logs without exactly 50 rows / epochs 0-49:', report['epoch_problems'])
        sys.exit(1)
    print('All included logs have 50 consecutive rows (epochs 1-50).')
    for task in TASKS:
        for family, run, series, _ in runs_for(task):
            rows.extend(analyse(task, family, run, series, data[(task, run)]))
    write_csv(rows)
    for task in TASKS:
        for family in FAMILIES:
            if family != 'fmpe':
                plot_family(family, task, data, report)
        plot_comparison(task, data, rows, report)
        plot_resim(task, data, report)


if __name__ == '__main__':
    main()
