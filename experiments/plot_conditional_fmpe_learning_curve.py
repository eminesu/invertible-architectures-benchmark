"""Learning curve of the prior-conditioned FMPE: train and validation flow
matching loss per epoch from experiments/runs/<run>/log.csv, on one plot.

    .venv/bin/python experiments/plot_conditional_fmpe_learning_curve.py bal_cfmpe_s0

Several runs (seeds) are overlaid, one colour per run, solid = train, dashed =
val. Only conditional-FMPE runs belong here; the other models' curves are
plotted elsewhere. The validation loss uses fixed (x0, t) draws, so its epochs
are comparable; the train loss is the mean over the epoch's batches.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).parent / 'runs'


def read_log(run):
    with open(RUNS / run / 'log.csv') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for run, c in zip(args.runs, colors):
        log = read_log(run)
        ep = log['epoch'] + 1
        label = '' if len(args.runs) == 1 else f'{run}, '
        ax.plot(ep, log['train_loss'], color=c, lw=2, label=f'{label}train')
        ax.plot(ep, log['val_loss'], color=c, lw=2, ls='--', label=f'{label}validation')
        best = int(np.argmin(log['val_loss']))
        ax.annotate(f"final: train {log['train_loss'][-1]:.3f}, val {log['val_loss'][-1]:.3f}\n"
                    f"best val {log['val_loss'][best]:.3f} at epoch {best + 1}",
                    xy=(ep[-1], log['val_loss'][-1]), xytext=(-8, 40), textcoords='offset points',
                    ha='right', fontsize=8, color='0.25')
    ax.set_xlabel('epoch')
    ax.set_ylabel('flow matching loss')
    ax.grid(alpha=0.3, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'Prior-conditioned FMPE, ballistics: {", ".join(args.runs)}', fontsize=10)
    fig.tight_layout()
    out = args.out or Path(__file__).parent / 'figures' / f'cfmpe_learning_curve_{"_".join(args.runs)}.png'
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
