"""Learning curve of a conditional FFF run from experiments/runs/<run>/log.csv,
three panels, train vs. validation on each:
  (a) the training objective (surrogate NLL + beta x reconstruction);
  (b) the exact NLL -log p(x | y) (encoder change of variables, raw x units),
      the clean overfitting check: the surrogate's *value* is ~ tr(I), not the
      log-det, so (a) is flat and noisy even when the density improves. Runs
      trained before train_nll was logged show val only;
  (c) the reconstruction error ||x - g(f(x))||^2 (log scale; must stay near
      zero for the likelihood to be valid).

    .venv/bin/python experiments/plot_fff_learning_curve.py kin_fff_s0 kin_mfff_s0

Same style as the conditional-FMPE learning curve
(experiments/plot_conditional_fmpe_learning_curve.py on emine-conditional-fmpe): one colour per run,
solid = train, dashed = val. The loss is FFF's training objective (surrogate
NLL + beta x reconstruction); it is not comparable across models or across
Euclidean vs. torus runs (different latent spaces), only along one run's
epochs. The validation loss uses fixed Hutchinson draws, so its epochs are
comparable; the train loss is the mean over the epoch's batches.
"""

import argparse
import csv
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

    fig, (ax, ax_n, ax_r) = plt.subplots(1, 3, figsize=(16, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for run, c in zip(args.runs, colors):
        log = read_log(run)
        ep = log['epoch'] + 1
        single = len(args.runs) == 1
        label = '' if single else f'{run}, '
        nll_final = f"{log['val_nll'][-1]:.3f}"
        ax.plot(ep, log['train_loss'], color=c, lw=2, label=f'{label}train')
        ax.plot(ep, log['val_loss'], color=c, lw=2, ls='--', label=f'{label}validation')
        if 'train_nll' in log:
            ax_n.plot(ep, log['train_nll'], color=c, lw=2, label=f'{label}train')
        ax_n.plot(ep, log['val_nll'], color=c, lw=2, ls='--',
                  label=f'{label}validation' + ('' if single else f' (final {nll_final})'))
        if single:  # with several runs the annotations collide; the final val NLL is in the legend instead
            ax.annotate(f"final: train {log['train_loss'][-1]:.3f}, val {log['val_loss'][-1]:.3f}",
                        xy=(ep[-1], log['val_loss'][-1]), xytext=(-8, 40), textcoords='offset points',
                        ha='right', fontsize=8, color='0.25')
            best = int(np.argmin(log['val_nll']))
            final = (f"train {log['train_nll'][-1]:.3f}, " if 'train_nll' in log else '') + f'val {nll_final}'
            ax_n.annotate(f"final: {final}\nbest val {log['val_nll'][best]:.3f} at epoch {best + 1}",
                          xy=(ep[-1], log['val_nll'][-1]), xytext=(-8, 40), textcoords='offset points',
                          ha='right', fontsize=8, color='0.25')
        ax_r.plot(ep, log['train_recon'], color=c, lw=2, label=f'{label}train')
        ax_r.plot(ep, log['val_recon'], color=c, lw=2, ls='--', label=f'{label}validation')
    ax.set_ylabel('FFF loss (surrogate NLL + beta x reconstruction)')
    ax_n.set_ylabel('exact NLL  -log p(x | y)')
    ax_r.set_ylabel('reconstruction error ||x - g(f(x))||^2')
    ax_r.set_yscale('log')
    for a in (ax, ax_n, ax_r):
        a.set_xlabel('epoch')
        a.grid(alpha=0.3, lw=0.5)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.legend(fontsize=8, frameon=False)
    fig.suptitle(f'Conditional FFF: {", ".join(args.runs)}', fontsize=10)
    fig.tight_layout()
    out = args.out or Path(__file__).parent / 'figures' / f'fff_learning_curve_{"_".join(args.runs)}.png'
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
