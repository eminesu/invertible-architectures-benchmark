"""Calibration / confidence diagnostics for posterior samplers.

Answers: is a model over-confident (posterior too narrow), under-confident
(too wide) or calibrated? Works for any model with the shared interface

    sample(y_star, n) -> x,   x: (M, n, dim_x)

Test conditions come with the true parameter x_true (metrics/conditions/*.npz),
so we can check *frequentist coverage*: if the model's posterior is right, the
true x falls inside its q-credible region for a fraction q of the conditions.

Three diagnostics, all computed per y* and then aggregated:

  1. Rank statistics (SBC, Talts et al. 2018). Per dimension, rank_d = fraction
     of the n posterior samples below x_true[d] (ties broken at random, needed
     for the integer-valued v0 of ballistics). Calibrated -> Uniform(0, 1).
       U-shape (x_true in the tails)  -> over-confident
       hump (x_true near the middle)  -> under-confident
       skew                           -> biased
  2. Central-interval coverage. Fraction of y* with x_true inside the central
     alpha-interval, i.e. |rank - 0.5| <= alpha/2, at alpha in COVERAGE_LEVELS.
     coverage < alpha -> over-confident, > alpha -> under-confident.
     `signed_error` = mean(coverage - alpha): negative / positive accordingly.
  3. Joint (HPD-like) coverage, which also sees multimodality and correlations.
     Density at a point is proxied by the distance to its k-th nearest posterior
     sample (small = dense). alpha*(y*) = fraction of posterior samples that are
     denser than x_true; calibrated -> Uniform(0, 1), same reading as above.

Optionally (needs the ground-truth posterior, metrics/ground_truth.py):

  4. Spread ratio. Per y*, sqrt(trace cov) of the model samples over that of the
     ground-truth samples. log ratio < 0 -> too narrow, > 0 -> too wide. This
     separates "too narrow/wide" from "wrong location", which MMD cannot.

Everything is computed in x units standardised by the std of x_true over the
test set so that all dimensions weigh equally (v0 of ballistics has ~10x the
scale of the others).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kstest

COVERAGE_LEVELS = np.linspace(0.05, 0.95, 19)
TOL = 0.03  # |signed_error| below this is reported as "calibrated"
KNN = 10
N_REF = 200  # reference samples per y* for the joint statistic


def _as_np(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def ranks(samples, x_true, rng=None):
    """Per-dimension randomised rank of x_true among the samples.
    samples (M, n, d), x_true (M, d) -> (M, d) in [0, 1]."""
    rng = np.random.RandomState(0) if rng is None else rng
    lt = (samples < x_true[:, None, :]).mean(1)
    eq = (samples == x_true[:, None, :]).mean(1)
    return lt + rng.uniform(size=lt.shape) * eq


def central_coverage(r, levels=COVERAGE_LEVELS):
    """r: ranks (M,) or (M, d) -> empirical coverage of the central intervals,
    shape (len(levels),) or (len(levels), d)."""
    dev = np.abs(np.asarray(r) - 0.5)
    return np.stack([(dev <= a / 2).mean(0) for a in levels])


def joint_hpd_level(samples, x_true, k=KNN, n_ref=N_REF, rng=None):
    """alpha*(y*) per condition, (M,): fraction of posterior samples that lie in
    a denser (smaller k-NN radius) place than x_true. Uniform if calibrated."""
    rng = np.random.RandomState(0) if rng is None else rng
    M, n, _ = samples.shape
    out = np.empty(M)
    for i in range(M):
        tree = cKDTree(samples[i])
        r_true = tree.query(x_true[i], k=k)[0][-1]
        ref = samples[i][rng.choice(n, size=min(n_ref, n), replace=False)]
        r_ref = tree.query(ref, k=k + 1)[0][:, -1]  # k+1: the point itself is its own 1st neighbour
        out[i] = (r_ref < r_true).mean()
    return out


def verdict(signed_error, tol=TOL):
    if signed_error < -tol:
        return 'over-confident'
    if signed_error > tol:
        return 'under-confident'
    return 'calibrated'


def _uniformity(r):
    """KS distance to U(0,1), p-value and the mass in the outer 10% of the rank
    range (0.10 if calibrated; much larger -> U-shape)."""
    r = np.asarray(r).ravel()
    ks = kstest(r, 'uniform')
    return {'ks': float(ks.statistic), 'ks_p': float(ks.pvalue),
            'tail_mass': float(((r < 0.05) | (r > 0.95)).mean())}


def spread(samples):
    """sqrt(trace of covariance) per condition: (M, n, d) -> (M,)."""
    return np.sqrt(samples.var(axis=1, ddof=1).sum(-1))


def calibration_report(samples, x_true, gt_samples=None, scale=None, levels=COVERAGE_LEVELS, seed=0):
    """Full report as a JSON-serialisable dict (plus arrays under '_arrays').

    samples: (M, n, d) model samples, x_true: (M, d), gt_samples: optional list
    of M arrays (n_i, d) from the ground-truth posterior."""
    S, xt = _as_np(samples), _as_np(x_true)
    scale = xt.std(0) if scale is None else np.asarray(scale)
    # Non-finite samples (e.g. an inverse that blows up) would silently count as
    # "below x_true" in the ranks and crash the k-NN step. Conditions with any
    # such sample are left out, and the fractions are reported.
    ok = np.isfinite(S).all(axis=(1, 2))
    bad = {'nonfinite_sample_fraction': float(1 - np.isfinite(S).mean()),
           'nonfinite_condition_fraction': float(1 - ok.mean())}
    if ok.sum() < 50:
        raise ValueError(f'only {int(ok.sum())} of {len(ok)} conditions have all-finite samples; '
                         f'calibration is not meaningful ({bad})')
    if not ok.all():
        S, xt = S[ok], xt[ok]
        if gt_samples is not None:
            gt_samples = [g for g, k in zip(gt_samples, ok) if k]
    S, xt = S / scale, xt / scale
    rng = np.random.RandomState(seed)
    M, _, d = S.shape

    r = ranks(S, xt, rng)
    cov_dim = central_coverage(r, levels)  # (L, d)
    cov = cov_dim.mean(1)
    se_dim = (cov_dim - levels[:, None]).mean(0)
    se = float((cov - levels).mean())

    alpha_star = joint_hpd_level(S, xt, rng=rng)
    cov_joint = np.array([(alpha_star <= a).mean() for a in levels])
    se_joint = float((cov_joint - levels).mean())

    rep = {
        'n_conditions': int(M), 'n_conditions_total': int(len(ok)), **bad,
        'levels': levels.tolist(),
        'marginal': {'coverage': cov.tolist(), 'signed_error': se, 'abs_error': float(np.abs(cov - levels).mean()),
                     'verdict': verdict(se), **_uniformity(r),
                     'per_dim': [{'dim': j, 'signed_error': float(se_dim[j]), 'verdict': verdict(se_dim[j]),
                                  **_uniformity(r[:, j])} for j in range(d)]},
        'joint': {'coverage': cov_joint.tolist(), 'signed_error': se_joint,
                  'abs_error': float(np.abs(cov_joint - levels).mean()), 'verdict': verdict(se_joint),
                  **_uniformity(alpha_star)},
    }
    arrays = {'ranks': r, 'alpha_star': alpha_star}

    if gt_samples is not None:
        gt_sp = np.array([spread(np.asarray(g, dtype=np.float64)[None] / scale)[0] for g in gt_samples])
        log_ratio = np.log(spread(S) / gt_sp)
        rep['spread_vs_ground_truth'] = {
            'median_log_ratio': float(np.median(log_ratio)), 'median_ratio': float(np.exp(np.median(log_ratio))),
            'frac_too_narrow': float((log_ratio < -np.log(1.25)).mean()),
            'frac_too_wide': float((log_ratio > np.log(1.25)).mean()),
            'verdict': verdict(float(np.median(log_ratio)), tol=0.1)}
        arrays['log_spread_ratio'] = log_ratio
    rep['_arrays'] = arrays
    return rep


def calibration(sample_fn, benchmark, name=None, n_samples=1000, with_ground_truth=True, verbose=True):
    """Run the diagnostics for one model on the shared test conditions."""
    from metrics.benchmarks import load_test_conditions
    y_star, x_true = load_test_conditions(benchmark)
    x = _as_np(sample_fn(np.asarray(y_star, dtype=np.float32), n_samples))
    gt = None
    if with_ground_truth:
        from metrics.ground_truth import ground_truth_posterior
        gt = ground_truth_posterior(benchmark, y_star, verbose=verbose)
    rep = calibration_report(x, x_true, gt)
    rep.update(benchmark=benchmark, model=name)
    if verbose:
        print_report(rep)
    return rep


def print_report(rep):
    m, j = rep['marginal'], rep['joint']
    if rep.get('nonfinite_condition_fraction', 0) > 0:
        print(f"WARNING {rep['benchmark']} {rep.get('model') or ''}: {100 * rep['nonfinite_condition_fraction']:.1f}% of y* "
              f"have non-finite samples and were left out ({100 * rep['nonfinite_sample_fraction']:.2f}% of all samples)")
    print(f"{rep['benchmark']:<11} {rep.get('model') or '':<16} "
          f"marginal: {m['verdict']} (signed err {m['signed_error']:+.3f}, tails {m['tail_mass']:.3f} vs 0.100)   "
          f"joint: {j['verdict']} ({j['signed_error']:+.3f})")
    for p in m['per_dim']:
        print(f"{'':<28} x{p['dim']}: {p['verdict']:<15} signed err {p['signed_error']:+.3f}  "
              f"KS {p['ks']:.3f}  tails {p['tail_mass']:.3f}")
    if 'spread_vs_ground_truth' in rep:
        s = rep['spread_vs_ground_truth']
        print(f"{'':<28} spread / ground truth: median x{s['median_ratio']:.2f} ({s['verdict']}); "
              f"{100 * s['frac_too_narrow']:.0f}% of y* >25% too narrow, {100 * s['frac_too_wide']:.0f}% >25% too wide")


def save(rep, out_dir, benchmark, name):
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = rep.pop('_arrays')
    np.savez(out_dir / f'{benchmark}_{name}_calibration.npz', **arrays)
    (out_dir / f'{benchmark}_{name}_calibration.json').write_text(json.dumps(rep, indent=2))


def plot(reports, out_path):
    """Coverage curves (diagonal = calibrated; below = over-confident, above =
    under-confident) and rank histograms for several models.
    reports: {label: report-with-_arrays}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, rep in reports.items():
        lv = rep['levels']
        ax[0].plot(lv, rep['marginal']['coverage'], marker='o', ms=3, label=label)
        ax[1].plot(lv, rep['joint']['coverage'], marker='o', ms=3, label=label)
        ax[2].hist(rep['_arrays']['ranks'].ravel(), bins=20, range=(0, 1), density=True, histtype='step', label=label)
    for a, t in zip(ax[:2], ['Marginal central-interval coverage', 'Joint (HPD-like) coverage']):
        a.plot([0, 1], [0, 1], 'k--', lw=1)
        a.set(xlabel='nominal level', ylabel='empirical coverage', title=t)
        a.text(0.55, 0.12, 'over-confident', color='gray'); a.text(0.05, 0.85, 'under-confident', color='gray')
    ax[2].axhline(1, color='k', ls='--', lw=1)
    ax[2].set(xlabel='rank of x_true (all dims)', ylabel='density', title='Rank histogram (flat = calibrated)')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_summary(results_dir, out_dir=None, tol=TOL):
    """One set of PNGs per benchmark from every *_calibration.{json,npz} in results_dir:
      <bench>_signed_error.png   bar chart, one row per run: left of 0 = over-confident,
                                 right = under-confident, green band = calibrated
      <bench>_rank_histograms.png  small multiples of the SBC rank histogram per run (flat = calibrated)
      <bench>_coverage_curves.png  marginal and joint coverage curves of all runs (diagonal = calibrated)
      <bench>_spread_vs_calibration.png  (runs scored with the ground truth) posterior width vs the ground truth, and
                                 that against the calibration verdict: do the two checks agree?
      <bench>_spread_distribution.png  per-y* spread ratio of every run
    Returns the list of files written."""
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / 'calibration_plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bench = {}
    for f in sorted(results_dir.glob('*_calibration.json')):
        npz = f.with_suffix('.npz')
        if not npz.exists():
            continue
        rep = json.load(open(f))
        d = np.load(npz)
        rep['_arrays'] = {k: d[k] for k in d.files}
        by_bench.setdefault(rep['benchmark'], {})[rep['model']] = rep
    colour = {'over-confident': 'tab:red', 'under-confident': 'tab:blue', 'calibrated': 'tab:green'}
    written = []
    for bench, runs in by_bench.items():
        names = sorted(runs, key=lambda n: runs[n]['marginal']['signed_error'])

        fig, ax = plt.subplots(figsize=(10, max(3.0, 0.34 * len(names) + 1.6)))
        y, h = np.arange(len(names)), 0.38
        ax.barh(y - h / 2, [runs[n]['marginal']['signed_error'] for n in names], h, label='marginal (per dimension, averaged)')
        ax.barh(y + h / 2, [runs[n]['joint']['signed_error'] for n in names], h, label='joint (HPD-like)')
        ax.axvspan(-tol, tol, color='green', alpha=0.12, label=f'calibrated (±{tol})')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('signed coverage error   (← over-confident  |  under-confident →)')
        ax.set_title(f'{bench}: confidence of each model')
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        p = out_dir / f'{bench}_signed_error.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

        ncols = min(4, len(names))
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
        for a in axes.ravel():
            a.axis('off')
        for a, n in zip(axes.ravel(), names):
            a.axis('on')
            a.hist(runs[n]['_arrays']['ranks'].ravel(), bins=20, range=(0, 1), density=True, color='gray')
            a.axhline(1, color='k', ls='--', lw=0.8)
            a.set_title(f"{n}\n{runs[n]['marginal']['verdict']}", fontsize=7, color=colour[runs[n]['marginal']['verdict']])
            a.tick_params(labelsize=6)
        fig.suptitle(f'{bench}: rank of x_true among posterior samples (flat = calibrated, U = over-, hump = under-confident)',
                     fontsize=9)
        fig.tight_layout()
        p = out_dir / f'{bench}_rank_histograms.png'
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for n in names:
            lv = runs[n]['levels']
            ax[0].plot(lv, runs[n]['marginal']['coverage'], lw=1, label=n)
            ax[1].plot(lv, runs[n]['joint']['coverage'], lw=1, label=n)
        for a, t in zip(ax, ['Marginal central-interval coverage', 'Joint (HPD-like) coverage']):
            a.plot([0, 1], [0, 1], 'k--', lw=1)
            a.set(xlabel='nominal level', ylabel='empirical coverage', title=f'{bench}: {t}')
            a.text(0.55, 0.08, 'over-confident', color='gray')
            a.text(0.03, 0.88, 'under-confident', color='gray')
        ax[1].legend(fontsize=6, ncol=2, loc='lower right')
        fig.tight_layout()
        p = out_dir / f'{bench}_coverage_curves.png'
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))

        # ---- ground-truth spread vs the calibration check (needs the runs scored with GT=1) ----
        gt_names = [n for n in names if 'spread_vs_ground_truth' in runs[n] and 'log_spread_ratio' in runs[n]['_arrays']]
        if not gt_names:
            continue
        ratio = {n: runs[n]['spread_vs_ground_truth']['median_ratio'] for n in gt_names}
        order = sorted(gt_names, key=lambda n: ratio[n])
        sv_colour = {'over-confident': 'tab:red', 'under-confident': 'tab:blue', 'calibrated': 'tab:green'}

        fig, ax = plt.subplots(1, 2, figsize=(15, max(4.5, 0.34 * len(order) + 1.8)))
        y = np.arange(len(order))
        ax[0].barh(y, [np.log2(ratio[n]) for n in order],
                   color=[sv_colour[runs[n]['spread_vs_ground_truth']['verdict']] for n in order])
        ax[0].axvline(0, color='k', lw=0.8)
        ax[0].axvspan(-0.15, 0.15, color='green', alpha=0.12)
        ax[0].set_yticks(y)
        ax[0].set_yticklabels(order, fontsize=8)
        ax[0].set_xlabel('log2( model spread / ground-truth spread )    (← too narrow | too wide →)')
        ax[0].set_title(f'{bench}: width of the posterior vs the ground truth (median over y*)')
        for yi, n in zip(y, order):
            ax[0].text(ax[0].get_xlim()[1], yi, f' x{ratio[n]:.2f}', va='center', fontsize=7)
        xs = np.array([runs[n]['marginal']['signed_error'] for n in gt_names])
        ys = np.array([np.log2(ratio[n]) for n in gt_names])
        ax[1].axhline(0, color='k', lw=0.8)
        ax[1].axvline(0, color='k', lw=0.8)
        ax[1].axvspan(-tol, tol, color='green', alpha=0.12)
        ax[1].scatter(xs, ys, c=[colour[runs[n]['marginal']['verdict']] for n in gt_names], s=36, zorder=3)
        for n, a, b in zip(gt_names, xs, ys):
            ax[1].annotate(n, (a, b), fontsize=6, xytext=(3, 3), textcoords='offset points')
        ax[1].set_xlabel('calibration check: signed coverage error  (← over-confident | under-confident →)')
        ax[1].set_ylabel('ground-truth check: log2 spread ratio  (↓ too narrow | too wide ↑)')
        ax[1].set_title('two independent checks: points in the lower-left / upper-right agree')
        fig.tight_layout()
        p = out_dir / f'{bench}_spread_vs_calibration.png'
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

        fig, ax = plt.subplots(figsize=(10, max(3.0, 0.34 * len(order) + 1.6)))
        ax.boxplot([np.log2(np.exp(runs[n]['_arrays']['log_spread_ratio'])) for n in order], vert=False, whis=(5, 95),
                   showfliers=False, labels=order)
        ax.axvline(0, color='k', lw=0.8)
        ax.axvspan(-0.32, 0.32, color='green', alpha=0.12)
        ax.set_xlabel('per-y* log2( model spread / ground-truth spread )   (green band = within 25%)')
        ax.set_title(f'{bench}: spread ratio over the 1000 test observations (box = quartiles, whiskers = 5-95%)')
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        p = out_dir / f'{bench}_spread_distribution.png'
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))
    return written