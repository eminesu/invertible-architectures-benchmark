"""Create a reviewable Markdown report from actual completed suite artifacts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'experiments'/'runs'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', default='ae_suite')
    p.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    p.add_argument('--out', default='docs/autoencoder_results.md')
    p.add_argument('--plot', action='store_true', help='Create a comparison chart (requires matplotlib)')
    args = p.parse_args()
    entries = json.loads((RUNS/f'{args.prefix}_{args.profile}_manifest.json').read_text(encoding='utf-8'))
    rows, val_rows, records = [], [], []
    for entry in entries:
        folder = RUNS/entry['run']
        if not (folder/'summary.json').exists():
            continue
        cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        train = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        item = {**entry, 'config': cfg, 'training': train}
        val_rows.append(f"| {entry['problem']} | {entry['family']} | {entry['ablation']} | {cfg['n_params']} | {train['validation_selection_score']:.5f} | {train['best_epoch']} |")
        ev = folder/'evaluation'/'summary.json'
        diag = folder/'diagnostics'/'summary.json'
        if ev.exists():
            result = json.loads(ev.read_text(encoding='utf-8')); item['evaluation'] = result
            r = result['err_resim']['mean']
            r_text = f'{r:.5f}' if r is not None else 'undefined'
            rows.append(f"| {entry['problem']} | {entry['family']} | {entry['ablation']} | {result['err_post']['mean']:.5f} | {r_text} | {result['resim_failed_fraction']:.3%} | {result['inference_seconds']*1000:.1f} |")
        if diag.exists():
            item['diagnostics'] = json.loads(diag.read_text(encoding='utf-8'))
        records.append(item)
    min_gt = {b: min(r['evaluation']['n_gt_min'] for r in records if r['problem'] == b and 'evaluation' in r)
              for b in ['kinematics', 'ballistics'] if any(r['problem'] == b and 'evaluation' in r for r in records)}
    report = '\n'.join([
        '# Autoencoder experiment results', '',
        f'Profile: **{args.profile}**. Generated from completed run artifacts; {len(records)} trained runs.', '',
        'Pilot training uses 30,000 train / 2,000 validation pairs, 12 epochs and a matched <=100k parameter budget. '
        'Pilot evaluation uses the first 16 committed test conditions, 256 model / reference samples, '
        'and a 20M proposal cap. These are exploratory results, not a replication of the paper tables.', '',
        'Shared headline metrics remain raw-unit IMQ MMD² (unbiased, biased also saved), '
        'Gaussian reference widths 0.01 / 0.02, and clamp 10. No rounding/dequantization is used in training. '
        'Ballistics diagnostics include separate standardized and rounded-speed sensitivity scores.', '',
        f'Minimum accepted reference samples per condition: {min_gt}. '
        'The proposal cap can leave a shorter reference; the MMD estimator handles unequal sample counts.', '',
        '## Evaluation', '',
        '| Problem | Family | Ablation | Err_post | Err_resim | Failed | Total sampling ms |',
        '|---|---|---|---|---|---|---|', *rows, '',
        'Times are measured on the same CPU/container, including host conversion, with 8192-row decoder chunks. '
        'They are not comparable to the paper GPU figures. Checkpoint selection uses validation only, '
        'with a 100-unit penalty per invalid re-simulation; that selection score is not a headline metric.', '',
        '## Validation probes', '',
        '| Problem | Family | Ablation | Parameters | Best validation score | Epoch |',
        '|---|---|---|---|---|---|', *val_rows, '',
        'One-factor ablations are probes. A single seed and a short schedule do not establish architectural superiority. '
        'Full profiles use the existing provisional 1M/50-epoch schedule and <=3M parameters; '
        'team decisions on the definitive schedule, reference estimator/kernel and integer-speed handling remain open.', '',
        'Detailed JSON accompanies this report. Each run keeps config, best/final checkpoints, training log, '
        'evaluation arrays, latent/reconstruction summaries and physical posterior plots under experiments/runs/.', '',
    ])
    out = ROOT/args.out; out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for row, problem in enumerate(['kinematics', 'ballistics']):
            base = [r for r in records if r['problem'] == problem and r['ablation'] == 'baseline'
                    and 'evaluation' in r]
            for col, metric in enumerate(['err_post', 'err_resim']):
                values = [r['evaluation'][metric]['mean'] for r in base]
                axes[row, col].plot([r['family'] for r in base], values, 'o', markersize=8)
                axes[row, col].set_yscale('log')
                axes[row, col].set_ylim(min(values)*0.7, max(values)*1.3)
                for i, value in enumerate(values):
                    axes[row, col].annotate(f'{value:.3g}', (i, value), xytext=(0, 9),
                                           textcoords='offset points', ha='center')
                axes[row, col].set_title(f'{problem}: {metric} ({args.profile})')
                axes[row, col].set_ylabel('Mean error (log scale)')
        fig.tight_layout(); fig.savefig(out.with_suffix('.png'), dpi=150); plt.close(fig)
        with out.open('a', encoding='utf-8') as f:
            f.write(f'\n![Baseline comparison]({out.with_suffix(".png").name})\n')
    print(out)


if __name__ == '__main__':
    main()
