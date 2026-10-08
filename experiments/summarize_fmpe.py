"""Markdown table of FMPE results (NFE sweep) next to the other models, from
metrics/results/*.json.

    .venv/bin/python experiments/summarize_fmpe.py
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / 'metrics' / 'results'


def rows(benchmark):
    out = []
    for p in sorted(RESULTS.glob(f'{benchmark}_*.json')):
        s = json.loads(p.read_text())
        if s['model'].startswith('prelim'):
            continue
        out.append(s)
    # FMPE runs carry an 'nfe' field; order: other models first, then by implementation and NFE.
    return sorted(out, key=lambda s: ('nfe' in s, s.get('implementation', ''), s.get('device', ''),
                                      s.get('nfe') or 0, s['model']))


def main():
    for b in sys.argv[1:] or ['kinematics', 'ballistics']:
        print(f'\n### {b}\n')
        print('| Model | NFE | Err_post (biased) | Err_resim mean (median [q1, q3]) | failed | time, s |')
        print('|---|---|---|---|---|---|')
        for s in rows(b):
            p, r = s['err_post'], s['err_resim']
            pb = s.get('err_post_biased', {}).get('mean', float('nan'))
            nfe = s.get('nfe', '')
            dev = s.get('device', 'cpu')
            t = f"{s['inference_seconds']:.1f}" + ('' if dev == 'cpu' else f' ({dev})')
            print(f"| {s['model']} | {nfe} | {p['mean']:.4f} ({pb:.4f}) | "
                  f"{r['mean']:.1e} ({r['median']:.1e} [{r['q1']:.1e}, {r['q3']:.1e}]) | "
                  f"{100 * s['resim_failed_fraction']:.2f}% | {t} |")


if __name__ == '__main__':
    main()
