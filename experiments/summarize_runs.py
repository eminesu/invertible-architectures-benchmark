"""Print one row per finished run (experiments/runs/*/summary.json), and write runs.csv."""

import csv
import glob
import json
import os

RUNS = os.path.join(os.path.dirname(__file__), 'runs')
COLS = ['run', 'problem', 'components', 'n_params', 'val_nll', 'test_nll',
        'err_resim_provisional', 'resim_failed_fraction', 'inference_seconds', 'train_seconds']

rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, '*', 'summary.json')))]
with open(os.path.join(RUNS, 'runs.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, COLS, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
print(f"{'run':<28}{'K':>4}{'params':>11}{'test NLL':>10}{'resim':>9}{'infer s':>9}")
for r in rows:
    resim = r['err_resim_provisional']
    print(f"{r['run']:<28}{r['components']:>4}{r['n_params']:>11,}{r['test_nll']:>10.3f}"
          f"{resim if resim is not None else float('nan'):>9.4f}{r['inference_seconds']:>9.3f}")
