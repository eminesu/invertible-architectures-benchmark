"""Helpers for experiments/run_ikflow_baseline.sh: choose the best run of a group, tabulate results.

    python experiments/ikflow_pick.py pick --task fr3_6d --select err_post --min-gain 0.03 \
        --ref fr3_6d_ikf_ref  fr3_6d_ikf_ref fr3_6d_ikf_p1_lr_lr1e-3 ...        # prints the winning run
    python experiments/ikflow_pick.py table --split val --match ikf --tasks fr3_6d

`pick` reads metrics/results/<task>_<run>_val.json (written by evaluate_ikflow.py --split val). A
candidate replaces the reference only if it is better by more than --min-gain, relative: validation
has 200 poses, so smaller differences are treated as noise and the reference stays. --select is
err_post (MMD to the exact posterior, lower is better), coverage (higher), pos (median position
error, lower) or resim (lower). The winner goes to stdout, a table of the group to stderr.
"""

import argparse
import json
import pathlib
import sys

RESULTS = pathlib.Path(__file__).resolve().parent.parent / 'metrics' / 'results'
SELECT = {'err_post': ('post', 'min'), 'coverage': ('cov', 'max'), 'pos': ('pos', 'min'), 'resim': ('resim', 'min')}


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def row_of(d):
    ang = d.get('rot_err_deg') or d.get('axis_err_deg')
    return {'post': get(d, 'err_post', 'mean'), 'cov': get(d, 'coverage', 'mean'),
            'pos': get(d, 'pos_err_mm', 'median'), 'ang': get(ang, 'median') if ang else None,
            'viol': d.get('joint_limit_violation_fraction'), 'resim': get(d, 'err_resim', 'mean'),
            'params': d.get('n_params'), 'ms': d.get('inference_ms_per_condition')}


def load(task, run, split):
    suffix = '' if split == 'test' else f'_{split}'
    f = RESULTS / f'{task}_{run}{suffix}.json'
    return row_of(json.loads(f.read_text())) if f.exists() else None


def fmt(v, spec):
    return '-' if v is None else format(v, spec)


def print_table(rows, out):
    header = ['task', 'run', 'Err_post', 'coverage', 'pos mm', 'ang deg', 'limits viol', 'Err_resim', 'ms/pose', 'params']
    table = [[r['task'], r['run'], fmt(r['post'], '.4f'), fmt(r['cov'], '.3f'), fmt(r['pos'], '.1f'),
              fmt(r['ang'], '.1f'), fmt(r['viol'], '.3f'), fmt(r['resim'], '.3f'), fmt(r['ms'], '.2f'),
              '-' if r['params'] is None else f"{r['params'] / 1e6:.2f}M"] for r in rows]
    if not table:
        print('(no results)', file=out)
        return
    widths = [max(len(header[i]), *(len(t[i]) for t in table)) for i in range(len(header))]
    line = '  '.join('{:<%d}' % w for w in widths)
    print(line.format(*header), file=out)
    for t in table:
        print(line.format(*t), file=out)


def cmd_pick(a):
    field, direction = SELECT[a.select]
    rows = {run: load(a.task, run, a.split) for run in a.runs}
    for run, r in rows.items():
        if r is None or r[field] is None:
            print(f'warning: no {a.split} result for {run}; ignored', file=sys.stderr)
    ref = rows.get(a.ref)
    if ref is None or ref[field] is None:
        sys.exit(f'reference run {a.ref} has no {a.split} result (evaluate_ikflow.py --split {a.split})')
    scale = max(abs(ref[field]), 1e-12)
    best, best_v = a.ref, ref[field]
    for run, r in rows.items():
        if run == a.ref or r is None or r[field] is None:
            continue
        gain = (ref[field] - r[field]) / scale if direction == 'min' else (r[field] - ref[field]) / scale
        if gain > a.min_gain and (r[field] < best_v if direction == 'min' else r[field] > best_v):
            best, best_v = run, r[field]
    print_table([{'task': a.task, 'run': run, **r} for run, r in rows.items() if r is not None], sys.stderr)
    print(f'selected by {a.select} (min gain {a.min_gain:.0%} over {a.ref}): {best}', file=sys.stderr)
    print(best)


def cmd_table(a):
    suffix = '' if a.split == 'test' else f'_{a.split}'
    rows = []
    for task in a.tasks.split():
        for f in sorted(RESULTS.glob(f'{task}_*{suffix}.json')):
            name = f.stem[len(task) + 1:]
            if 'oracle' in name or name.endswith('_calibration'):
                continue
            if a.split == 'test' and name.endswith('_val'):
                continue
            if suffix:
                if not name.endswith(suffix):
                    continue
                name = name[:-len(suffix)]
            if a.match and a.match not in name:
                continue
            rows.append({'task': task, 'run': name, **row_of(json.loads(f.read_text()))})
    rows.sort(key=lambda r: (r['task'], r['post'] is None, r['post']))
    print_table(rows, sys.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('pick')
    q.add_argument('--task', required=True)
    q.add_argument('--split', default='val', choices=['val', 'test'])
    q.add_argument('--select', default='err_post', choices=list(SELECT))
    q.add_argument('--min-gain', type=float, default=0.03)
    q.add_argument('--ref', required=True)
    q.add_argument('runs', nargs='+')
    q.set_defaults(fn=cmd_pick)
    q = sub.add_parser('table')
    q.add_argument('--split', default='val', choices=['val', 'test'])
    q.add_argument('--match', default=None)
    q.add_argument('--tasks', default='fr3_5d fr3_6d')
    q.set_defaults(fn=cmd_table)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
