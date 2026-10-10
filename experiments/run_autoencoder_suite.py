"""Run all three families on both problems, optionally with TODO ablations.

Each stage is a separate process; failures are recorded and return nonzero.
Existing completed stages are reused, so an interrupted suite can be resumed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT/'experiments'/'autoencoder_bench.py'
RUNS = ROOT/'experiments'/'runs'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--prefix', default='ae_suite')
    p.add_argument('--ablations', action='store_true')
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--hidden', type=int, help='Override width (otherwise budget-matched within each profile)')
    args = p.parse_args()
    jobs = [(b, f, 'baseline', []) for b in ('kinematics', 'ballistics')
            for f in ('standard', 'cvae', 'invertible')]
    if args.ablations:
        # One-factor probes, not a claim of exhaustive hyperparameter tuning.
        for b in ('kinematics', 'ballistics'):
            for label, opts in [
                ('wz10', ['--w-z', '10']), ('wz0', ['--w-z', '0']),
                ('wy10', ['--w-y', '10']), ('marginal', ['--latent-mmd', 'marginal']),
                ('true_y', ['--recon-condition', 'true']), ('wide_kernel', ['--mmd-scale', '2']),
                ('z4', ['--z-dims', '4']), ('silu', ['--activation', 'silu']),
                ('depth2', ['--layers', '2']), ('lr_small', ['--lr', '0.0003'])]:
                jobs.append((b, 'standard', label, opts))
            jobs.append((b, 'cvae', 'beta1', ['--beta', '1']))
            jobs.append((b, 'invertible', 'orth', ['--orth-weight', '1']))
    records = []
    RUNS.mkdir(exist_ok=True)
    manifest = RUNS/f'{args.prefix}_{args.profile}_manifest.json'
    for problem, family, label, opts in jobs:
        run = f'{args.prefix}_{problem}_{family}_{label}_{args.profile}_s{args.seed}'
        folder = RUNS/run
        record = {'run': run, 'problem': problem, 'family': family, 'ablation': label}
        stages = [('train', [sys.executable, str(BENCH), 'train', '--family', family,
                    '--problem', problem, '--profile', args.profile, '--run-name', run, *opts], folder/'summary.json'),
                  ('evaluate', [sys.executable, str(BENCH), 'evaluate', run, '--scope', args.profile], folder/'evaluation'/'summary.json'),
                  ('diagnose', [sys.executable, str(BENCH), 'diagnose', run], folder/'diagnostics'/'summary.json')]
        for stage, command, artifact in stages:
            if stage == 'evaluate' and args.skip_evaluation:
                continue
            if artifact.exists():
                record[stage] = 'existing'
                continue
            if stage == 'train' and folder.exists():
                record['error'] = 'Incomplete training directory exists; use a new prefix to preserve it'
                break
            command += ['--device', args.device, '--seed', str(args.seed), '--threads', str(args.threads)]
            if stage == 'train' and args.hidden is not None:
                command += ['--hidden', str(args.hidden)]
            print(f'Running {stage}: {run}', flush=True)
            result = subprocess.run(command, cwd=ROOT)
            record[stage] = 'complete' if result.returncode == 0 else f'failed ({result.returncode})'
            if result.returncode:
                record['error'] = f'{stage} failed'
                break
        records.append(record)
        manifest.write_text(json.dumps(records, indent=2), encoding='utf-8')
    return int(any('error' in r for r in records))


if __name__ == '__main__':
    sys.exit(main())
