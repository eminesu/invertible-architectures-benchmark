"""Plan FIF/CFMPE experiments on FR3. Nothing runs without --execute.

Defaults match the team's headline budget: ~3M parameters, 1M training rows,
20k validation rows, 50 epochs, batch 1000, seed 0. Stages are separate so
samplers can be chosen on validation before scoring the held-out test poses.

    python experiments/run_fr3_fif_cfmpe.py --stage train
    python experiments/run_fr3_fif_cfmpe.py --stage train --budgets 300000 1000000 3000000
    python experiments/run_fr3_fif_cfmpe.py --stage val
    python experiments/run_fr3_fif_cfmpe.py --stage test --methods cfmpe --solver euler:64

Append --execute to actually run the printed commands. The gt stage is also
explicit: it generates both exact reference caches using the existing tracer.
val/test can generate missing caches through the shared evaluator as well.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SOLVERS = ['midpoint:8', 'midpoint:16', 'midpoint:32', 'euler:32', 'euler:64']


def run_name(args, method, budget, seed):
    architecture = f'_{args.fif_arch}' if method == 'fif' else ''
    return (f'{args.problem}_{method}{architecture}_p{budget}_n{args.n_train}'
            f'_e{args.epochs}_s{seed}' + (f'_{args.tag}' if args.tag else ''))


def commands(args):
    if args.stage == 'gt':
        return [[sys.executable, '-m', 'robot_ik.scripts.generate_gt', args.problem,
                 '--split', split, '--device', args.device] for split in ('val', 'test')]
    planned = []
    for budget in args.budgets:
        for seed in args.seeds:
            for method in args.methods:
                name = run_name(args, method, budget, seed)
                if args.stage == 'train':
                    script = 'train_fif.py' if method == 'fif' else 'train_conditional_fmpe.py'
                    command = [sys.executable, f'experiments/{script}', '--problem', args.problem,
                               '--run' if method == 'fif' else '--run-name', name,
                               '--device', args.device, '--threads', str(args.threads),
                               '--seed', str(seed), '--param-budget', str(budget),
                               '--n-train', str(args.n_train), '--n-val', str(args.n_val),
                               '--epochs', str(args.epochs), '--batch-size', str(args.batch_size)]
                    if method == 'fif':
                        command += ['--arch', args.fif_arch, '--sigma', '0']
                    else:
                        command += ['--alpha', '1', '--nominal-fraction', str(args.nominal_fraction)]
                else:
                    script = 'evaluate_fif.py' if method == 'fif' else 'evaluate_conditional_fmpe.py'
                    command = [sys.executable, f'experiments/{script}', name,
                               '--split', args.stage, '--device', args.device,
                               '--threads', str(args.threads), '--seed', str(args.evaluation_seed)]
                    if method == 'cfmpe':
                        command += ['--solver', *(args.solver or VALIDATION_SOLVERS),
                                    '--chunk', str(args.chunk)]
                        if args.redraw_nonfinite:
                            command.append('--redraw-nonfinite')
                planned.append(command)
    return planned


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=('gt', 'train', 'val', 'test'), required=True)
    p.add_argument('--problem', choices=('fr3_6d', 'fr3_5d'), default='fr3_6d')
    p.add_argument('--methods', nargs='+', choices=('fif', 'cfmpe'), default=['fif', 'cfmpe'])
    p.add_argument('--budgets', nargs='+', type=int, default=[3_000_000])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--fif-arch', choices=('mlp', 'paper'), default='mlp')
    p.add_argument('--nominal-fraction', type=float, default=0.25)
    p.add_argument('--solver', nargs='+', help='method:steps; required and singular on test for CFMPE')
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--redraw-nonfinite', action='store_true')
    p.add_argument('--evaluation-seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--tag', default='', help='Optional folder suffix to distinguish experiment protocols')
    p.add_argument('--execute', action='store_true', help='Actually run; without this flag, print the plan only')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(*args.budgets, args.n_train, args.n_val, args.batch_size, args.epochs, args.threads, args.chunk) < 1:
        p.error('Sizes, budgets, epochs, threads and chunk must be positive')
    if len(args.budgets) != len(set(args.budgets)) or len(args.seeds) != len(set(args.seeds)):
        p.error('Budgets and seeds must be unique')
    if len(args.methods) != len(set(args.methods)):
        p.error('Methods must be unique')
    if args.tag and any(not (c.isalnum() or c in '_-') for c in args.tag):
        p.error('--tag may contain only letters, numbers, underscores and hyphens')
    if not 0 <= args.nominal_fraction < 1:
        p.error('--nominal-fraction must be in [0, 1)')
    if args.stage == 'test' and 'cfmpe' in args.methods and (args.solver is None or len(args.solver) != 1):
        p.error('Choose one CFMPE solver on validation first, then supply --solver method:steps for test')
    for solver in args.solver or []:
        method, separator, steps = solver.partition(':')
        if method not in ('euler', 'midpoint', 'rk4') or separator != ':' or not steps.isdigit() or int(steps) < 1:
            p.error(f'Invalid solver {solver!r}')
    planned = commands(args)
    print('Executing commands.' if args.execute else 'Plan only. No training, sampling or cache generation will run.')
    for command in planned:
        print(subprocess.list2cmdline(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
    return planned


if __name__ == '__main__':
    main()
