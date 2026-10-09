"""Diffusion posterior estimation (NPSE) with BayesFlow's DiffusionModel
(bayesflow 2.0.14, `bf.networks.DiffusionModel`, Keras 3 on the torch backend),
the spec's primary tool. Trained on the same data and schedule as the baselines
(train_mdn.py: 1M samples, 50 epochs, batch 1000, Adam + cosine, ~3M
parameters), then scored with metrics.evaluate.

    .venv/bin/python experiments/bayesflow_npse.py --problem kinematics --device mps --sampler
    .venv/bin/python experiments/bayesflow_npse.py --problem kinematics --evaluate-only --sampler euler:32 euler_maruyama:128

The network is BayesFlow's default for DiffusionModel (TimeMLP: 5 residual
blocks, Fourier time embedding, mish) with all library defaults except the
width, which is solved to fit --param-budget. Fixed formulation, the same on
both benchmarks: variance-preserving (VP) cosine schedule on log-SNR in
[-12, 12], velocity prediction (v-parameterization) trained with the noise loss
and sigmoid weighting (Kingma et al. 2023). Samplers: the reverse SDE
(euler_maruyama, or the library default two_step_adaptive) and the
probability-flow ODE (euler, rk45, tsit5); the number of network evaluations
per sample is counted and stored with every result.

Writes experiments/runs/<run-name>/{config.json,log.csv,approximator.keras} and
metrics/results/<benchmark>_<run-name>_<method>x<steps>.{json,npz}.
"""

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('TQDM_DISABLE', '1')  # BayesFlow's per-call sampling progress bars
# Keras picks the torch device at import time (MPS if available), so --device is
# read before importing it. Time inference on cpu, like the other models.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='cpu')
os.environ['KERAS_TORCH_DEVICE'] = _pre.parse_known_args()[0].device

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS, make_dataset  # noqa: E402
from metrics.benchmarks import load_test_conditions  # noqa: E402
from metrics.evaluate import RESULTS_DIR, evaluate  # noqa: E402

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def build(x_dims, y_dims, width):
    network = bf.networks.DiffusionModel(subnet_kwargs={'widths': (width,) * 5})
    approximator = bf.ContinuousApproximator(inference_network=network, standardize='all')
    approximator.build_from_data({'inference_variables': np.zeros((2, x_dims), 'float32'),
                                  'inference_conditions': np.zeros((2, y_dims), 'float32')})
    return approximator


def count_nfe(approximator, y, method, steps):
    """Network evaluations per sample, counted by wrapping DiffusionModel.velocity,
    which both the ODE and the SDE samplers call once per evaluation (adaptive
    steps: averaged over the given conditions)."""
    net = approximator.inference_network
    calls, velocity = [0], net.velocity
    def counted(*a, **kw):
        calls[0] += 1
        return velocity(*a, **kw)
    net.velocity = counted
    try:
        approximator.sample(num_samples=1, conditions={'inference_conditions': y}, method=method, steps=steps)
    finally:
        net.velocity = velocity
    return calls[0]


def width_for_budget(x_dims, y_dims, target):
    def count(w):
        return sum(int(np.prod(v.shape)) for v in build(x_dims, y_dims, w).trainable_weights)
    lo, hi = 8, 2048
    while lo < hi:  # largest width with count <= target
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if count(mid) <= target else (lo, mid - 1)
    return lo, count(lo)


class EpochLog(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.writer = csv.writer(open(path, 'w', newline=''))
        self.writer.writerow(['epoch', 'train_loss', 'val_loss', 'seconds'])
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.writer.writerow([epoch, logs.get('loss'), logs.get('val_loss'), time.time() - self.t0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem', choices=list(MODELS), required=True)
    p.add_argument('--param-budget', type=int, default=3_000_000)
    p.add_argument('--n-train', type=int, default=1_000_000)
    p.add_argument('--n-val', type=int, default=20_000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sampler', nargs='*', default=['euler:32'],
                   help="method:steps for evaluation ('adaptive' steps accepted); none: train only")
    p.add_argument('--evaluate-only', action='store_true', help='load the saved run instead of training')
    p.add_argument('--device', default='cpu', help='torch device for Keras (training: mps is faster)')
    p.add_argument('--run-name', default=None)
    p.add_argument('--name', default=None, help='result name prefix; default: the run name')
    args = p.parse_args()

    keras.utils.set_random_seed(args.seed)
    problem = MODELS[args.problem]()
    y_dims, x_dims = problem.n_observations, problem.n_parameters
    run_name = args.run_name or f'{args.problem[:3]}_bf_npse_s{args.seed}'
    out = os.path.join(RUNS, run_name)

    if args.evaluate_only:
        approximator = keras.saving.load_model(os.path.join(out, 'approximator.keras'))
        cfg = json.load(open(os.path.join(out, 'config.json')))
    else:
        width, n_params = width_for_budget(x_dims, y_dims, args.param_budget)
        os.makedirs(out, exist_ok=True)
        cfg = {**vars(args), 'width_resolved': width, 'n_params': n_params, 'bayesflow': bf.__version__,
               'keras': keras.__version__, 'torch': torch.__version__}
        json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=2)
        print(f'{run_name}: TimeMLP width {width} x 5 blocks, {n_params:,} trainable parameters')

        # Same data seeds as train_mdn.py / train_fmpe.py.
        x_tr, y_tr = make_dataset(args.problem, args.n_train, seed=10_000 + args.seed)
        x_va, y_va = make_dataset(args.problem, args.n_val, seed=20_000 + args.seed)
        train = bf.datasets.OfflineDataset({'inference_variables': x_tr, 'inference_conditions': y_tr},
                                           batch_size=args.batch_size, adapter=None)
        val = bf.datasets.OfflineDataset({'inference_variables': x_va, 'inference_conditions': y_va},
                                         batch_size=10_000, adapter=None, shuffle=False)

        approximator = build(x_dims, y_dims, width)
        steps = args.epochs * (args.n_train // args.batch_size)
        approximator.compile(optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(args.lr, steps),
            weight_decay=args.weight_decay, clipnorm=args.grad_clip))
        t0 = time.time()
        approximator.fit(dataset=train, validation_data=val, epochs=args.epochs, verbose=2,
                         callbacks=[EpochLog(os.path.join(out, 'log.csv'))])
        cfg['train_seconds'] = time.time() - t0
        json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=2)
        approximator.save(os.path.join(out, 'approximator.keras'))

    def sampler(method, steps):
        def sample(y, n):
            # Chunk the conditions: BayesFlow integrates all M * n states at once.
            xs = [approximator.sample(num_samples=n, conditions={'inference_conditions': y[s:s + 100]},
                                      method=method, steps=steps)['inference_variables']
                  for s in range(0, len(y), 100)]
            return np.concatenate(xs)
        return sample

    for spec in args.sampler:
        method, steps = spec.split(':')
        steps = steps if steps == 'adaptive' else int(steps)
        name = f'{args.name or run_name}_{method}x{steps}'
        nfe = count_nfe(approximator, np.asarray(load_test_conditions(args.problem)[0][:100], 'float32'),
                        method, steps)
        print(f'--- {name}: {nfe} NFE per sample', flush=True)
        torch.manual_seed(args.seed)
        load_before = os.getloadavg()  # timing is only comparable on an otherwise idle machine
        summary = evaluate(sampler(method, steps), args.problem, name=name)
        summary.update(method=method, steps=steps, nfe=nfe, n_params=cfg['n_params'], device=args.device,
                       implementation=f"bayesflow {cfg['bayesflow']}", load_avg_before=load_before,
                       seconds_per_nfe=summary['inference_seconds'] / nfe)
        (RESULTS_DIR / f'{args.problem}_{name}.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
