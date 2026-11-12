"""Score a trained conditional FFF run (Euclidean or torus) with the shared
metrics (metrics.evaluate).

    .venv/bin/python experiments/evaluate_fff.py kin_fff_s0 kin_mfff_s0

Sampling is one decoder pass per sample (1 NFE). Writes
metrics/results/<benchmark>_<run>.{json,npz}; the json gets the manifold, beta,
NFE, parameter count (encoder + decoder; only the decoder runs at sampling) and
the mean reconstruction error on the test ground-truth samples added, since the
likelihood is only valid when that is near zero. Inference time is measured on
--device; use cpu to compare with the other models' numbers.

--split val (FR3 tasks) scores the validation conditions and writes the files with a
_val suffix, so they never overwrite the test ones. On the FR3 tasks the json also
gets the per-joint share of samples outside the URDF joint limits
(joint_limit_violation_per_joint), on the same samples that were scored.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from experiments.train_fff import build_model
from metrics.benchmarks import load_test_conditions
from metrics.evaluate import RESULTS_DIR, evaluate
from metrics.ground_truth import ground_truth_posterior
from models.fff import count_parameters

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_fff.py run from its config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    model = build_model(cfg, MODELS[cfg['problem']]())
    keys = model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device), strict=False)
    # angle_std was added with the wrapped latent; uniform-latent runs trained before do not need it.
    assert not keys.unexpected_keys, keys
    assert not keys.missing_keys or (keys.missing_keys == ['angle_std'] and model.latent == 'uniform'), keys
    return model.to(device).eval(), cfg


@torch.no_grad()
def reconstruction_error(model, benchmark, split='test'):
    """Mean ambient ||u - g(f(u))||^2 over the ground-truth posterior samples of
    all y* of the split, i.e. on the data the posteriors live on."""
    y_star, _ = load_test_conditions(benchmark, split)
    gt = ground_truth_posterior(benchmark, y_star, verbose=False)
    x = torch.as_tensor(np.concatenate(gt), dtype=torch.float32, device=model.y_mean.device)
    y = torch.as_tensor(np.repeat(y_star, [len(g) for g in gt], 0), dtype=torch.float32, device=x.device)
    total = 0.0
    for s in range(0, len(x), 100_000):
        u, c = model.embed(x[s:s + 100_000]), model.context(y[s:s + 100_000])
        total += ((u - model.decode(model.encode(u, c), c)) ** 2).sum().item()
    return total / len(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the latent draws')
    p.add_argument('--split', default='test', choices=['test', 'val'],
                   help="conditions to score: 'test' (reported) or 'val' (model selection, FR3 tasks)")
    args = p.parse_args()

    for run in args.runs:
        model, cfg = load_run(run, args.device)
        bench = cfg['problem']
        torch.manual_seed(args.seed)
        name = run + ('_val' if args.split == 'val' else '')
        print(f"--- {name}: {cfg['manifold']}, beta {cfg['beta']:g}, 1 NFE per sample", flush=True)
        last = {}

        def sample_fn(y, n):  # keeps the latest (= scored; the earlier calls are warm-up) samples
            last['x'] = model.sample(y, n)
            return last['x']

        summary = evaluate(sample_fn, bench, name=name, split=args.split)
        summary.update(manifold=cfg['manifold'], latent=cfg.get('latent') if cfg['manifold'] == 'torus' else None,
                       beta=cfg['beta'], y_noise=cfg.get('y_noise'), nfe=1, n_params=count_parameters(model),
                       n_params_decoder=count_parameters(model.decoder), device=args.device, split=args.split,
                       recon_gt=reconstruction_error(model, bench, args.split))
        if bench.startswith('fr3'):
            from robot_ik.kinematics import joint_limits
            lo, hi = (torch.as_tensor(a, dtype=torch.float32, device=last['x'].device) for a in joint_limits())
            out = (last['x'] < lo) | (last['x'] > hi)
            summary['joint_limit_violation_per_joint'] = out.float().mean((0, 1)).tolist()
            print(f"{'':<28} outside limits per joint: "
                  + ' '.join(f'{v:.4f}' for v in summary['joint_limit_violation_per_joint']))
        print(f"{'':<28} reconstruction error on GT samples: {summary['recon_gt']:.2e}")
        (RESULTS_DIR / f"{bench}_{name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
