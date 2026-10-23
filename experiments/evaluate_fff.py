"""Score a trained conditional FFF run (Euclidean or torus) with the shared
metrics (metrics.evaluate).

    .venv/bin/python experiments/evaluate_fff.py kin_fff_s0 kin_mfff_s0

Sampling is one decoder pass per sample (1 NFE). Writes
metrics/results/<benchmark>_<run>.{json,npz}; the json gets the manifold, beta,
NFE, parameter count (encoder + decoder; only the decoder runs at sampling) and
the mean reconstruction error on the test ground-truth samples added, since the
likelihood is only valid when that is near zero. Inference time is measured on
--device; use cpu to compare with the other models' numbers.
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
def reconstruction_error(model, benchmark):
    """Mean ambient ||u - g(f(u))||^2 over the ground-truth posterior samples of
    all test y*, i.e. on the data the posteriors live on."""
    y_star, _ = load_test_conditions(benchmark)
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
    args = p.parse_args()

    for run in args.runs:
        model, cfg = load_run(run, args.device)
        torch.manual_seed(args.seed)
        print(f"--- {run}: {cfg['manifold']}, beta {cfg['beta']:g}, 1 NFE per sample", flush=True)
        summary = evaluate(lambda y, n: model.sample(y, n), cfg['problem'], name=run)
        summary.update(manifold=cfg['manifold'], beta=cfg['beta'], nfe=1, n_params=count_parameters(model),
                       n_params_decoder=count_parameters(model.decoder), device=args.device,
                       recon_gt=reconstruction_error(model, cfg['problem']))
        print(f"{'':<28} reconstruction error on GT samples: {summary['recon_gt']:.2e}")
        (RESULTS_DIR / f"{cfg['problem']}_{run}.json").write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
