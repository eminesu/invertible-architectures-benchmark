"""Score a trained INN / cINN run with the shared metrics (metrics.evaluate).

    .venv/bin/python experiments/evaluate_inn.py kin_inn_ml_s0
    .venv/bin/python experiments/evaluate_inn.py kin_cinn_s0 --name kin_cinn_s0 --device cpu

Rebuilds the model of a train_inn.py run from its config.json + model.pt and writes
metrics/results/<benchmark>_<name>.{json,npz} (the json gets the parameter count and device
added), which experiments/summarize_fmpe.py already tables next to the other models.
Inference time is measured on --device; use cpu to compare with the other models.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from experiments.train_inn import build_model
from metrics.evaluate import RESULTS_DIR, evaluate
from models.inn import count_parameters

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_inn.py run from its config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    problem = MODELS[cfg['problem']]()
    model = build_model(cfg, problem.n_parameters, problem.n_observations)
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--name', default=None, help='result name; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the z draws')
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    torch.manual_seed(args.seed)
    name = args.name or args.run
    summary = evaluate(lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32, device=args.device), n),
                       cfg['problem'], name=name)
    summary.update(n_params=count_parameters(model), device=args.device)
    (RESULTS_DIR / f"{cfg['problem']}_{name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
