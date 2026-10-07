"""Score a trained autoencoder run with the shared metrics (metrics.evaluate).

    .venv/bin/python experiments/evaluate_autoencoder.py kin_ae_s0 [--name ae_s0] [--device cuda]

Writes metrics/results/<benchmark>_<name>.{json,npz}. Time inference on a GPU
for reported numbers (docs/autoencoder.md, Finding 2).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from metrics.evaluate import evaluate
from models.autoencoder import Autoencoder

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_autoencoder.py run from its config.json + model.pt."""
    cfg = json.load(open(os.path.join(RUNS, run, 'config.json')))
    problem = MODELS[cfg['problem']]()
    model = Autoencoder(problem.n_parameters, problem.n_observations, cfg['hidden_resolved'], cfg['layers'],
                        cfg['activation'], cfg['latent_mmd'])
    model.load_state_dict(torch.load(os.path.join(RUNS, run, 'model.pt'), map_location=device))
    return model.to(device).eval(), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--name', default=None, help='result name; default: the run name')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0, help='seed for the z draws')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_run(args.run, args.device)
    evaluate(model.sample, cfg['problem'], name=args.name or args.run)


if __name__ == '__main__':
    main()
