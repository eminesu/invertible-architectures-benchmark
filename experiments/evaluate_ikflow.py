"""Score a trained IKFlow run with the shared metrics (metrics.evaluate).

    .venv/bin/python experiments/evaluate_ikflow.py fr3_6d_ikflow --split val
    .venv/bin/python experiments/evaluate_ikflow.py fr3_6d_ikflow --split val --latent-scale 0.25

--latent-scale is the std of z at sampling time. The default, 1.0, samples the model's own
distribution and is the setting to compare Err_post / coverage across methods; the paper's setting
is 0.25 (more accurate samples, less diverse: its Fig. 7), which is saved as <name>_ls0.25 so both
can be scored next to each other (experiments/summarize_fr3.py lists them as separate rows).
Rebuilds the model of a train_ikflow.py run from its config.json + model.pt and writes
metrics/results/<benchmark>_<name>.{json,npz}. Inference time is measured on --device.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.toy_data import MODELS
from experiments.train_ikflow import build_model
from metrics.evaluate import RESULTS_DIR, evaluate
from models.ikflow import count_parameters

RUNS = os.path.join(os.path.dirname(__file__), 'runs')


def load_run(run, device='cpu'):
    """Rebuild the model of a train_ikflow.py run from its config.json + model.pt."""
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
    p.add_argument('--latent-scale', type=float, default=1.0, help='std of z when sampling (paper: 0.25)')
    p.add_argument('--split', default='test', choices=['test', 'val'],
                   help="'val' (FR3 tasks only): select models on it, report on 'test'; results are saved as <name>_val")
    args = p.parse_args()

    model, cfg = load_run(args.run, args.device)
    model.latent_scale = args.latent_scale
    torch.manual_seed(args.seed)
    name = args.name or args.run
    if args.latent_scale != 1.0:
        name = f'{name}_ls{args.latent_scale:g}'
    if args.split != 'test':
        name = f'{name}_{args.split}'   # metrics.evaluate names its files by benchmark and name only: do not overwrite the test result
    summary = evaluate(lambda y, n: model.sample(torch.as_tensor(y, dtype=torch.float32, device=args.device), n),
                       cfg['problem'], name=name, split=args.split)
    summary.update(n_params=count_parameters(model), device=args.device, latent_scale=args.latent_scale)
    (RESULTS_DIR / f"{cfg['problem']}_{name}.json").write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
