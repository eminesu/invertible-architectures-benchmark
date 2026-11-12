"""Read-only audit of existing FR3 training recipes and a bounded CUDA pilot.

No training, checkpoint writes, GT generation, or shared-class modifications.
Full training arrays are reconstructed in RAM and discarded; only summaries,
hashes, figures, and small pilot files are written to the chosen output folder.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.toy_data import MODELS
from experiments.fr3_fif_cfmpe_adapters import make_conditional_dataset
from experiments.train_fif import dataset as fif_dataset
from metrics.benchmarks import conditions_path, load_test_conditions
from robot_ik import kinematics as K

DEFAULT_RUNS = [f'fr3_{d}_{m}_s0' for d in ('6d', '5d') for m in ('fif', 'cfmpe')]
FK_TOL = 0.000002  # float32 storage of joints/poses; tested against float64 FK
LIMIT_TOL = 0.0000005  # float32 rounding at a URDF endpoint


def fixed_json(value):
    """Preserve numeric precision without exponent notation, including tiny errors."""
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError('Non-finite JSON statistic')
        result = np.format_float_positional(float(value), unique=True, trim='-')
        if '.' not in result:
            result += '.'
        return result + '0' * max(0, 5 - len(result.split('.')[1]))
    if isinstance(value, dict):
        return '{' + ',\n'.join(json.dumps(str(k)) + ': ' + fixed_json(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple, np.ndarray)):
        return '[' + ', '.join(fixed_json(v) for v in value) + ']'
    if isinstance(value, np.bool_):
        value = bool(value)
    return json.dumps(value, ensure_ascii=False)


def save_json(path, value):
    path.write_text(fixed_json(value) + '\n', encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f'{v:.5f}' if isinstance(v, (float, np.floating)) else v
                             for k, v in row.items()})


def sha_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def row_keys(array):
    array = np.ascontiguousarray(array)
    return array.view(np.dtype((np.void, array.dtype.itemsize * array.shape[1]))).ravel()


def overlap(a, b):
    return int(np.intersect1d(row_keys(a), row_keys(b), assume_unique=False).size)


def resimulation(x, y, problem):
    max_abs = 0.0
    sum_sq = 0.0
    bad = 0
    for start in range(0, len(x), 20000):
        predicted = MODELS[problem]().forward_process(x[start:start + 20000])
        delta = predicted - y[start:start + 20000]
        max_abs = max(max_abs, float(np.abs(delta).max()))
        sum_sq += float(np.square(delta).sum())
        bad += int((np.abs(delta).max(axis=1) > FK_TOL).sum())
    return {'max_abs_encoded': max_abs, 'max_abs_encoded_micro_units': max_abs * 1000000,
            'rms_per_encoded_component': float(np.sqrt(sum_sq / y.size)),
            'rows_over_tolerance': bad, 'tolerance': FK_TOL}


def normalization(x, y, c, method, state):
    xt = torch.from_numpy(x)
    context = torch.from_numpy(y if c is None else np.concatenate([y, c], axis=1))
    if method == 'FIF':
        xs = xt.var(0, unbiased=False).sqrt().clamp_min(0.000001)
        ys = context.std(0, unbiased=False).clamp_min(0.000001)
    else:
        xs, ys = xt.std(0), context.std(0).clamp_min(0.000001)
    expected = dict(x_mean=xt.mean(0), x_std=xs, y_mean=context.mean(0), y_std=ys)
    errors = {k: float((state[k].cpu() - v).abs().max()) for k, v in expected.items()}
    return {'max_buffer_abs_errors': errors,
            'all_close': all(v <= 0.000001 for v in errors.values()),
            'min_x_std': float(xs.min()), 'min_context_std': float(ys.min()),
            'context_dimensions': context.shape[1],
            'std_convention': 'population' if method == 'FIF' else 'sample (correction=1)'}


def geometry(x, task, count, seed):
    from scipy.spatial import cKDTree
    indices = np.random.RandomState(seed).choice(len(x), min(count, len(x)), replace=False)
    q = torch.as_tensor(x[indices], dtype=torch.float64)
    J, _, R = K.jacobian(q)
    if task == '5d':
        a = R[:, :, 2]
        u1, u2 = K.tangent_basis(a)
        da = torch.cross(J[:, 3:, :], a[:, :, None].expand_as(J[:, 3:, :]), dim=1)
        J = torch.cat([J[:, :3, :], (u1[:, :, None] * da).sum(1)[:, None, :],
                       (u2[:, :, None] * da).sum(1)[:, None, :]], dim=1)
    # Position rows divided by characteristic length 1 m, angle rows in rad.
    sv = torch.linalg.svdvals(J).numpy()
    condition = sv[:, 0] / sv[:, -1]
    lo, hi = K.joint_limits()
    z = (x[indices].astype(np.float64) - lo) / (hi - lo)
    distances = cKDTree(z).query(z, k=2)[0][:, 1]
    return {'sample_count': len(q), 'characteristic_length_m': 1.0,
            'rank_relative_tolerance': 0.0000000001,
            'rank_deficient_count': int((sv[:, -1] <= sv[:, 0] * 0.0000000001).sum()),
            'near_singular_condition_over_1000': int((condition > 1000).sum()),
            'condition_median': float(np.median(condition)),
            'condition_p95': float(np.quantile(condition, 0.95)),
            'condition_max': float(condition.max()), 'smallest_singular_min': float(sv[:, -1].min()),
            'regular_task_rank': J.shape[1], 'regular_posterior_dimension': 7 - J.shape[1],
            'nearest_neighbor_unit_box_min': float(distances.min()),
            'nearest_neighbor_unit_box_median': float(np.median(distances)),
            'rows_with_neighbor_within_0.000001_in_subsample': int((distances < 0.000001).sum())}, condition


def rotation_checks(y, task):
    a = y[:, 3:6].astype(np.float64)
    normerr = float(np.abs((a * a).sum(1) - 1).max())
    result = {'max_unit_vector_squared_norm_error': normerr}
    if task == '6d':
        b = y[:, 6:9].astype(np.float64)
        result['max_second_vector_squared_norm_error'] = float(np.abs((b * b).sum(1) - 1).max())
        result['max_column_dot_abs'] = float(np.abs((a * b).sum(1)).max())
        # Before any Gram-Schmidt correction, so this tests the saved encoding.
        R = np.stack([a, b, np.cross(a, b)], axis=2)
        result['max_determinant_error'] = float(np.abs(np.linalg.det(R) - 1).max())
    result['valid'] = all(v <= FK_TOL for v in result.values())
    return result


def distribution(x, y, c, run, split, rows, uniform_rows):
    from scipy.stats import chisquare
    lo, hi = K.joint_limits()
    u = (x.astype(np.float64) - (lo + hi) / 2) / ((hi - lo) / 2)
    if c is not None:
        u /= c
    for j in range(7):
        hist, _ = np.histogram(u[:, j], bins=np.linspace(-1.000001, 1.000001, 41))
        test = chisquare(hist)
        uniform_rows.append(dict(run=run, split=split, joint=j + 1,
                                 chi_square=float(test.statistic), p_value=float(test.pvalue),
                                 # Family-wise threshold for 4 runs x 2 splits x 7 joints.
                                 passes_bonferroni_0_01=bool(test.pvalue > 0.01 / 56)))
    groups = [('x', x), ('y', y)] + ([] if c is None else [('c', c)])
    for group, values in groups:
        for j in range(values.shape[1]):
            v = values[:, j].astype(np.float64)
            q = np.quantile(v, [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1])
            rows.append(dict(run=run, split=split, group=group, dimension=j + 1,
                             count=len(v), minimum=q[0], p0_1=q[1], p1=q[2], median=q[3],
                             p99=q[4], p99_9=q[5], maximum=q[6], mean=float(v.mean()), std=float(v.std())))
    return {'max_abs_joint_correlation': float(np.abs(np.corrcoef(u.T) - np.eye(7)).max()),
            'conditional_unit_box_outside_tolerance': int((np.abs(u) > 1.000001).any(1).sum()),
            'nominal_rows': len(x) if c is None else int((c == 1).all(1).sum()),
            'nominal_fraction': 1.0 if c is None else float((c == 1).all(1).mean())}, u


def plots(out, run, x, y, c, u, condition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    lo, hi = K.joint_limits()
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for j, ax in enumerate(axes.flat):
        if j >= 7:
            ax.axis('off')
            continue
        ax.hist(x[:, j], bins=np.linspace(lo[j], hi[j], 51), density=True,
                color='#336699', alpha=0.75, label='All training rows')
        if c is not None:
            ax.hist(x[(c == 1).all(1), j], bins=np.linspace(lo[j], hi[j], 51), density=True,
                    histtype='step', color='#b65c24', label='Nominal-prior rows')
        ax.axhline(1 / (hi[j] - lo[j]), color='black', ls='--', lw=1, label='Nominal uniform density')
        ax.set(title=f'Joint {j + 1}', xlabel='Angle (rad)', ylabel='Density')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(run + ': bounded joint distributions')
    fig.tight_layout()
    fig.savefig(out / f'{run}_joints.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    labels = ['p_x (m)', 'p_y (m)', 'p_z (m)'] + (['r1_x', 'r1_y', 'r1_z', 'r2_x', 'r2_y', 'r2_z']
                                                                if y.shape[1] == 9 else ['axis_x', 'axis_y', 'axis_z'])
    for j, ax in enumerate(axes.flat):
        if j >= y.shape[1]:
            ax.axis('off')
            continue
        ax.hist(y[:, j], bins=60, density=True, color='#39856a')
        ax.set(title=labels[j], ylabel='Density')
    fig.suptitle(run + ': encoded pose marginals (workspace is not uniform)')
    fig.tight_layout()
    fig.savefig(out / f'{run}_poses.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(7):
        axes[0].hist(u[:, j], bins=40, range=(-1, 1), density=True, histtype='step', label=f'q{j + 1}')
    axes[0].set(xlabel='Joint offset / conditional half-width', ylabel='Density', title='Expected density: 0.50000')
    axes[0].legend(fontsize=7)
    axes[1].hist(np.log10(condition), bins=40, color='#8469a6')
    axes[1].set(xlabel='log10(Jacobian condition number)', ylabel='Count', title='Scaled task Jacobian; sampled rows')
    for ax in axes:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
    fig.tight_layout()
    fig.savefig(out / f'{run}_conditioning.png', dpi=140)
    plt.close(fig)


def independent_checks():
    with np.load(ROOT / 'tests/fixtures/jrl_fr3_fk.npz') as f:
        q, T = f['q'], f['T_hand']
        fixture_meta = {k: str(f[k]) for k in ('jrl_commit', 'end_effector')}
    p, R = K.fk(q)
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos = float(np.abs(p.numpy() - T[:, :3, 3]).max())
    rot = float(np.abs(R.numpy() @ Rz - T[:, :3, :3]).max())
    lo, hi = K.joint_limits()
    q0 = (lo + hi)[None, :] / 2
    p0, R0 = K.fk(q0)
    periodic = []
    for j in range(7):
        moved = q0.copy()
        moved[:, j] += 2 * np.pi
        pp, RR = K.fk(moved)
        periodic.append(float(max((pp - p0).abs().max(), (RR - R0).abs().max())))
    return {'jrl': {'rows': len(q), **fixture_meta, 'position_max_abs_m': pos,
                    'position_max_abs_nanometres': pos * 1000000000,
                    'rotation_matrix_max_abs': rot, 'passes': max(pos, rot) <= FK_TOL},
            'angular_fk_periodicity_max_abs_per_joint': periodic,
            'periodicity_probe_outside_prior': True,
            'joint_types': [j.attrib['type'] for j in ET.parse(K.URDF).getroot().findall('joint') if j.attrib['type'] != 'fixed'],
            'lower_rad': lo, 'upper_rad': hi, 'width_rad': hi - lo,
            'all_widths_less_than_full_turn': bool(((hi - lo) < 2 * np.pi).all())}


def audit_conditions():
    result = {}
    for task in ('6d', '5d'):
        problem = 'fr3_' + task
        for split in ('val', 'test'):
            y, x = load_test_conditions(problem, split)
            result[f'{problem}_{split}'] = {'count': len(x), 'source': str(conditions_path(problem, split).relative_to(ROOT)),
                                           'sha256': sha_file(conditions_path(problem, split)),
                                           'resimulation': resimulation(x, y, problem)}
        vx, tx = (load_test_conditions(problem, s)[1] for s in ('val', 'test'))
        result[f'{problem}_val_test_overlap'] = overlap(vx.astype(np.float32), tx.astype(np.float32))
    return result


def audit_cached_references():
    """Bounded, deterministic sample check; no tracing or cache writes."""
    lo, hi = K.joint_limits()
    results = {}
    for path in sorted((ROOT / 'metrics/cache').glob('gt_fr3_*.npz')):
        problem = 'fr3_6d' if 'fr3_6d_' in path.name else 'fr3_5d'
        with np.load(path) as f:
            y = f['y_star']
            item = {'sha256': sha_file(path), 'poses': len(y), 'checks_per_pose_per_draw_set': 16,
                    'all_closed': bool(f['diag_all_closed'].all()),
                    'loops_min': int(f['diag_n_loops'].min()), 'loops_max': int(f['diag_n_loops'].max()),
                    'has_generating_configuration_on_fiber_diagnostic': 'diag_x_true_on_fiber' in f.files}
            for key in ('samples', 'oracle'):
                all_samples = f[key]
                indices = np.linspace(0, all_samples.shape[1] - 1, min(16, all_samples.shape[1]), dtype=int)
                selected = all_samples[:, indices, :].reshape(-1, 7)
                paired = np.repeat(y, len(indices), axis=0)
                item[key] = {'shape': list(all_samples.shape), 'finite_entire_array': bool(np.isfinite(all_samples).all()),
                             'checked_rows': len(selected),
                             'limit_violations_checked': int(((selected < lo - LIMIT_TOL) | (selected > hi + LIMIT_TOL)).any(1).sum()),
                             'resimulation': resimulation(selected, paired, problem)}
            results[path.name] = item
    return results


def gpu_pilot(out, sizes, repeats):
    """Local alternate FK path only: do not change the training generators."""
    from matplotlib import pyplot as plt
    device = torch.device('cuda')
    rows = []
    gpu = torch.cuda.get_device_name(0)
    torch.cuda.synchronize()
    # Initialize both chains, kernels, and host allocator outside measurements.
    warm = K.sample_uniform(max(sizes), np.random.RandomState(910))
    with torch.no_grad():
        for _ in range(3):
            K.encode(*K.fk(torch.as_tensor(warm, device=device)), '6d').cpu().numpy()
    torch.cuda.synchronize()
    for n in sizes:
        for repeat in range(repeats):
            rng = np.random.RandomState(1000 + repeat)
            started = time.perf_counter()
            q = K.sample_uniform(n, rng)
            cpu_y = K.encode(*K.fk(q), '6d').numpy().astype(np.float32)
            cpu_seconds = time.perf_counter() - started
            rng = np.random.RandomState(1000 + repeat)
            torch.cuda.synchronize()
            started = time.perf_counter()
            q = K.sample_uniform(n, rng)
            qt = torch.as_tensor(q, device=device)
            gpu_y = K.encode(*K.fk(qt), '6d').cpu().numpy().astype(np.float32)
            torch.cuda.synchronize()
            gpu_seconds = time.perf_counter() - started
            delta = float(np.abs(gpu_y - cpu_y).max())
            path = out / f'pilot_n{n}_r{repeat}.npz'
            started = time.perf_counter()
            np.savez(path, x=q.astype(np.float32), y=gpu_y)
            save_seconds = time.perf_counter() - started
            torch.cuda.synchronize()
            started = time.perf_counter()
            with np.load(path) as data:
                loaded = [torch.from_numpy(data[k]).to(device) for k in ('x', 'y')]
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - started
            rows.append(dict(n=n, repeat=repeat, cpu_generation_s=cpu_seconds,
                             gpu_generation_s=gpu_seconds, save_npz_s=save_seconds,
                             load_and_cuda_transfer_s=load_seconds, file_bytes=path.stat().st_size,
                             cpu_gpu_max_abs=delta))
            del loaded, qt
        print(f'GPU pilot: n={n}, {repeats} repeats complete', flush=True)
    fits = {}
    projections = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ('cpu_generation_s', 'gpu_generation_s', 'save_npz_s', 'load_and_cuda_transfer_s'):
        medians = np.array([np.median([r[metric] for r in rows if r['n'] == n]) for n in sizes])
        slope, intercept = np.polyfit(np.log(sizes), np.log(medians), 1)
        predicted = intercept + slope * np.log(sizes)
        variance = np.square(np.log(medians) - np.log(medians).mean()).sum()
        fits[metric] = {'alpha': float(slope), 'coefficient': float(np.exp(intercept)),
                        'r_squared_log': float(1 - np.square(predicted - np.log(medians)).sum() / variance) if variance else 0.0}
        ax.loglog(sizes, medians, 'o-', label=metric)
        for n in (20000, 1000000, 1020000):
            projections.append(dict(metric=metric, projected_rows=n,
                                    power_law_seconds=float(np.exp(intercept) * n ** slope),
                                    largest_size_linear_seconds=float(medians[-1] * n / sizes[-1])))
    ax.set(xlabel='Rows', ylabel='Median wall time (s)', title='Warm FR3 6D pilot; generation includes CPU output')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out.parent / 'figures/scaling.png', dpi=150)
    plt.close(fig)
    save_csv(out.parent / 'timings.csv', rows)
    save_csv(out.parent / 'projections.csv', projections)
    return {'gpu': gpu, 'dtype_fk': 'float64', 'dtype_stored': 'float32', 'sizes': sizes,
            'repeats': repeats, 'warmup_largest_batch_passes': 3, 'fits': fits,
            'parity_passes': all(r['cpu_gpu_max_abs'] <= FK_TOL for r in rows),
            'maximum_new_gpu_batch': max(sizes), 'scope': '6D nominal generation, NPZ save/load and CUDA transfer; no GT tracing or model training'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'output/fr3_data_suitability')
    p.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    p.add_argument('--conditioning-samples', type=int, default=10000)
    p.add_argument('--pilot-sizes', nargs='+', type=int, default=[1000, 5000, 10000, 20000])
    p.add_argument('--pilot-repeats', type=int, default=5)
    p.add_argument('--skip-gpu-pilot', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        p.error('Choose a new --out folder; existing audit artifacts are never overwritten')
    if len(args.pilot_sizes) < 3 or min(args.pilot_sizes) < 2 or max(args.pilot_sizes) > 20000:
        p.error('Pilot requires at least 3 sizes, each between 2 and 20000; scaling needs separate authorization')
    if args.pilot_repeats < 2 or args.conditioning_samples < 2:
        p.error('Need at least 2 repeats and conditioning rows')
    args.pilot_sizes = sorted(set(args.pilot_sizes))
    if len(args.pilot_sizes) < 3:
        p.error('Need at least 3 distinct pilot sizes')
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    for directory in ('figures', 'pilot'):
        (args.out / directory).mkdir()
    sources = ['robot_ik/tasks.py', 'robot_ik/kinematics.py', 'robot_ik/assets/fr3/fr3.urdf',
               'robot_ik/assets/fr3/PROVENANCE.md', 'data/toy_data.py', 'experiments/train_fif.py',
               'experiments/train_conditional_fmpe.py', 'experiments/fr3_fif_cfmpe_adapters.py',
               'models/fif.py', 'models/fmpe.py', 'tests/fixtures/jrl_fr3_fk.npz',
               'experiments/analyze_fr3_data_suitability.py']
    metadata = {'created_local': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'argv': sys.argv,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status_before': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'python': sys.version, 'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
                'versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'matplotlib', 'pytorch-kinematics')},
                'cuda_runtime': torch.version.cuda, 'source_sha256': {s: sha_file(ROOT / s) for s in sources},
                'historical_limitation': 'Reconstructed with current checked-in recipes and saved configs/seeds; original training-array hashes were not recorded.',
                'training_raw_data_persisted': False}
    save_json(args.out / 'metadata.json', metadata)
    checks = {'independent_checks': independent_checks(), 'conditions': audit_conditions(), 'runs': {}}
    stats, uniform = [], []
    lo, hi = K.joint_limits()
    # A conservative physical workspace radius from the URDF chain translations.
    reach_bound = sum(np.linalg.norm([float(v) for v in j.find('origin').attrib.get('xyz', '0 0 0').split()])
                      for j in ET.parse(K.URDF).getroot().findall('joint') if j.find('origin') is not None)
    for run in args.runs:
        print(f'Auditing {run}', flush=True)
        folder = ROOT / 'experiments/runs' / run
        config = json.loads((folder / 'config.json').read_text())
        problem, method = config['problem'], config['method']
        state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        item = {'problem': problem, 'method': method, 'n_params': config['n_params'],
                'config_sha256': sha_file(folder / 'config.json'), 'checkpoint_sha256': sha_file(folder / 'model.pt'),
                'prior_mode': config['prior_mode'], 'splits': {}}
        train_x = None
        for split, n, seed in zip(('train', 'validation'), (config['n_train'], config['n_val']), config['data_seeds']):
            started = time.perf_counter()
            if method == 'FIF':
                xt, yt, rejected = fif_dataset(problem, n, seed)
                x, y, c = xt.numpy(), yt.numpy(), None
            else:
                x, y, c = make_conditional_dataset(problem, n, seed, config['prior_range'], config['nominal_fraction'])
                rejected = 0
            elapsed = time.perf_counter() - started
            print(f'  {split}: reconstructed {n} rows in {elapsed:.5f}s', flush=True)
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all() and (c is None or np.isfinite(c).all()))
            if not finite:
                raise RuntimeError(f'Non-finite data in {run}/{split}; inspect without silently filtering')
            result = {'n': n, 'seed': seed, 'x_shape': list(x.shape), 'y_shape': list(y.shape),
                      'c_shape': None if c is None else list(c.shape), 'dtype': str(x.dtype),
                      'array_bytes': x.nbytes + y.nbytes + (0 if c is None else c.nbytes),
                      'reconstruction_cpu_seconds': elapsed, 'all_finite': finite, 'rejected_draws': rejected,
                      'sha256_arrays': {k: sha_array(v) for k, v in [('x', x), ('y', y), ('c', c)] if v is not None},
                      'joint_limit_violation_rows': int(((x < lo - LIMIT_TOL) | (x > hi + LIMIT_TOL)).any(1).sum()),
                      'duplicate_joint_rows': len(x) - len(np.unique(row_keys(x))),
                      'duplicate_pose_rows': len(y) - len(np.unique(row_keys(y))),
                      'physical_reach_bound_m': float(reach_bound),
                      'position_outside_reach_bound': int((np.linalg.norm(y[:, :3], axis=1) > reach_bound + FK_TOL).sum()),
                      'orientation': rotation_checks(y, problem[-2:]),
                      'resimulation': resimulation(x, y, problem),
                      'committed_condition_overlap': {s: overlap(x, load_test_conditions(problem, s)[1].astype(np.float32)) for s in ('val', 'test')}}
            result['distribution'], u = distribution(x, y, c, run, split, stats, uniform)
            if split == 'train':
                item['normalization'] = normalization(x, y, c, method, state)
                item['geometry'], condition = geometry(x, problem[-2:], args.conditioning_samples, 781)
                plots(args.out / 'figures', run, x, y, c, u, condition)
                train_x = x.copy()
            else:
                item['train_validation_overlap'] = overlap(train_x, x)
            item['splits'][split] = result
            del x, y, c, u
            gc.collect()
        checks['runs'][run] = item
        save_json(args.out / 'checks.json', checks)
        del state, train_x
        gc.collect()
    print('Checking existing reference-cache subsets (no GT generation)', flush=True)
    checks['cached_reference_subsets'] = audit_cached_references()
    if args.skip_gpu_pilot:
        checks['gpu_pilot'] = {'status': 'skipped explicitly'}
    elif not torch.cuda.is_available():
        checks['gpu_pilot'] = {'status': 'CUDA unavailable; no projection fabricated'}
    else:
        checks['gpu_pilot'] = gpu_pilot(args.out / 'pilot', args.pilot_sizes, args.pilot_repeats)
    save_csv(args.out / 'statistics.csv', stats)
    save_csv(args.out / 'uniformity.csv', uniform)
    save_json(args.out / 'checks.json', checks)
    print(f'Completed: {args.out}', flush=True)


if __name__ == '__main__':
    main()
