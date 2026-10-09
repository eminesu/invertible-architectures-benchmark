import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
os.environ.setdefault('TQDM_DISABLE', '1')

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from models.npse import NPSE, build, count_parameters, width_for_budget  # noqa: E402

RUNS = Path(__file__).parent.parent / 'experiments' / 'runs'


def test_width_for_budget_fits_3M():
    """The reported runs: width 409; 2,994,309 (kinematics) and 2,993,900 (ballistics) parameters."""
    for y_dims in (2, 1):
        w, n = width_for_budget(4, y_dims, 3_000_000)
        assert n <= 3_000_000 < count_parameters(build(4, y_dims, w + 1))
    assert width_for_budget(4, 2, 3_000_000) == (409, 2_994_309)
    assert width_for_budget(4, 1, 3_000_000) == (409, 2_993_900)


@pytest.mark.parametrize('method, steps, expected', [('euler', 7, 7), ('euler_maruyama', 5, 5), ('rk45', 2, 12)])
def test_nfe_count(method, steps, expected):
    model = NPSE(build(4, 2, 16), method, steps)
    assert model.nfe(np.zeros((3, 2))) == expected


def test_sample_interface_and_chunking():
    keras.utils.set_random_seed(0)
    model = NPSE(build(4, 1, 16), 'euler', 4, chunk=3)
    x = model.sample(np.random.RandomState(0).randn(7, 1), 5)
    assert x.shape == (7, 5, 4) and np.isfinite(x).all()


@pytest.mark.parametrize('run, y_dims, n_params', [('kin_bf_npse_s0', 2, 2_994_309), ('bal_bf_npse_s0', 1, 2_993_900)])
def test_saved_run_loads_and_conditions_on_y(run, y_dims, n_params):
    """A trained run (gitignored, not committed) loads, and its samples move with y*."""
    path = RUNS / run / 'approximator.keras'
    if not path.exists():
        pytest.skip(f'{path} not present')
    model = NPSE.load(str(path), method='euler', steps=8)
    assert model.n_params == n_params
    keras.utils.set_random_seed(0)
    y = np.array([[1.0, 1.0], [1.5, -1.0]] if y_dims == 2 else [[3.0], [12.0]], dtype='float32')
    x = model.sample(y, 200)
    assert x.shape == (2, 200, 4)
    assert np.abs(x[0].mean(0) - x[1].mean(0)).max() > 0.2
