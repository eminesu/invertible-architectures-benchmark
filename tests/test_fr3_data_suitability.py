"""Audit controls must detect corrupted pairs and incorrect normalization."""

import json

import numpy as np
import pytest
import torch

from experiments.analyze_fr3_data_suitability import fixed_json, normalization, overlap, resimulation
from models.fif import FIF
from robot_ik import kinematics as K


def test_resimulation_rejects_mispaired_pose():
    x = K.sample_uniform(12, np.random.RandomState(5)).astype(np.float32)
    y = K.encode(*K.fk(x), '6d').numpy().astype(np.float32)
    assert resimulation(x, y, 'fr3_6d')['rows_over_tolerance'] == 0
    y[3, 0] += 0.01
    assert resimulation(x, y, 'fr3_6d')['rows_over_tolerance'] == 1


def test_training_only_normalization_control_detects_wrong_buffer():
    rng = np.random.RandomState(4)
    x = rng.randn(20, 7).astype(np.float32)
    y = rng.randn(20, 9).astype(np.float32)
    model = FIF(x_dims=7, y_dims=9, z_dims=1, hidden=8, layers=1, kde_points=8)
    model.fit(torch.from_numpy(x), torch.from_numpy(y))
    state = model.state_dict()
    assert normalization(x, y, None, 'FIF', state)['all_close']
    state['x_mean'][0] += 0.01
    assert not normalization(x, y, None, 'FIF', state)['all_close']


def test_overlap_checks_whole_rows():
    a = np.array([[1, 2], [2, 1], [1, 2]], dtype=np.float32)
    assert overlap(a, np.array([[2, 2]], dtype=np.float32)) == 0
    assert overlap(a, np.array([[1, 2]], dtype=np.float32)) == 1


def test_json_retains_tiny_error_without_exponents():
    result = fixed_json({'error': 0.000000000123, 'regular': 1.5})
    assert 'e-' not in result
    assert json.loads(result)['error'] == 0.000000000123
    with pytest.raises(ValueError):
        fixed_json({'error': float('nan')})
