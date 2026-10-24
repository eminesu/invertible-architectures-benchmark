import numpy as np
import pytest
import torch

from data.toy_data import MODELS, make_dataset
from models.autoencoder import Autoencoder, count_parameters, mlp_params, width_for_budget


@pytest.mark.parametrize('problem, y_dims', [('kinematics', 2), ('ballistics', 1)])
def test_code_split_has_no_bottleneck(problem, y_dims):
    m = MODELS[problem]()
    model = Autoencoder(m.n_parameters, m.n_observations, hidden=32)
    assert model.y_dims == y_dims
    assert model.y_dims + model.z_dims == m.n_parameters == 4
    y_hat, z = model.encode(torch.randn(7, 4))
    assert y_hat.shape == (7, y_dims) and z.shape == (7, 4 - y_dims)
    assert model.decode(y_hat, z).shape == (7, 4)


def test_parameter_budget():
    h = width_for_budget(4, 4, 3_000_000)
    n = count_parameters(Autoencoder(4, 2, h, 4))
    assert n == 2 * mlp_params(4, 4, h, 4)
    assert 2.95e6 < n <= 3e6
    assert count_parameters(Autoencoder(4, 2, h + 1, 4)) > 3e6


def test_losses_finite_and_backprop():
    x, y = (torch.from_numpy(a) for a in make_dataset('kinematics', 256, seed=0))
    model = Autoencoder(4, 2, hidden=32)
    model.set_normalization(x, y)
    terms = model.losses(x, y)
    assert set(terms) == {'recon', 'y', 'z'}
    sum(terms.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_joint_latent_mmd_does_not_train_y_hat():
    """y_hat is detached inside L_z: the MMD gradient only reaches the encoder
    through z, so the y-slot is trained by L_y alone."""
    x, y = (torch.from_numpy(a) for a in make_dataset('kinematics', 256, seed=0))
    model = Autoencoder(4, 2, hidden=32)
    model.set_normalization(x, y)
    model.losses(x, y)['z'].backward()
    last = model.encoder[-1]
    assert torch.all(last.weight.grad[:2] == 0) and torch.all(last.bias.grad[:2] == 0)  # y_hat rows
    assert last.weight.grad[2:].abs().sum() > 0                                          # z rows


def test_sample_shape_and_units():
    x, y = (torch.from_numpy(a) for a in make_dataset('ballistics', 512, seed=0))
    model = Autoencoder(4, 1, hidden=32)
    model.set_normalization(x, y)
    y_star = y[:5].numpy()
    s = model.sample(y_star, 11)
    assert s.shape == (5, 11, 4) and s.dtype == torch.float32
    assert torch.equal(model.sample(torch.from_numpy(y_star), 3, generator=torch.Generator().manual_seed(0)),
                       model.sample(y_star, 3, generator=torch.Generator().manual_seed(0)))


def test_sample_decodes_given_y_with_prior_z():
    """sample() must be x = D(y*, z), z ~ N(0, I), with y* standardized like training."""
    x, y = (torch.from_numpy(a) for a in make_dataset('kinematics', 512, seed=0))
    model = Autoencoder(4, 2, hidden=32)
    model.set_normalization(x, y)
    y_star = y[:3]
    s = model.sample(y_star, 4, generator=torch.Generator().manual_seed(1))
    z = torch.randn(12, 2, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model.decode(((y_star - model.y_mean) / model.y_std).repeat_interleave(4, 0), z)
    assert np.allclose(s.view(12, 4).numpy(), (ref * model.x_std + model.x_mean).numpy(), atol=1e-6)
