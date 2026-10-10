"""Standard (plain) autoencoder baseline, Kruse et al. (2021), arXiv:2101.10763, Sec. 2.

The inverse problem is solved by representing x as (y, z): y the observation,
z a latent carrying the information f throws away. Two ordinary MLPs, no
architectural invertibility:

    encoder  E: x -> (y_hat, z)      dim(y_hat) = dim(y), dim(z) = dim(x) - dim(y)
    decoder  D: (y, z) -> x

The code has the same total size as x, so nothing is bottlenecked. At zero
reconstruction loss D is exactly the inverse of E ("soft" invertibility: reached
by training, not guaranteed by the architecture).

Losses (weights a, b are not given in the paper):

    L = L_recon + a * L_y + b * L_z
    L_recon = ||x - D(E(x))||^2
    L_y     = ||y_hat - y||^2
    L_z     = MMD([y_hat, z], [y, eps]),  eps ~ N(0, I)       (Eq. 3)

L_z uses the shared kernel from metrics/mmd.py (the one Err_post uses). Taking
it on the joint (y_hat, z) rather than on z alone makes z independent of y as
well as Gaussian, which is what makes test-time sampling z ~ N(0, I) valid. y_hat
is detached inside L_z, as in the INN code of Ardizzone et al. (2019), so the
MMD only shapes z and L_y alone is responsible for y_hat.

Test time uses the decoder only: x = D(y*, z), z ~ N(0, I).

All losses are computed in standardized x / y units (training-set mean / std),
like models/mdn.py. `sample` takes and returns raw units.
"""

import numpy as np
import torch
import torch.nn as nn

from metrics.mmd import mmd_torch

ACTIVATIONS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}


def mlp(d_in, d_out, hidden, n_layers, activation):
    """n_layers hidden Linear+activation layers, then a linear output layer."""
    act = ACTIVATIONS[activation]
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def mlp_params(d_in, d_out, hidden, n_layers):
    return (d_in * hidden + hidden) + (n_layers - 1) * (hidden * hidden + hidden) + hidden * d_out + d_out


def width_for_budget(x_dims, n_layers, target):
    """Largest hidden width such that encoder + decoder stay at or under `target`.
    Both nets map a dim(x)-vector to a dim(x)-vector, so they have equal size."""
    h = 1
    while 2 * mlp_params(x_dims, x_dims, h + 1, n_layers) <= target:
        h += 1
    return h


class Autoencoder(nn.Module):

    def __init__(self, x_dims, y_dims, hidden, n_layers=4, activation='relu', latent_mmd='joint',
                 z_dims=None, recon_condition='predicted', mmd_scale=1.0):
        super().__init__()
        assert latent_mmd in ('joint', 'marginal')
        self.x_dims, self.y_dims = x_dims, y_dims
        self.z_dims = x_dims - y_dims if z_dims is None else z_dims
        if self.z_dims < 1 or recon_condition not in ('predicted', 'true') or mmd_scale <= 0:
            raise ValueError('Invalid latent dimension, reconstruction condition or MMD scale')
        self.recon_condition, self.mmd_scale = recon_condition, mmd_scale
        self.latent_mmd = latent_mmd
        self.encoder = mlp(x_dims, y_dims + self.z_dims, hidden, n_layers, activation)
        self.decoder = mlp(y_dims + self.z_dims, x_dims, hidden, n_layers, activation)

        self.register_buffer('x_mean', torch.zeros(x_dims))
        self.register_buffer('x_std', torch.ones(x_dims))
        self.register_buffer('y_mean', torch.zeros(y_dims))
        self.register_buffer('y_std', torch.ones(y_dims))

    def set_normalization(self, x, y):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0).clamp_min(1e-6))

    # All of encode / decode / losses work in standardized units.
    def encode(self, x_n):
        c = self.encoder(x_n)
        return c[:, :self.y_dims], c[:, self.y_dims:]

    def decode(self, y_n, z):
        return self.decoder(torch.cat([y_n, z], dim=1))

    def losses(self, x, y):
        """Raw-unit batch (x, y) -> dict of the three scalar loss terms."""
        x_n = (x - self.x_mean) / self.x_std
        y_n = (y - self.y_mean) / self.y_std
        y_hat, z = self.encode(x_n)
        # TODO(berker): decode from y_hat (literal D(E(x)), as here) or from the
        # true y_n? Test time feeds the true y*; the two coincide once L_y -> 0.
        x_rec = self.decode(y_hat if self.recon_condition == 'predicted' else y_n, z)
        eps = torch.randn_like(z)
        if self.latent_mmd == 'joint':
            l_z = mmd_torch(torch.cat([y_hat.detach(), z], 1) / self.mmd_scale,
                            torch.cat([y_n, eps], 1) / self.mmd_scale)
        else:
            l_z = mmd_torch(z / self.mmd_scale, eps / self.mmd_scale)
        return {
            'recon': ((x_rec - x_n) ** 2).sum(1).mean(),
            'y': ((y_hat - y_n) ** 2).sum(1).mean(),
            'z': l_z,
        }

    @torch.no_grad()
    def sample(self, y_star, n, generator=None):
        """Posterior samples for each y*: z ~ N(0, I)^n, x = D(y*, z).

        y_star: (M, dim_y) raw units, numpy or tensor. Returns (M, n, dim_x) raw
        units, a tensor on the model's device. This is the `sample(y_star, n)`
        interface metrics.evaluate expects."""
        dev = self.x_mean.device
        y = torch.as_tensor(np.asarray(y_star) if not torch.is_tensor(y_star) else y_star,
                            dtype=self.x_mean.dtype, device=dev)
        M = y.shape[0]
        y_n = ((y - self.y_mean) / self.y_std).repeat_interleave(n, dim=0)
        z = torch.randn(M * n, self.z_dims, device=dev, dtype=y.dtype, generator=generator)
        # Bound decoder activations for the shared 1000 x 1000 evaluation.
        x_n = torch.cat([self.decode(y_n[s:s+8192], z[s:s+8192])
                         for s in range(0, M*n, 8192)], dim=0)
        return (x_n * self.x_std + self.x_mean).view(M, n, self.x_dims)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
