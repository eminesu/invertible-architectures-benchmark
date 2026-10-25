"""Conditional FIF for the fixed-prior inverse problems (arXiv:2306.01843).

The condition is a *clean* simulator output, held fixed in all Jacobians.
A Gaussian KDE over clean outputs supplies the observation-noise bridge.
This implementation is independent of other flow models in this repository.
"""

import math

import numpy as np
import torch
from torch import nn


class ConditionedNet(nn.Module):
    """A dense network with condition-dependent scale and shift at each layer."""

    def __init__(self, inputs, outputs, context, hidden, layers, activation=nn.SiLU):
        super().__init__()
        self.hidden = nn.ModuleList(
            nn.Linear(inputs if i == 0 else hidden, hidden) for i in range(layers))
        self.condition = nn.ModuleList(nn.Linear(context, 2 * hidden) for _ in range(layers))
        self.output = nn.Linear(hidden, outputs)
        self.activation = activation()

    def forward(self, value, context):
        for layer, conditioning in zip(self.hidden, self.condition):
            scale, shift = conditioning(context).chunk(2, dim=-1)
            value = self.activation(layer(value) * (1 + 0.1 * scale.tanh()) + shift)
        return self.output(value)


class PaperNet(nn.Module):
    """Appendix E.2 ordering, adapted with the same conditioning as the MLP."""

    def __init__(self, inputs, outputs, context, hidden, layers, blocks, encode):
        super().__init__()
        self.encode = encode
        self.mlp = ConditionedNet(inputs, outputs, context, hidden, layers, nn.ReLU)
        latent = outputs if encode else inputs
        self.residuals = nn.ModuleList(
            ConditionedNet(latent, latent, context, hidden, layers, nn.ReLU)
            for _ in range(blocks))

    def forward(self, value, context):
        if self.encode:
            value = self.mlp(value, context)
        for block in self.residuals:
            value = value + block(value, context)
        return value if self.encode else self.mlp(value, context)


def parameter_count(x_dims, y_dims, z_dims, hidden, layers, arch='mlp', blocks=2):
    if arch not in ('mlp', 'paper') or layers < 1 or blocks < 0:
        raise ValueError('Invalid architecture, layer count or residual block count')
    linear = 2 * (x_dims + z_dims) + 2 * layers + 4 * layers * (y_dims + 1)
    count = 2 * (layers - 1) * hidden**2 + linear * hidden + x_dims + z_dims
    if arch == 'paper':
        residual = ((layers - 1) * hidden**2
                    + (2 * z_dims + layers + 2 * layers * (y_dims + 1)) * hidden + z_dims)
        count += 2 * blocks * residual
    return count


def width_for_budget(y_dims, z_dims, budget, layers=None, x_dims=4, arch='mlp', blocks=2):
    layers = (2 if arch == 'paper' else 4) if layers is None else layers

    def count(hidden):
        return parameter_count(x_dims, y_dims, z_dims, hidden, layers, arch, blocks)

    if layers < 1 or budget < count(1):
        raise ValueError('Invalid layer count or parameter budget')
    low, high = 1, 2
    while count(high) <= budget:
        high *= 2
    while high - low > 1:
        mid = (low + high) // 2
        if count(mid) <= budget:
            low = mid
        else:
            high = mid
    return low


class FIF(nn.Module):
    def __init__(self, x_dims=4, y_dims=2, z_dims=2, hidden=256, layers=None,
                 beta=100.0, sigma=0.01, kde_points=1024, discrete_speed=False,
                 arch='mlp', blocks=2):
        super().__init__()
        if arch not in ('mlp', 'paper') or blocks < 0:
            raise ValueError('Expected arch="mlp" or "paper" and nonnegative blocks')
        layers = (2 if arch == 'paper' else 4) if layers is None else layers
        if not 1 <= z_dims < x_dims or min(y_dims, hidden, layers, kde_points) < 1:
            raise ValueError('FIF needs positive dimensions and a genuine bottleneck')
        if not math.isfinite(beta) or beta <= 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError('beta must be positive and sigma nonnegative')
        if discrete_speed and x_dims != 4:
            raise ValueError('Discrete speed is specific to the 4-D ballistics parameters')
        self.x_dims, self.y_dims, self.z_dims = x_dims, y_dims, z_dims
        self.beta, self.sigma, self.discrete_speed = beta, sigma, discrete_speed
        self.arch = arch
        if arch == 'paper':
            self.encoder = PaperNet(x_dims, z_dims, y_dims, hidden, layers, blocks, encode=True)
            self.decoder = PaperNet(z_dims, x_dims, y_dims, hidden, layers, blocks, encode=False)
        else:
            self.encoder = ConditionedNet(x_dims, z_dims, y_dims, hidden, layers)
            self.decoder = ConditionedNet(z_dims, x_dims, y_dims, hidden, layers)
        for name, dims in (('x', x_dims), ('y', y_dims)):
            self.register_buffer(name + '_mean', torch.zeros(dims))
            self.register_buffer(name + '_std', torch.ones(dims))
        self.register_buffer('centres', torch.zeros(kde_points, y_dims))
        self.register_buffer('kde_cov', torch.eye(y_dims))
        self.register_buffer('fitted', torch.tensor(False))

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.x_mean.dtype, device=self.x_mean.device)

    @torch.no_grad()
    def fit(self, x, y, seed=0):
        """Fit scaling and a Scott-bandwidth KDE, using training data only."""
        x, y = self._tensor(x), self._tensor(y)
        if x.ndim != 2 or x.shape[1] != self.x_dims or y.shape != (len(x), self.y_dims):
            raise ValueError('Expected paired arrays (N, x_dims), (N, y_dims)')
        if len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('At least two finite training pairs are required')
        self.x_mean.copy_(x.mean(0))
        variance = x.var(0, unbiased=False)
        if self.discrete_speed:
            variance[3] += 1 / 12  # variance of the independent dequantization noise
        self.x_std.copy_(variance.sqrt().clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0, unbiased=False).clamp_min(1e-6))
        context = (y - self.y_mean) / self.y_std
        count = len(self.centres)
        indices = np.random.default_rng(seed).choice(len(y), count, replace=len(y) < count)
        self.centres.copy_(context[torch.as_tensor(indices, device=context.device)])
        bandwidth = count ** (-1 / (self.y_dims + 4))
        covariance = context.T @ context / (len(context) - 1)
        self.kde_cov.copy_(bandwidth**2 * covariance + 1e-6 * torch.eye(
            self.y_dims, device=context.device, dtype=context.dtype))
        self.fitted.fill_(True)

    @torch.enable_grad()
    def loss_terms(self, x, y):
        """FIF Eq. 7 with the exact trace (small latent space), in scaled units.

        Only the encoder receives the volume-term gradient. The decoder is
        trained by reconstruction. The surrogate's value is NOT a log density.
        No BatchNorm or dropout: differentiating sums gives per-example Jacobians.
        """
        x, y = self._tensor(x), self._tensor(y)
        if x.ndim != 2 or x.shape[1] != self.x_dims or y.shape != (len(x), self.y_dims):
            raise ValueError('Expected paired two-dimensional training arrays')
        if self.discrete_speed:
            x = x.clone()
            x[:, 3] += torch.rand_like(x[:, 3]) - 0.5
        inputs = ((x - self.x_mean) / self.x_std).detach().requires_grad_(True)
        context = ((y - self.y_mean) / self.y_std).detach()
        latent = self.encoder(inputs, context)
        reconstructed = self.decoder(latent, context)
        encoder_jac = torch.stack([torch.autograd.grad(
            latent[:, j].sum(), inputs, create_graph=True, retain_graph=True)[0]
            for j in range(self.z_dims)], dim=1)
        decoder_jac = torch.stack([torch.autograd.grad(
            reconstructed[:, i].sum(), latent, retain_graph=True)[0]
            for i in range(self.x_dims)], dim=1)
        trace = torch.einsum('bji,bij->b', encoder_jac, decoder_jac.detach()).mean()
        latent_energy = 0.5 * (latent.square() + math.log(2 * math.pi)).sum(1).mean()
        reconstruction = (reconstructed - inputs).square().sum(1).mean()
        return {'loss': latent_energy - trace + self.beta * reconstruction,
                'recon': reconstruction, 'latent': latent_energy, 'trace': trace}

    @torch.no_grad()
    def clean_context(self, observations, n, generator=None):
        """Draw t | y from KDE(t) N(y; t, sigma^2 I), in normalized units.

        The KDE is an equal-weight Gaussian mixture with a shared covariance.
        Its posterior component probabilities, means and covariance are analytic.
        """
        y = (observations - self.y_mean) / self.y_std
        if self.sigma == 0:
            return y.repeat_interleave(n, dim=0)
        if not self.fitted.item():
            raise RuntimeError('Call fit() before sampling with observation noise')
        noise = torch.diag((self.sigma / self.y_std).square())
        chol = torch.linalg.cholesky(self.kde_cov + noise)
        delta = y[:, None, :] - self.centres[None, :, :]
        precision_delta = torch.cholesky_solve(delta.transpose(1, 2), chol).transpose(1, 2)
        weights = (-0.5 * (delta * precision_delta).sum(-1)).softmax(-1)
        component = torch.multinomial(weights, n, replacement=True, generator=generator)
        centres = self.centres[component]
        gain = torch.cholesky_solve(self.kde_cov, chol).T
        mean = centres + (y[:, None, :] - centres) @ gain.T
        identity = torch.eye(self.y_dims, device=y.device, dtype=y.dtype)
        residual = identity - gain
        # Joseph form avoids cancellation when observation noise is very small.
        covariance = residual @ self.kde_cov @ residual.T + gain @ noise @ gain.T
        factor = torch.linalg.cholesky((covariance + covariance.T) / 2)
        epsilon = torch.randn(mean.shape, device=y.device, dtype=y.dtype, generator=generator)
        return (mean + epsilon @ factor.T).reshape(-1, self.y_dims)

    @torch.no_grad()
    def sample(self, y_star, n, generator=None, chunk_size=8192):
        """Raw observations (M, y_dims) -> raw parameter samples (M, n, x_dims)."""
        y = self._tensor(y_star)
        if y.ndim != 2 or y.shape[1] != self.y_dims or not torch.isfinite(y).all():
            raise ValueError('y_star must be a finite (M, y_dims) array')
        if (not isinstance(n, int) or isinstance(n, bool) or n < 1
                or not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1):
            raise ValueError('n and chunk_size must be positive integers')
        if len(y) == 0:
            return y.new_empty((0, n, self.x_dims))
        context = self.clean_context(y, n, generator)
        samples = y.new_empty((len(y) * n, self.x_dims))
        for start in range(0, len(samples), chunk_size):
            stop = min(start + chunk_size, len(samples))
            z = torch.randn(stop - start, self.z_dims, device=y.device,
                            dtype=y.dtype, generator=generator)
            samples[start:stop] = self.decoder(z, context[start:stop]) * self.x_std + self.x_mean
        if self.discrete_speed:
            samples[:, 3] = (samples[:, 3] + 0.5).floor().clamp_min(0)
        return samples.reshape(len(y), n, self.x_dims)
