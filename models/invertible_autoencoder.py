"""Teng-style tied-transpose InvAuto, benchmark Eq. 7.

Decoder reverses biases and nonlinearities and uses encoder weight transposes.
Transpose is an approximate inverse learned through reconstruction, not a
guaranteed matrix inverse. Rectangular boundary layers preserve the 4-D code;
hidden square layers permit orthogonality diagnostics. No hard projection is
used in the baseline; optional orthogonality regularization is an ablation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.autoencoder import Autoencoder


class InvertibleAutoencoder(Autoencoder):
    def __init__(self, x_dims, y_dims, hidden, n_layers=4, latent_mmd='joint',
                 recon_condition='predicted', mmd_scale=1.0, slope=2.0):
        super().__init__(x_dims, y_dims, hidden, n_layers, 'relu', latent_mmd,
                         recon_condition=recon_condition, mmd_scale=mmd_scale)
        if slope <= 0:
            raise ValueError('slope must be positive')
        self.slope = slope
        dims = [x_dims] + [hidden] * n_layers + [x_dims]
        self.encoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        del self.decoder  # decoder uses exactly the encoder parameters

    def activation(self, x, inverse=False):
        # Modified LeakyReLU, Teng et al. Eqs. 5-6, alpha=2 by default.
        a = 1/self.slope if inverse else self.slope
        return torch.where(x >= 0, x/a, x*a)

    def encode(self, x_n):
        for i, layer in enumerate(self.encoder):
            x_n = layer(x_n)
            if i < len(self.encoder)-1:
                x_n = self.activation(x_n)
        return x_n[:, :self.y_dims], x_n[:, self.y_dims:]

    def decode(self, y_n, z):
        h = torch.cat([y_n, z], 1)
        for i in range(len(self.encoder)-1, -1, -1):
            if i < len(self.encoder)-1:
                h = self.activation(h, inverse=True)
            layer = self.encoder[i]
            h = F.linear(h-layer.bias, layer.weight.T)
        return h

    def orthogonality_terms(self):
        terms = []
        for layer in self.encoder:
            w = layer.weight
            gram = w.T @ w if w.shape[0] >= w.shape[1] else w @ w.T
            identity = torch.eye(len(gram), device=w.device, dtype=w.dtype)
            terms.append((gram-identity).square().sum()/len(gram))
        return torch.stack(terms)
