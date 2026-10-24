"""Conditional VAE: q(z|x,y), p(x|z,y), standard-normal latent prior.

Squared reconstruction plus beta * analytic Gaussian KL (benchmark Eq. 8).
The decoder is deterministic; posterior diversity comes from the latent draw.
"""
import torch
from models.autoencoder import Autoencoder, mlp


class ConditionalVAE(Autoencoder):
    def __init__(self, x_dims, y_dims, hidden, n_layers=4, activation='relu', z_dims=None):
        super().__init__(x_dims, y_dims, hidden, n_layers, activation, z_dims=z_dims)
        self.encoder = mlp(x_dims + y_dims, 2 * self.z_dims, hidden, n_layers, activation)

    def posterior(self, x_n, y_n):
        mu, logvar = self.encoder(torch.cat([x_n, y_n], 1)).chunk(2, dim=1)
        # Numerical guard; record the bound as part of this implementation.
        return mu, logvar.clamp(-12, 12)

    def losses(self, x, y):
        x_n, y_n = (x-self.x_mean)/self.x_std, (y-self.y_mean)/self.y_std
        mu, logvar = self.posterior(x_n, y_n)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        rec = self.decode(y_n, z)
        return {'recon': (rec-x_n).square().sum(1).mean(),
                'kl': 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1).mean()}
