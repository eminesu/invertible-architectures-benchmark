"""IKFlow, Ames, Morgan & Konidaris (2022), arXiv:2111.08933 (IEEE RA-L 7(2)).

A conditional normalizing flow over the joint angles, conditioned on the end-effector pose,
trained by maximum likelihood (paper Eqs. 5, 6). Built on the same GLOW-style coupling block as
models.inn.CINN (models.inn.Flow; the paper uses FrEIA's GLOWCouplingBlock). What distinguishes
IKFlow from the cINN of models.inn, and what this module adds:

* Width. The flow acts on `width` >= dim(x) channels; the joint vector is zero-padded up to the
  width (paper Table I: Panda, 7 joints, width 9). The latent has dimension `width`, and the
  padding is dropped again after sampling.
* SoftFlow (paper Sec. IV-C, Kim et al. 2020). Solutions often lie on a lower-dimensional set
  than the latent, which makes the likelihood singular. Each training step draws a level
  c ~ U(0, 1), adds N(0, (c * softflow_scale)^2) noise to the padded joint vector (softflow_scale
  1e-3 in raw joint units, i.e. radians) and feeds c to every coupling subnet next to the pose.
  At test time c = 0, which removes the noise.
* Latent sub-sampling (paper Sec. IV-D). Samples use z ~ N(0, latent_scale^2 I); the paper uses
  0.25 for accuracy at some cost in diversity (its Fig. 7). latent_scale is an attribute, so it
  can be changed after training.

Checked against the paper: with conditioning size dim(y) + 1 (7-D pose + c in the paper), 3 hidden
layers of width 1024 and the repo's block, the closed-form parameter count reproduces all rows of
Table I (e.g. 12 blocks, width 9: 5.093e7); see tests/test_ikflow.py and docs/ikflow.md.

All losses and the flow work in standardized x / y units (training-set mean / std), like the other
flow models; the SoftFlow noise is converted so that it is `softflow_scale` in raw joint units.
Not specified by the paper, chosen here: soft-clamp 2.0 (FrEIA style, as in models.inn), identity
initialisation, the exact noise parameterisation (std = c * scale, as in SoftFlow).
"""

import torch

from models.inn import Flow, _FlowModel, count_parameters, flow_params, width_for_budget  # noqa: F401


class IKFlow(_FlowModel):
    """x -> z | y, with padding and SoftFlow noise; see the module docstring."""

    loss_terms = ('nll',)

    def __init__(self, x_dims, y_dims, width=None, n_blocks=12, hidden=1024, n_layers=3,
                 activation='leaky_relu', clamp=2.0, softflow_scale=1e-3, latent_scale=0.25, perm_seed=0):
        super().__init__(x_dims, y_dims)
        self.width = x_dims + 2 if width is None else width   # the paper's Panda: 7 joints, width 9
        if self.width < x_dims:
            raise ValueError(f'IKFlow width {self.width} must be at least dim(x) = {x_dims}')
        self.pad = self.width - x_dims
        self.z_dims = self.width
        self.softflow_scale, self.latent_scale = softflow_scale, latent_scale
        # condition = standardized pose (y) and the SoftFlow noise level c
        self.flow = Flow(self.width, y_dims + 1, n_blocks, hidden, n_layers, activation, clamp, perm_seed)

    def _pad(self, x_n):
        if not self.pad:
            return x_n
        return torch.cat([x_n, x_n.new_zeros(len(x_n), self.pad)], dim=1)

    def _noise_scale(self):
        """Std of the SoftFlow noise at level c = 1, for the standardized padded vector: softflow_scale
        in raw joint units on the joints, softflow_scale on the (unit-scaled) padding."""
        return torch.cat([self.softflow_scale / self.x_std, self.x_std.new_full((self.pad,), self.softflow_scale)])

    def forward(self, x, y, c=None):
        """Raw (x, y) and noise level c (B, 1; default 0) -> (z, log_det) of the padded, standardized
        map, without noise (the noise is added in `losses`)."""
        x_n, y_n = self._norm(x, y)
        c = x_n.new_zeros(len(x_n), 1) if c is None else c
        return self.flow(self._pad(x_n), torch.cat([y_n, c], dim=1))

    def losses(self, x, y):
        """SoftFlow likelihood, Eq. (6): 1/2 |z|^2 - log|det J| with c ~ U(0, 1) and noise c * scale,
        redrawn on every call (also in validation, so train and validation losses are comparable)."""
        x_n, y_n = self._norm(x, y)
        u = self._pad(x_n)
        c = torch.rand(len(u), 1, device=u.device, dtype=u.dtype)
        u = u + torch.randn_like(u) * c * self._noise_scale()
        z, log_det = self.flow(u, torch.cat([y_n, c], dim=1))
        return {'nll': (0.5 * (z ** 2).sum(1) - log_det).mean()}

    def _decode(self, y_n, z):
        """Standardized y and z -> standardized x: noise level 0, z scaled by latent_scale, padding dropped."""
        c0 = y_n.new_zeros(len(y_n), 1)
        u = self.flow.inverse(z * self.latent_scale, torch.cat([y_n, c0], dim=1))
        return u[:, :self.x_dims]
