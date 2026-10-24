"""Model factory and parameter-budget matching for the three AE baselines."""
from models.autoencoder import Autoencoder, mlp_params
from models.cvae import ConditionalVAE
from models.invertible_autoencoder import InvertibleAutoencoder

FAMILIES = ('standard', 'cvae', 'invertible')


def parameter_count(family, x, y, z, hidden, layers):
    code = y+z
    if family == 'standard':
        return mlp_params(x, code, hidden, layers)+mlp_params(code, x, hidden, layers)
    if family == 'cvae':
        return mlp_params(x+y, 2*z, hidden, layers)+mlp_params(code, x, hidden, layers)
    return mlp_params(x, x, hidden, layers)


def build_model(cfg):
    y = 2 if cfg['problem'] == 'kinematics' else 1
    family, layers = cfg['family'], cfg.get('layers', 4)
    z = cfg.get('z_dims')
    z = 4-y if z is None else z
    if family == 'invertible' and z != 4-y:
        raise ValueError('InvAuto requires a 4-D code; larger z is a standard/CVAE ablation')
    if layers < 1 or z < 1:
        raise ValueError('layers and z_dims must be positive')
    hidden = cfg.get('hidden_resolved') or cfg.get('hidden')
    if hidden is None:
        budget = cfg.get('param_budget', 3_000_000)
        low, high = 1, 2
        if parameter_count(family, 4, y, z, low, layers) > budget:
            raise ValueError('Parameter budget too small')
        while parameter_count(family, 4, y, z, high, layers) <= budget:
            high *= 2
        while high-low > 1:
            mid = (low+high)//2
            if parameter_count(family, 4, y, z, mid, layers) <= budget:
                low = mid
            else:
                high = mid
        hidden = low
    if hidden < 1:
        raise ValueError('hidden must be positive')
    common = dict(x_dims=4, y_dims=y, hidden=hidden, n_layers=layers)
    if family == 'cvae':
        return ConditionalVAE(**common, z_dims=z, activation=cfg.get('activation', 'relu'))
    settings = dict(latent_mmd=cfg.get('latent_mmd', 'joint'),
                    recon_condition=cfg.get('recon_condition', 'predicted'),
                    mmd_scale=cfg.get('mmd_scale', 1.0))
    if family == 'invertible':
        return InvertibleAutoencoder(**common, **settings, slope=cfg.get('slope', 2.0))
    if family == 'standard':
        return Autoencoder(**common, **settings, z_dims=z, activation=cfg.get('activation', 'relu'))
    raise ValueError(f'Unknown model family: {family}')
