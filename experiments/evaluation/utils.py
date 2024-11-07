import numpy as np

from msgwam import config
from msgwam.means import PrescribedWind
from msgwam.propagators import TransientPropagator

def get_min_dr(round_to: int=25, **kwargs) -> float:
    """
    Calculate the minimum dr value that can be resolved by the bottom boundary
    condition, given some configuration overrides.

    Parameters
    ----------
    kwargs
        Settings to pass to `config.override`.

    Returns
    -------
    float
        Minimum resolvable dr.

    """

    with config.override(**kwargs):
        mean = PrescribedWind()
        cg = TransientPropagator(mean)._get_cg_r(mean)
        distance = np.nanmax(cg * config.dt)

    return np.ceil(distance / round_to) * round_to
