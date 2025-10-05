from msgwam import config

from ... import hyperparameters as hp
from ...scenarios import get_mima_scenario

from .utils import get_site_and_lat

def save_training_context() -> None:
    """Save the mean states for the training integrations."""

    n_site = hp.task_id // 12
    month = (hp.task_id % 12) + 1
    site, _ = get_site_and_lat(n_site)

    with config.override(name=f'mima-{site}'):
        path = f'data/ml-accel/context/{site}-{month}.nc'
        get_mima_scenario(month=month).to_netcdf(path)