from hashlib import sha1

from msgwam import config
from msgwam.sources.spectra import _gaussians

from ... import hyperparameters as hp
from ...scenarios.idealized import get_descending_jets
from .overrides import get_overrides

def save_training_context() -> None:
    """
    Save longer versions of the mean wind and source spectrum files, generated
    with the same processes as in the test scenarios but with different seeds.
    """

    kwargs = get_overrides()
    wind_seed = _make_seed(config.name, 'wind', hp.task_id)
    kwargs['seed'] = _make_seed(config.name, 'spectrum', hp.task_id)
    kwargs['n_source'] = 1000
    
    with config.override(**kwargs):
        ds = get_descending_jets(seed=wind_seed)
        ds.to_netcdf(config.prescribed_wind_file)
        _gaussians().to_netcdf(config.spectrum_file)

def _make_seed(*args) -> int:
    """
    Make a random seed by hashing a string composed of the given arguments.

    Parameters
    ----------
    args
        List of objects to use in generating the string. Must support conversion
        to string in some way or another.

    Returns
    -------
    int
        Appropriately bounded seed.

    """

    to_hash = ''.join(map(str, args))
    hashed = sha1(to_hash.encode()).digest()

    return int.from_bytes(hashed, 'big') % 2 ** 32
