from msgwam import config
from msgwam.sources.spectra import _gaussians

from .idealized import get_descending_jets
from .plotting import plot_mean_state

__all__ = ['plot_mean_state', 'save_mean_state', 'save_spectrum']

def save_mean_state(scenario: str) -> None:
    """
    Save a mean state to use as a prescribed wind field.

    Parameters
    ----------
    scenario
        Name of the scenario for which to save the mean wind. There must be a
        function `get_{scenario}` in the global namespace accepting an integer
        seed and returning a `Dataset`.

    """

    with config.override(dt=30):
        path = config.prescribed_wind_file
        scenario = scenario.replace('-', '_')
        globals()[f'get_{scenario}'](1).to_netcdf(path)

def save_spectrum() -> None:
    """Save a source spectrum."""

    with config.override(spectrum_type='gaussians', n_source=1000):
        _gaussians().to_netcdf(config.spectrum_file)