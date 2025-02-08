from msgwam import config
from msgwam.sources.spectra import _gaussians

from .icon import get_ICON
from .idealized import get_descending_jets
from .plotting import plot_mean_state, plot_spectrum, plot_windows

__all__ = [
    'plot_mean_state',
    'plot_spectrum',
    'plot_windows',
    'save_mean_state',
    'save_spectrum'
]

def save_mean_state(scenario: str, *args: str) -> None:
    """
    Save a mean state to use as a prescribed wind field.

    Parameters
    ----------
    scenario
        Name of the scenario for which to save the mean wind. There must be a
        function `get_{scenario}` in the global namespace accepting an integer
        seed and returning a `Dataset`.
    args
        Other arguments to pass to the scenario function.

    """

    with config.override(n_grid=401, dt=30):
        path = config.prescribed_wind_file
        scenario = scenario.replace('-', '_')
        globals()[f'get_{scenario}'](*args).to_netcdf(path)

def save_spectrum() -> None:
    """Save a source spectrum."""

    with config.override(spectrum_type='gaussians', n_source=1000):
        _gaussians().to_netcdf(config.spectrum_file)