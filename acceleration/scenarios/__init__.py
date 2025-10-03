from msgwam import config

from .mima import get_mima_scenario
from .plotting import plot_mean_state, plot_mean_scales

__all__ = [
    'get_mima_scenario',
    'plot_mean_state',
    'plot_mean_scales',
    'save_mean_state',
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
        path = config.prescribed_mean_file
        scenario = scenario.replace('-', '_')
        globals()[f'get_{scenario}'](*args).to_netcdf(path)
