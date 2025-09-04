import numpy as np
import xarray as xr

from .. import config

def get_spectrum() -> xr.Dataset:
    """
    Return the properties of the spectrum specified by the loaded configuration
    file. Functions in this module (excluding utilities) should return a dataset
    with coordinates time (optional), phase speed, direction of propagation, and
    any other dimensions, and variables for dk, dl, intrinsic frequency, and
    flux. The dataset will be passed to `_postprocess`, and the remaining
    variables will be added at launch time by the `Source`.

    Returns
    -------
    xr.Dataset
        Dataset of wave properties.

    """

    n = config.n_source // 4
    edges, flux = _get_edges_and_flux(n)
    cp = (edges[:-1] + edges[1:]) / 2
    phi = np.pi * np.arange(4) / 2

    ones = np.ones(n)
    data = {'phi' : phi, 'cp' : cp}
    data['dk'] = data['dl'] = ('cp', ones)
    data['dc'] = ('cp', np.diff(edges))

    omega_hat = 2 * np.pi / config.T_hat_source
    data['omega_hat'] = ('cp', ones * omega_hat)
    data['flux'] = ('cp', flux)

    ds = xr.Dataset(data).stack(channel=['cp', 'phi'])
    return ds[['dk', 'dl', 'dc', 'omega_hat', 'flux']]

def _get_edges_and_flux(n: int) -> np.ndarray:
    """
    Calculate the edges of the source ray volumes in phase speed. Will be either
    equally spaced in phase speed, or spaced so as to carry equal flux.

    Parameters
    ----------
    n
        How many source ray volumes there will be.

    Returns
    -------
    np.ndarray
        `n + 1` ray volume boundaries, from 0 to `config.c_max`.

    """

    grid = np.linspace(0, config.c_max, 1000001)
    func_name = '_' + config.spectrum_type
    flux_func = globals()[func_name]
    flux_fine = flux_func(grid)

    if config.equal_flux:
        totals = np.cumsum(flux_fine)
        targets = np.arange(n + 1) * totals[-1] / n
        edges = grid[np.argmin(abs(totals[:, None] - targets), axis=0)]
        
        return edges, np.ones(n) * totals[-1] / n

    flux = np.zeros(n)
    edges = np.linspace(0, config.c_max, n + 1)
    idx = np.argmax(grid[:, None] <= edges, axis=1) - 1
    np.add.at(flux, np.maximum(0, idx), flux_fine)

    return edges, flux

def _mima(cp: np.ndarray) -> xr.Dataset:
    """
    Constant-in-time spectrum designed to mirror the source used in the MiMA
    test runs. Behavior varies depending on latitude.
    """

    arg = abs(config.latitude) - (config.lat_tropics - config.source_dlat)
    arg = min(1, max(0, arg / (2 * config.source_dlat)))

    flux_bc = config.flux_bc_tr * (1 - arg) + config.flux_bc_ex * arg
    cp_width = config.cp_width_tr * (1 - arg) + config.cp_width_ex * arg
    flux = np.exp(-0.5 * (cp / cp_width) ** 2)

    return flux_bc * flux / flux.sum() / 2
