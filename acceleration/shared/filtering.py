import cftime
import xarray as xr

from msgwam.constants import EPOCH

from scipy.ndimage import gaussian_filter1d as _filter

def gaussian_filter(da: xr.DataArray, **kwargs: float) -> xr.DataArray:
    """
    Apply a Gaussian filter to a `DataArray`.

    Parameters
    ----------
    da
        Array of data to filter.
    kwargs
        Keys should correspond to coordinates of `da`, and values should be the
        desired widths of the filter in that direction.

    Returns
    -------
    xr.DataArray
        Filtered array.

    """

    for name, width in kwargs.items():
        if name in ['seconds', 'minutes', 'hours', 'days']:
            coord = cftime.date2num(da['time'], f'{name} since {EPOCH}')
            i = list(da.coords).index('time')

        else:
            coord = da[name]
            i = list(da.coords).index(name)

        sigma = max(1, int(width / abs(coord[1] - coord[0]) / 4))
        filtered = _filter(da.values, sigma, axis=i)
        da = xr.DataArray(filtered, da.coords)

    return da
    
