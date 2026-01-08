import sys

from . import config
from .integration import integrate
from .plotting import (
    init_plotting,
    plot_integration
)

if __name__ == '__main__':
    config.load(sys.argv[1])
    init_plotting()

    ds = integrate()
    ds.to_netcdf(f'data/{config.name}/integration.nc')
    plot_integration(ds, f'plots/{config.name}/integration.png')
