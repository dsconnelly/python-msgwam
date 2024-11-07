import sys

from . import config
from .integration import integrate
from .plotting import plot_integration, plot_ray_count, plot_source

if __name__ == '__main__':
    config.load(sys.argv[1])

    ds = integrate()
    ds.to_netcdf(f'data/{config.name}/integration.nc')
    plot_integration(ds, f'plots/{config.name}/integration.png')
    plot_ray_count(ds, f'plots/{config.name}/ray-count.png')
    plot_source(f'plots/{config.name}/source.png')
