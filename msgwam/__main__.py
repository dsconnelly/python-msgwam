import sys

from . import config
from .integration import integrate
from .plotting import (
    plot_boundary,
    plot_integration,
    plot_ray_count,
    plot_source
)

if __name__ == '__main__':
    config.load(sys.argv[1])

    ds = integrate()
    ds.to_netcdf(f'data/{config.name}/integration.nc')

    plot_boundary(ds, f'plots/{config.name}/boundary.png')
    plot_integration(ds, f'plots/{config.name}/integration.png')
    plot_source(f'plots/{config.name}/source.png')

    if config.propagator_type == 'transient':
        plot_ray_count(ds, f'plots/{config.name}/ray-count.png')
