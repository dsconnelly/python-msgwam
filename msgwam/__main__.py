import argparse

from . import config
from .integration import SBDF2Integrator
from .plotting import plot_integration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')
    args = parser.parse_args()

    config.load(args.config_path)
    ds = SBDF2Integrator().integrate()
    ds.to_netcdf(f'data/{config.name}/integration.nc')