import argparse

from . import config
from .integrate import SBDF2Integrator
from .plotting import plot_integration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='path to config file')

    args = parser.parse_args()
    config.load_config(args.config_path)
    
    ds = SBDF2Integrator().integrate().to_dataset()
    plot_integration(ds, f'plots/{config.name}.png')
    ds.to_netcdf(f'data/{config.name}.nc')
