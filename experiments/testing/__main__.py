import sys

sys.path.insert(0, '.')
from msgwam import config

from analysis import plot_fluxes, plot_scores
from scenarios import save_mean_state
from strategies import integrate

if __name__ == '__main__':
    config_path, *tasks = sys.argv[1:]
    config.load(config_path)

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')
        globals()[func_name](*args)
