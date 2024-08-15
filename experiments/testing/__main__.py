import os
import sys

import torch
torch.set_default_dtype(torch.float64)

n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f'Pytorch will use {n_cpus} CPUs')
torch.set_num_threads(n_cpus)

sys.path.insert(0, '.')
from msgwam import config

from analysis import plot_fluxes, plot_lifetimes, plot_scores
from scenarios import save_mean_state
from strategies import integrate

if __name__ == '__main__':
    config_path, *tasks = sys.argv[1:]
    config.load(config_path)

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')
        globals()[func_name](*args)
