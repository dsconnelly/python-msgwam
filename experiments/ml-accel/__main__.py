import os
import sys

import torch
torch.set_default_dtype(torch.float64)

from msgwam import config

import hyperparameters as params

from inversion import solve_inversion
from training import train_network
from preprocessing import save_indices, save_training_data

if __name__ == '__main__':
    config_path, hparam_path, *tasks = sys.argv[1:]
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    config.load(config_path)
    params.load(hparam_path, task_id)

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')
        globals()[func_name](*args)
