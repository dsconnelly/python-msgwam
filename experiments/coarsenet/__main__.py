import sys

import torch
torch.set_default_dtype(torch.float64)

from msgwam import config

from preprocessing import save_training_data

if __name__ == '__main__':
    config_path, *tasks = sys.argv[1:]
    config.load(config_path)

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')
        globals()[func_name](*args)
