import sys

from os.path import abspath, dirname

import numpy as np

sys.path.insert(0, '.')
from msgwam import config

from .hyperparameters import load
from .evaluation import *
from .learning import *

if __name__ == '__main__':
    config_path, *tasks = sys.argv[1:]
    config.load(config_path)

    cwd = dirname(abspath(__file__))
    hp_dir = cwd + '/../../hyperparameters'
    load(f'{hp_dir}/{config.name}.toml')

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')

        with np.errstate(invalid='raise'):
            globals()[func_name](*args)
