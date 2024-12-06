import sys

from os.path import abspath, dirname

sys.path.insert(0, '.')
from msgwam import config

from .evaluation import *
from .learning import *

if __name__ == '__main__':
    config_path, *tasks = sys.argv[1:]
    config.load(config_path)

    hp_dir = dirname(abspath(__file__)) + '/../../hyperparameters'
    hp.load(f'{hp_dir}/{config.name}.toml')

    for task in tasks:
        func_name, *args = task.split(':')
        func_name = func_name.replace('-', '_')
        globals()[func_name](*args)
