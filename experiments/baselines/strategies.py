import sys

import numpy as np

sys.path.insert(0, '.')
from msgwam import config

def reference() -> None:
    pass

def ICON() -> None:
    config.n_ray_max = 2500
    config.purge = True

def do_nothing() -> None:
    config.n_ray_max = 250
    config.purge = True

def coarse_square() -> None:
    n_source = int(config.n_source / np.sqrt(10))
    n_source = n_source + (n_source % 2)

    config.n_ray_max = 250
    config.n_source = n_source
    config.dr_init *= 10
    config.purge = True

def coarse_tall() -> None:
    config.n_ray_max = 250
    config.dr_init *= 10
    config.purge = True

def coarse_wide() -> None:
    n_source = int(config.n_source / 10)
    n_source = n_source + (n_source % 2)

    config.n_ray_max = 250
    config.n_source = n_source
    config.purge = True

def stochastic() -> None:
    config.n_ray_max = 250
    config.epsilon = 0.1
    config.purge = True