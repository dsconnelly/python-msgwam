import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, '.')
from msgwam.sources import _c_from

def plot_power(name: str) -> None:
    with xr.open_datset(f'data/baselines/{name}.nc') as ds:
        c = _c_from(ds['k'], ds['l'], ds['m']).values
        kms = ds['z'].values / 1000