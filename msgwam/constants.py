import numpy as np

EPOCH = '0001-01-01'
ROT_EARTH = 2 * np.pi / 86400

PROP_NAMES = [
    'r', 'dr',
    'k', 'l', 'm',
    'dk', 'dl', 'dm',
    'dens', 'age', 'meta',
    'attrition'
]