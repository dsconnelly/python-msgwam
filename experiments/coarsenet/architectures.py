import sys

import torch, torch.nn as nn

sys.path.insert(0, '.')
from msgwam import config
from msgwam.dispersion import cg_r
from msgwam.rays import RayCollection

PROPS = ['dr', 'm', 'dm']
LOOKUP = RayCollection.props[:-2]
INDICES = RayCollection.indices

class CoarseNet(nn.Module):
    """
    CoarseNet currently takes in a zonal wind profile and `config.n_source`
    values for each ray volume property, and predicts the properties of one ray
    volume that is intended to replace it. At present, the neural network only
    operates on the zonal wind and the (dr, m, dm) values and returns those,
    with the remaining properties taken from the input or calculated to
    conserve momentum flux.
    """

    def __init__(self) -> None:
        """Initialize a CoarseNet."""

        super().__init__()

        n_wind = config.n_grid - 1
        n_inputs = n_wind + config.n_source * len(PROPS)
        n_outputs = len(PROPS)

        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_outputs)
        )

        self.to(torch.double)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the CoarseNet, including unpacking the ray volume data, applying
        the neural network, and repacking the calculated wave properties as
        a spectrum tensor.

        Parameters
        ----------
        X
            Tensor containing zonal wind profiles and ray volume properties,
            structured as in the output of `preprocess._generate_inputs`.
        
        Returns
        -------
        Z
            Tensor containing the properties of the replacement ray volumes.

        """

        n_wind = config.n_grid - 1
        u, data = X[:, :n_wind], X[:, n_wind:]
        data = data.reshape(X.shape[0], -1, config.n_source)
        
        jdx = [INDICES[prop] for prop in PROPS]
        to_use = data[:, jdx].reshape(-1, len(PROPS) * config.n_source)
        output = torch.exp(self.layers(torch.hstack((u, to_use))))

        j = PROPS.index('m')
        output[:, j] = -1 * output[:, j]

        Y = torch.zeros(len(LOOKUP), output.shape[0])
        for i, prop in enumerate(LOOKUP[:-1]):
            if prop in PROPS:
                Y[i] = output[:, PROPS.index(prop)]

            else:
                Y[i] = data[:, i, 0]

        def _get(prop: str) -> torch.Tensor:
            return data[:, LOOKUP.index(prop)]
        
        cg = cg_r(_get('k'), _get('l'), _get('m'))
        volume = _get('dk') * _get('dl') * _get('dm')
        flux = _get('k') * _get('dens') * volume * cg

        budget = torch.sum(torch.nan_to_num(torch.abs(flux)), dim=1)

