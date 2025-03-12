import torch, torch.nn as nn

from msgwam import config

from ...hyperparameters import architectures as hp
from .base import BaseNet
from .utils import get_block

class Encoder(BaseNet):
    """
    An `Encoder` takes the ray volume properties and returns a lower-dimensional
    representation of that state. The goal is for that latent vector to allow
    fluxes and time tendencies to be recovered.
    """

    def _forward(self, R: torch.Tensor):
        """
        The forward function is simply the composition of the convolutional part
        and the fully-connected part.        
        """

        return self._dense(self._conv(R))

    def _get_convolutions(self) -> nn.Sequential:
        """
        Build the convolutional part of the network, which moves detail from the
        sequence dimension to the channel dimension while reducing the overall
        data size through pooling.

        Returns
        -------
        nn.Sequential
            Module containing convolutions, activation functions, pooling, and
            potentially batch normalization.

        """

        n_c = 5
        n_s = config.n_max
        kernel_size = 7
        args = []

        while n_c * n_s > hp.n_latent:
            padding = (kernel_size - 1) // 2
            n_out = 8 if n_c == 5 else 2 * n_c
            pool = min((n_out * n_s) // (2 * hp.n_latent), 4)
            
            layer = nn.Conv1d(n_c, n_out, kernel_size, 2, padding)
            args.extend([layer, nn.ReLU(), nn.AvgPool1d(pool, pool)])

            if hp.batch_norm_pos == 1:
                args.append(nn.BatchNorm1d(n_out))

            n_c, n_s = n_out, n_s // 2 // pool
            kernel_size = max(kernel_size - 2, 3)

        return nn.Sequential(*args, nn.Flatten())

    def _init_layers(self):
        """
        The `Encoder` has two subparts: a convolutional network that reduces the
        sample dimension, and a dense network that parses the features. It is
        assumed that both `config.n_max` and `hp.n_latent` are powers of two.
        """

        self._conv = self._get_convolutions()
        sizes = [hp.n_latent] * hp.n_per_block
        self._dense = get_block(sizes, final=True)

    @property
    def _inputs(self) -> list[str]:
        """The only `Encoder` input is the ray volume information."""
        return ['R']

    @property
    def _output(self) -> str:
        """The `Encoder` outputs in the latent space."""
        return 'Z'
