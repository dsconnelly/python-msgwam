import torch, torch.nn as nn

from optuna.trial import Trial

from .utils import iter_pairs, maybe_interp, xavier_init

class UNet(nn.Module):
    def __init__(self, n_bins: int, trial: Trial) -> None:
        """
        Initialize a UNet with a specific number of phase speed bins.

        Parameters
        ----------
        n_bins
            Number of phase speed bins into which the bulk momentum is split.
        trial
            Current trial, used to sample parameters for the UNet.

        """

        super().__init__()
        self._n_bins = n_bins

        self._init_layers(trial)
        self.apply(xavier_init)
        self.to(torch.double)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the UNet architecture, by first downsampling and then upsampling
        with skip connections concatenated on.

        Parameters
        ----------
        X
            Tensor of input data, split into channels.

        Returns
        -------
        torch.Tensor
            Output of final convolutional layer.

        """

        out, skips = X, []
        for enc in self._encs[:-1]:
            skips = [enc(out)] + skips
            out = self._down(skips[0])

        out = self._encs[-1](out)
        for (up, skip, dec) in zip(self._ups, skips, self._decs):
            out = maybe_interp(up(out), skip.shape[-1])
            out = dec(torch.cat((out, skip), dim=1))

        return out

    def _init_layers(self, trial: Trial) -> None:
        """Initilize the encoders, decoders, and upsampling layers."""

        self._encs = nn.ModuleList()
        self._decs = nn.ModuleList()
        self._down = nn.MaxPool1d(2)
        self._ups = nn.ModuleList()

        n_skips = trial.suggest_int('unet_n_skips', 3, 3)
        n_base = 2 ** trial.suggest_int('unet_n_base', 4, 5)
        sizes = [n_base * (2 ** i) for i in range(n_skips)]
        sizes = [self._n_channels_in] + sizes

        for a, b in iter_pairs(sizes):
            self._encs.append(_make_conv(a, b))

        for a, b in iter_pairs(sizes[::-1][:-1]):
            self._ups.append(nn.ConvTranspose1d(a, b, 2, 2))

        for a, b in iter_pairs(sizes[::-1][:-2]):
            self._decs.append(_make_conv(a, b))

        self._decs.append(_make_conv(sizes[2], self._n_channels_out, 1, True))

    @property
    def _n_channels_in(self) -> int:
        """
        Number of input channels. There is one input channel for each phase
        speed bin and one for each of wind, buoyancy frequency, and latitude.
        """

        return 3 + self._n_bins
    
    @property
    def _n_channels_out(self) -> int:
        """
        The network has an output channel for each phase speed bin as well as
        for the predicte sinks.
        """

        return self._n_bins + 1
    
def _make_conv(
    a: int,
    b: int,
    depth: int=2,
    final: bool=True
) -> nn.Sequential:
    """
    Build a convolutional layer with standardized arguments.

    Parameters
    ----------
    a, b
        Input and output channel numbers.
    depth
        How many convolutional layers should be included in the block.
    final
        Whether this is the last block, in which case the output will not be
        passed through a ReLU.

    Returns
    -------
    nn.Sequential
        Module containing the convolutions and activations.

    """

    kwargs = dict(kernel_size=3, padding='same')
    args = [nn.Conv1d(a, b, **kwargs), nn.ReLU()]

    for _ in range(depth - 1):
        args = args + [nn.Conv1d(b, b, **kwargs)]
        args = args + [nn.ReLU()]

    if final:
        args = args[:-1]

    return nn.Sequential(*args)
