import torch, torch.nn as nn

from optuna.trial import Trial

from msgwam import config

from .utils import ACTIVATIONS, iter_pairs, maybe_interp, xavier_init

class UNet(nn.Module):
    _z: torch.Tensor

    def __init__(self, trial: Trial, n_bins: int) -> None:
        """
        Initialize a UNet with a specific number of phase speed bins.

        Parameters
        ----------
        trial
            Current trial, used to sample parameters for the UNet.
        n_bins
            Number of phase speed bins into which the bulk momentum is split.

        """

        super().__init__()

        self._n_bins = n_bins
        self._init_settings(trial)
        self._init_dense(trial)
        self._init_convs(trial)

        self.apply(xavier_init)
        self.float()

    def forward(
        self,
        C: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the joint block, then use the amplitude block to predict `W` and
        the shape block to predict `Y`.
        """

        C, meta = C[:, :-2], C[:, -2:]
        C = C.reshape(-1, 2, config.n_grid - 1)
        meta = self._meta_block(meta)[:, None]

        if self._use_z:
            z = torch.tile(self._z, (C.shape[0], 1, 1))
            meta = torch.cat((meta, z), dim=1)

        if not self._use_M_tot:
            M = M[:, :self._n_bins]

        X = torch.cat((C, meta, M), dim=1)
        Y, skips = X, []

        for enc in self._encs:
            skips = [enc(Y)] + skips
            Y = self._down(skips[0])

        Y = self._bottleneck(Y)
        for i, (up, dec) in enumerate(zip(self._ups, self._decs)):
            Y = maybe_interp(up(Y), skips[i].shape[-1])
            Y = dec(torch.cat((Y, skips[i]), dim=1))

        Y = self._postprocess(Y)

        if self._use_mask:
            mask = M[:, :self._n_bins] > M.min() + 1e-14
            Y = Y * mask[:, None]

        return Y

    def _init_settings(self, trial: Trial) -> None:
        """Sample general hyperparameters from the trial."""

        self._use_M_tot = trial.suggest_categorical('use_M_tot', [True, False])
        self._use_mask = trial.suggest_categorical('use_mask', [True, False])

        self._use_z = trial.suggest_categorical('use_z', [True, False])
        z = torch.linspace(-1, 1, config.n_grid - 1)
        self.register_buffer('_z', z)

        options = {
            'relu' : nn.functional.relu,
            'softplus' : nn.functional.softplus,
            'exp' : torch.exp
        }

        func_name = trial.suggest_categorical('pos_func', options.keys())
        self._pos_func = options[func_name]

    def _init_dense(self, trial: Trial) -> None:
        """Initialize the dense layer that processes the metadata."""

        act_str = trial.suggest_categorical('dense_act', ACTIVATIONS.keys())
        width = trial.suggest_int('meta_width', 32, 64, step=32)
        
        self._meta_block = nn.Sequential(
            nn.Linear(self._n_meta, width), ACTIVATIONS[act_str](),
            nn.Linear(width, config.n_grid - 1)
        )

    def _init_convs(self, trial: Trial) -> None:
        """Initilize the encoders, decoders, and upsampling layers."""

        args = (
            1 + 2 * trial.suggest_int('kernel', 1, 4),
            trial.suggest_categorical('use_bn', [True, False]),
            trial.suggest_categorical('conv_act', ACTIVATIONS.keys()),
            trial.suggest_float('dropout', 0, 0.2)
        )

        self._encs = nn.ModuleList()
        self._decs = nn.ModuleList()
        self._down = nn.MaxPool1d(2)
        self._ups = nn.ModuleList()

        n_skips = trial.suggest_int('n_skips', 3, 5)
        min_channels = trial.suggest_int('min_channels', 8, 32)
        sizes = [min_channels * (2 ** i) for i in range(n_skips)]
        sizes = [self._n_channels_in] + sizes

        for a, b in iter_pairs(sizes[:-1]):
            self._encs.append(_make_conv(a, b, *args))

        a, b = sizes[-2:]
        self._bottleneck = _make_conv(a, b, *args)

        for a, b in iter_pairs(sizes[::-1][:-1]):
            self._ups.append(nn.ConvTranspose1d(a, b, 2, 2))

        for a, b in iter_pairs(sizes[::-1][:-2]):
            self._decs.append(_make_conv(a, b, *args))

        a, b = sizes[2], self._n_channels_out
        self._decs.append(_make_conv(a, b, *args, True))

    @property
    def _n_channels_in(self) -> int:
        """
        The joint block has one channel for each phase speed bin, one for the
        wind, one for buoyancy frequency, and one for the encoded metadata.
        """

        return self._n_bins * (1 + self._use_M_tot) + 3 + self._use_z
    
    @property
    def _n_channels_out(self) -> int:
        """
        For each phase speed bin, the network returns horizontal and vertical
        momentum flux profiles.
        """

        return 2 * self._n_bins

    @property
    def _n_meta(self) -> int:
        """
        The metadata encoder block takes two pieces of information: the latitude
        and the natural log of the momentum budget.
        """

        return 2
    
    def _postprocess(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the outputs of the shape block.

        Parameters
        ----------
        Y
            Output of the shape block, with `2 * self._n_bins` channels.

        Returns
        -------
        torch.Tensor
            Postprocessed output. The vertical fluxes are constrained to be non-
            negative, and the sink (encoded as the horizontal flux in the lowest
            phase speed bin) is non-positive. The data is reshaped into vertical
            and horizontal profiles, and each is scaled to unit norm.

        """

        Y = Y.reshape(-1, 2, self._n_bins, Y.shape[-1])
        mask = torch.ones_like(Y)
        mask[:, 1] = -1

        return mask * self._pos_func(Y)

def _make_conv(
    n_in: int,
    n_out: int,
    kernel: int,
    use_bn: int,
    activation: str,
    dropout: float,
    final: bool=False
) -> nn.Sequential:
    """Make a convolutional layer."""

    args = [nn.Conv1d(n_in, n_out, kernel, padding='same')]
    args = args + [ACTIVATIONS[activation]()]
    args = args + [nn.Dropout(dropout)]

    if use_bn:
        args.insert(1, nn.BatchNorm1d(n_out))

    if final:
        while not isinstance(args[-1], nn.Conv1d):
            args = args[:-1]

    return nn.Sequential(*args)
