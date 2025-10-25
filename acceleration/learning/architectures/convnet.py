import torch, torch.nn as nn

from optuna.trial import Trial

from msgwam import config

from torch.linalg import vector_norm

from .utils import get_block, xavier_init

_ACTIVATIONS = {
    'relu' : nn.ReLU,
    'leaky' : nn.LeakyReLU,
    'tanh' : nn.Tanh
}

class ConvNet(nn.Module):
    _one: torch.Tensor

    def __init__(self, trial: Trial) -> None:
        """Instantiate the network layers."""

        super().__init__()
        self._init_layers(trial)
        self.register_buffer('_one', torch.ones(1))
        self.apply(xavier_init)

    def forward(
        self,
        C: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the joint block, then use the amplitude block to predict `W` and
        the shape block to predict `Y`.
        """

        C, meta = C[:, :-2], C[:, -2:]
        C = C.reshape(-1, 2, config.n_grid - 1)
        meta = self._meta_block(meta)[:, None]

        X = self._joint_block(torch.cat((C, meta, M), dim=1))
        W = self._amp_block(self._pool(X).squeeze())
        Y = self._shape_block(X)

        return self._postprocess(Y), W[..., None, None]

    def _init_layers(self, trial: Trial) -> None:
        """
        Initialize the convolutional and dense layers of the network, along with
        a few other important hyperparameters.

        Parameters
        ----------
        trial
            Current trial, used to define the architecture.

        """

        options = [1, 2, 3, 4, 6]
        i = trial.suggest_int('n_bin_idx', 1, len(options) - 1)
        self._n_bins = options[i]

        use_bn = trial.suggest_categorical('use_bn', [True, False])
        n_split = self._init_conv_blocks(trial, use_bn)
        self._init_dense_blocks(trial, n_split, use_bn)

        options = {
            'relu' : nn.functional.relu,
            'softplus' : nn.functional.softplus,
            'exp' : torch.exp
        }

        func_name = trial.suggest_categorical('pos_func', options.keys())
        self._pos_func = options[func_name]

    def _init_conv_blocks(self, trial: Trial, use_bn: bool) -> int:
        """
        Initialize the two convolutional parts of the network: the joint block
        and the block that predicts the shape profiles.

        Parameters
        ----------
        trial
            Current trial.
        use_bn
            Whether to use batch normalization.

        Returns
        -------
        int
            Number of layers after the last convolution in the joint block, so
            that the dense layers can be created with the correct size.

        """

        kernel = 1 + 2 * trial.suggest_int('kernel', 1, 4)
        dropout = trial.suggest_float('conv_dropout', 0, 0.2)
        use_res = trial.suggest_categorical('use_res', [True, False])
        act_str = trial.suggest_categorical('conv_act', _ACTIVATIONS.keys())
        
        min_channels = trial.suggest_int('min_channels', 32, 64)
        max_channels = 2 ** trial.suggest_int('max_channels', 7, 9)

        pool_options = [2, 5, 2]
        n_joint_convs = trial.suggest_int('n_joint_convs', 3, 6)
        n_pools = trial.suggest_int('n_pools', 0, len(pool_options))
        pools = pool_options[:n_pools] + [0] * (n_joint_convs - n_pools)
        
        sizes = [self._n_channels_in, min_channels]
        args = (kernel, use_bn, use_res, dropout, act_str)
        convs = []
        
        while len(sizes) < n_joint_convs + 1:
            sizes = sizes + [min(2 * sizes[-1], max_channels)]

        for a, b, pool in zip(sizes[:-1], sizes[1:], pools):
            convs.append(_ConvBlock(a, b, pool, *args))

        self._joint_block = nn.Sequential(*convs)

        n_shape_convs = trial.suggest_int('n_shape_convs', max(n_pools, 2), 6)
        pools = [0] * (n_shape_convs - n_pools) + pools[::-1][-n_pools:]
        sizes = [sizes[-1]] * n_shape_convs + [self._n_channels_out]

        convs = []
        for i, (a, b, pool) in enumerate(zip(sizes[:-1], sizes[1:], pools)):
            convs.append(_ConvBlock(a, b, -pool, *args, i == n_shape_convs - 1))

        self._shape_block = nn.Sequential(*convs)

        return sizes[0]

    def _init_dense_blocks(
        self,
        trial: Trial,
        n_split: int,
        use_bn: bool
    ) -> None:
        """
        Initialize the two fully-connected blocks of the network: the block that
        processes the metadata, and the block that predicts the amplitudes.

        Parameters
        ----------
        trial
            Current trial.
        n_split
            Number of channels after the last convolution of the joint block.
        use_bn
            Whether to use batch normalization.

        """

        dropout = trial.suggest_float('dense_dropout', 0.2, 0.5)
        act_str = trial.suggest_categorical('dense_act', _ACTIVATIONS.keys())
        width = trial.suggest_int('meta_width', 32, 64, step=32)
        
        self._meta_block = nn.Sequential(
            nn.Linear(self._n_meta, width), _ACTIVATIONS[act_str](),
            nn.Linear(width, config.n_grid - 1)
        )

        depth = trial.suggest_int('amp_depth', 2, 5)
        width = trial.suggest_int('amp_width', 128, 512)
        sizes = [n_split] + [width] * depth + [2]

        args = (sizes, -1 if use_bn else 0, act_str, dropout)
        self._amp_block = get_block(*args, final=True)
        self._pool = nn.AdaptiveAvgPool1d(1)

    @property
    def _n_channels_in(self) -> int:
        """
        The joint block has one channel for each phase speed bin, one for the
        wind, one for buoyancy frequency, and one for the encoded metadata.
        """

        return self._n_bins + 3
    
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

        Y_v, Y_h = Y.reshape(-1, 2, self._n_bins, Y.shape[-1]).transpose(0, 1)
        Y_h = torch.cat((-self._pos_func(Y_h[:, :1]), Y_h[:, 1:]), dim=1)
        Y = torch.stack((self._pos_func(Y_v), Y_h), dim=1)

        norms = vector_norm(Y, dim=(-2, -1), keepdim=True)
        return Y / torch.where(norms > 1e-12, norms, self._one)

class _ConvBlock(nn.Module):
    def __init__(
        self,
        n_in: int, n_out: int,
        pool: int, kernel: int,
        use_bn: bool, use_res: bool,
        dropout: float, activation: str,
        final: bool=False
    ) -> None:
        """
        Initialize a block containing a convolution and auxiliary modules.

        Parameters
        ----------
        n_in, n_out
            Number of input and output channels.
        pool
            If positive, the block will be followed by a max pooling operation
            with kernel `pool`. If negative, the block will be preceded by an
            upsampling operation with scale factor `pool`. If zero, the sequence
            length is not changed by this block.
        kernel
            Kernel size. Padding will be set to `'same'`.
        use_bn
            Whether to use batch normalization.
        use_res
            Whether to include an additive residual connection. Only has an
            effect if `n_in == n_out`.
        dropout
            Dropout rate.
        activation
            Key to `_ACTIVATIONS` specifying what activation function to use.
        final
            Whether this is an output layer of the overall network, in which
            case the last operation will be the convolution itself.

        """

        super().__init__()

        args = [nn.Conv1d(n_in, n_out, kernel, padding='same')]

        if use_bn:
            args = args + [nn.BatchNorm1d(n_out)]

        args = args + [_ACTIVATIONS[activation]()]
        args = args + [nn.Dropout(dropout)]
        
        if final:
            while not isinstance(args[-1], nn.Conv1d):
                args = args[:-1]
        
        if pool > 0:
            self._resample = nn.MaxPool1d(pool)
            args = args + [self._resample]

        elif pool < 0:
            kwargs = {'scale_factor' : abs(pool), 'mode' : 'nearest'}
            self._resample = nn.Upsample(**kwargs)
            args = [self._resample] + args

        else:
            self._resample = nn.Identity()

        self._layers = nn.Sequential(*args)
        self._use_res = use_res and (n_in == n_out) and (not final)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Convolve, and maybe apply the skip connection."""

        out = self._layers(X)
        if self._use_res:
            out = out + self._resample(X)

        return out
