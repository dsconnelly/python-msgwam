import torch, torch.nn as nn

from optuna.trial import Trial
from torch.linalg import vector_norm

from msgwam import config

from .utils import get_block, xavier_init

_ACTIVATIONS = {
    'relu' : nn.ReLU,
    'leaky' : nn.LeakyReLU,
    'tanh' : nn.Tanh
}

class ConvNet(nn.Module):
    def __init__(self, trial: Trial) -> None:
        """Instantiate the network layers."""

        super().__init__()
        self._init_layers(trial)
        self.apply(xavier_init)

        one = torch.ones(1)
        self.register_buffer('_one', one)

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

        pool_options = [5, 2, 2]
        n_joint_convs = trial.suggest_int('n_joint_convs', 3, 6)
        n_pools = min(n_joint_convs, len(pool_options))

        sizes = [self._n_channels_in, min_channels]
        pools = pool_options[:n_pools] + [0] * (n_joint_convs - n_pools)
        
        while len(sizes) < n_joint_convs + 1:
            sizes.append(min(2 * sizes[-1], max_channels))

        joint_convs = []
        for a, b, pool in zip(sizes[:-1], sizes[1:], pools):
            joint_convs.append(_ConvBlock(
                a, b,
                kernel, pool,
                use_bn, use_res,
                act_str, dropout
            ))

        n_shape_convs = trial.suggest_int('n_shape_convs', 2, 6)
        pools = [0] * (n_shape_convs - n_pools) + pools[::-1][-n_pools:]
        sizes = [sizes[-1]] * n_shape_convs + [self._n_channels_out]

        while len(pools) > n_shape_convs:
            pools[1] = pools[0] * pools[1]
            pools = pools[1:]

        shape_convs = []
        for i, (a, b, pool) in enumerate(zip(sizes[:-1], sizes[1:], pools)):
            shape_convs.append(_ConvBlock(
                a, b,
                kernel, -pool,
                use_bn, use_res,
                act_str, dropout,
                final=(i == n_shape_convs - 1)
            ))

        self._joint_block = nn.Sequential(*joint_convs)
        self._shape_block = nn.Sequential(*shape_convs)

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

        Y = Y.reshape(-1, 2, self._n_bins, Y.shape[-1])
        Y_h = torch.cat((-self._pos_func(Y[:, 1, :1]), Y[:, 1, 1:]), dim=1)
        Y = torch.stack((self._pos_func(Y[:, 0]), Y_h), dim=1)

        norms = vector_norm(Y, dim=(-2, -1), keepdim=True)
        return Y / torch.where(norms > 1e-12, norms, self._one)

class _ConvBlock(nn.Module):
    def __init__(
        self,
        n_in: int, n_out: int,
        kernel: int, pool: int,
        use_bn: bool, use_res: bool,
        activation: str,
        dropout: float,
        final: bool=False
    ) -> None:
        """
        Initialize a block containing a convolution and auxiliary modules. The
        convolutional block uses depthwise separable convolutions for speed.

        Parameters
        ----------
        n_in, n_out
            Number of input and output channels.
        kernel
            Kernel size. Padding will be set to `'same'`.
        pool
            If positive, the block will be followed by a max pooling operation
            with kernel `pool`. If negative, the block will be preceded by an
            upsampling operation with scale factor `pool`. If zero, the sequence
            length is not changed by this block.
        use_bn
            Whether to use batch normalization.
        use_res
            Whether to include an additive residual connection. Only has an
            effect if `n_in == n_out`.
        activation
            Key to `_ACTIVATIONS` specifying what activation function to use.
        dropout
            Dropout rate.
        final
            Whether this is an output layer of the overall network, in which
            case the last operation must be a convolution.

        """

        super().__init__()
        
        args = [
            nn.Conv1d(n_in, n_in, kernel, padding='same', groups=n_in),
            _ACTIVATIONS[activation](),
            nn.Conv1d(n_in, n_out, 1),
            nn.Dropout(dropout)
        ]

        if use_bn:
            args.insert(1, nn.BatchNorm1d(n_in))

        if final:
            while not isinstance(args[-1], nn.Conv1d):
                args = args[:-1]

        if pool > 0:
            self._resample = nn.MaxPool1d(pool)
            args.append(self._resample)

        elif pool < 0:
            kwargs = {'scale_factor' : abs(pool), 'mode' : 'nearest'}
            self._resample = nn.Upsample(**kwargs)
            args.insert(0, self._resample)

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