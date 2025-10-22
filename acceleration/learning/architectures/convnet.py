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
    def __init__(self, trial: Trial) -> None:
        """Instantiate the network layers."""

        super().__init__()
        self._init_layers(trial)
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
        W = self._amp_block(self._pool(X).squeeze())[..., None]
        Y = self._shape_block(X)

        Y = self._postprocess(Y)
        W = W.reshape(-1, 2, self._n_bins, 1)

        return Y, W

    def _init_layers(self, trial: Trial) -> None:
        """Initiate various blocks of convolutional and dense layers."""

        options = [1, 2, 5]
        i = trial.suggest_int('n_bin_idx', 1, len(options) - 1)
        self._n_bins = options[i]

        conv_act = trial.suggest_categorical('conv_act', _ACTIVATIONS.keys())
        fc_act = trial.suggest_categorical('fc_act', _ACTIVATIONS.keys())
        
        meta_width = trial.suggest_int('meta_width', 32, 64, step=32)
        self._meta_block = nn.Sequential(
            nn.Linear(self._n_meta, meta_width), _ACTIVATIONS[fc_act](),
            nn.Linear(meta_width, config.n_grid - 1)
        )

        conv_dropout = trial.suggest_float('conv_dropout', 0, 0.2)
        fc_dropout = trial.suggest_float('fc_dropout', 0.2, 0.5)

        kernel = 1 + 2 * trial.suggest_int('kernel', 1, 4)
        use_bn = trial.suggest_categorical('use_bn', [False, True])
        use_res = trial.suggest_categorical('use_res', [False, True])
        conv_args = (kernel, use_bn, use_res, conv_dropout, conv_act)
        
        min_channels = trial.suggest_int('min_channels', 32, 96)
        max_channels = 2 ** trial.suggest_int('max_channels', 6, 8)
        n_joint_convs = trial.suggest_int('n_joint_convs', 2, 6)

        sizes = [min_channels]
        for _ in range(n_joint_convs):
            sizes = sizes + [min(2 * sizes[-1], max_channels)]

        sizes = [self._n_channels_in] + sizes
        n_split = sizes[-1]

        convs = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            convs = convs + [_ConvBlock(a, b, *conv_args)]

        self._joint_block = nn.Sequential(*convs)
        self._pool = nn.AdaptiveAvgPool1d(1)

        depth = trial.suggest_int('amp_depth', 2, 4)
        width = trial.suggest_int('amp_width', 128, 512)

        sizes = [n_split] + [width] * depth + [self._n_channels_out]
        args = (sizes, -1 if use_bn else 0, fc_act, fc_dropout)
        self._amp_block = get_block(*args, final=True)

        n_shape_convs = trial.suggest_int('n_shape_convs', 2, 6)
        sizes = [n_split] * n_shape_convs + [self._n_channels_out]

        convs = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            is_final = b == self._n_channels_out
            convs = convs + [_ConvBlock(a, b, *conv_args, is_final)]

        self._shape_block = nn.Sequential(*convs)

        options = {
            'relu' : nn.functional.relu,
            'softplus' : nn.functional.softplus,
            'exp' : torch.exp
        }

        self._normalize = trial.suggest_categorical('normalize', [True, False])
        func_name = trial.suggest_categorical('pos_func', options.keys())
        self._pos_func = options[func_name]

    @property
    def _n_channels_in(self) -> int:
        """
        One channel for momentum, two for atmospheric column information, and
        one for the processed metadata.
        """

        return self._n_bins + 3
    
    @property
    def _n_channels_out(self) -> int:
        """One channel for momentum plus one for the sink."""

        return 2 * self._n_bins

    @property
    def _n_meta(self) -> int:
        """Provided are the budget and the latitude."""

        return 2

    def _postprocess(self, Y: torch.Tensor) -> torch.Tensor:
        """
        The vertical fluxes are constrained to be non-negative, and the sink
        (encoded as the horizontal flux in the lowest phase speed bin) must be
        non-positive. The vertical flux in each bin at the top of the domain
        must vanish. Normalization to sum to plus/minus 1 may also occur.
        """

        Y = Y.reshape(-1, 2, self._n_bins, config.n_grid - 1)
        Y_v, Y_h = self._pos_func(Y[:, 0]), Y[:, 1]

        mask = torch.ones_like(Y_v)
        mask[..., -1] = 0
        Y_v = Y_v * mask

        Y_h = torch.cat((-self._pos_func(Y_h[:, :1]), Y_h[:, 1:]), dim=1)
        Y = torch.stack((Y_v, Y_h), dim=1)

        norms = vector_norm(Y, dim=-1, keepdim=True)
        return Y / torch.where(norms > 1e-12, norms, torch.ones(1))

class _ConvBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, use_bn: bool,
        use_res: bool, dropout: float, activation: str, final: bool=False
    ) -> None:
        """
        Initialize the convolutional layer and various other layers as specified
        by the provided arguments.
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
        
        self._layers = nn.Sequential(*args)
        self._use_res = use_res and (n_in == n_out) and (not final)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Convolve, and maybe apply the skip connection."""

        out = self._layers(X)
        if self._use_res:
            out = out + X

        return out
