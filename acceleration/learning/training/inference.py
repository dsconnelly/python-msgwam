import json

from typing import Optional

import torch, torch.nn as nn

from optuna.trial import FixedTrial
from torch.nn.functional import pad as _PAD

from ..architectures import ConvNet

from .io import prepare_data
from .transforms import Transform

def serialize_model(
    inputs: Optional[tuple[ConvNet, Transform, Transform]]=None
) -> None:
    """
    Load a trained model, wrap it in a module containing appropriate input and
    output processing, and serialize the pipeline as a `ScriptModule`.

    Parameters
    ----------
    inputs
        Model and transforms to wrap. If `None` (e.g. if this function is being
        called from the command line) the trained model is loaded from disk.

    """

    if inputs is None:
        with open('data/ml-accel/models/hyperparameters.json') as f:
            trial = FixedTrial(json.load(f))
            model = ConvNet(trial).eval()
            n_bins = model._n_bins

        kwargs = dict(weights_only=True, map_location=torch.device('cpu'))
        state = torch.load('data/ml-accel/models/state-best.pkl', **kwargs)
        model.load_state_dict(state['model'])

        _, _, (C_trans, M_trans) = prepare_data(n_bins, 'te', 2500, False, 0)

    else:
        model, C_trans, M_trans = inputs

    with torch.inference_mode():
        wrapper = Inferer(model, C_trans, M_trans)
        scripted = torch.jit.script(wrapper.float())
        scripted = torch.jit.optimize_for_inference(scripted)

    torch.jit.save(scripted, 'data/ml-accel/models/scripted.jit')

class Inferer(nn.Module):
    def __init__(
        self,
        model: ConvNet,
        C_trans: Transform,
        M_trans: Transform,
        batch_size: int=64
    ) -> None:
        """
        Initialize a module that wraps a trained network with appropriate input
        and output processing.

        Parameters
        ----------
        model
            Trained neural network.
        C_trans, M_trans
            Transforms to apply to network inputs.
        batch_size
            Batch size to use at inference time. Batches larger than this value
            will be processed in chunks for improved performance.

        """

        super().__init__()
        self._batch_size = batch_size

        self._model = model
        self._C_trans = C_trans
        self._M_trans = M_trans

    def forward(
        self,
        C: torch.Tensor,
        M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the neural network.

        Parameters
        ----------
        C, M
            Neural network inputs.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensors of vertical and horizontal fluxes, respectively.

        """

        C, M = self._C_trans(C.float()), self._M_trans(M.float())
        out = torch.zeros((M.shape[0], 2, M.shape[1], M.shape[2])).float()

        i = 0
        while i < M.shape[0]:
            j = min(M.shape[0], i + self._batch_size)
            Y, W = self._model(C[i:j], M[i:j])
            out[i:j] = torch.exp(W) * Y

            i = j

        F_v = _PAD(out[:, 0], (1, 0))
        F_h = _PAD(out[:, 1], (0, 0, 0, 1))

        return F_v.double(), F_h.double()
