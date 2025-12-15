import json

from typing import Optional

import torch, torch.nn as nn

from optuna.trial import FixedTrial

from msgwam import config

from ..architectures import ConvNet

from .io import prepare_data
from .transforms import Transform, get_T_from_logits

def serialize_model(
    model_path: Optional[str]=None,
    inputs: Optional[tuple[ConvNet, Transform, Transform]]=None,
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

        _, _, transforms = prepare_data(n_bins, 'te', None, False, 0)

    else:
        model, *transforms = inputs

    with torch.inference_mode():
        wrapper = Inferer(model, *transforms)
        scripted = torch.jit.script(wrapper.float().cpu())
        scripted = torch.jit.optimize_for_inference(scripted)

    if model_path is not None:
        torch.jit.save(scripted, model_path)

    return scripted

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
        Y_trans
            Transform to invert on network outputs.
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
    ) -> torch.Tensor:
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

        N = C[:, None, -config.n_grid:-1].float()
        f = C[:, -1, None, None].float()
    
        n_bins = self._model._n_bins
        C, M = self._C_trans(C.float()), self._M_trans(M.float())
        out = torch.zeros((M.shape[0], n_bins, M.shape[2])).float()

        i = 0
        while i < M.shape[0]:
            j = min(M.shape[0], i + self._batch_size)
            out[i:j] = self._model(C[i:j], M[i:j])

            i = j

        return get_T_from_logits(N, f, out).double()
