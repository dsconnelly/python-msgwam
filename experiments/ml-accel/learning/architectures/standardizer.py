from typing import Any

import torch

from ..utils import standardize

class StandardizerMixin:
    """
    Mixin providing an interface for standardizing input features, initializing
    the statistics used to do so, and (un)pickling the related state. Since this
    class does not inherit from `nn.Module`, it should always be used as a mixin
    to a class that does.
    """

    def get_extra_state(self) -> dict[str, Any]:
        """
        Return the state related to input standardization.

        Returns
        -------
        dict[str]
            Dictionary containing mean and standard deviation tensors.

        """

        return {'means' : self.means, 'stds' : self.stds}
    
    def init_stats(self, *Xs: torch.Tensor) -> None:
        """
        Save the means and standard deviations to be used later. Must be called
        before `_standardize` can be called.

        Parameters
        ----------
        Xs
            List of tensors for which to compute and store statistics (e.g.,
            arrays of training inputs and targets).

        """

        self.means = [X.mean(dim=0) for X in Xs]
        self.stds = [X.std(dim=0) for X in Xs]
    
    def set_extra_state(self, state: dict[str]) -> None:
        """
        Set the state related to input standardization.

        Parameters
        ----------
        state
            Dictionary containing mean and standard deviation tensors.

        """

        self.means = state['means']
        self.stds = state['stds']

    def _standardize(self, X: torch.Tensor, i: int) -> torch.Tensor:
        """
        Standardize a tensor of input data using the precalculated statistics.

        Parameters
        ----------
        X
            Tensor to standardize.
        i
            Index into the lists of statistics stored by this object (e.g. zero
            for inputs and one for targets).

        Returns
        -------
        torch.Tensor
            Standardized tensor.

        """

        return standardize(X, self.means[i], self.stds[i])[0]
