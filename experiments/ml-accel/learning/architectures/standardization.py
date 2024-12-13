from typing import Any

import torch

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
    
    def init_stats(self, X: torch.Tensor) -> None:
        """
        Save the mean and standard deviation information to be used later. Must
        be called before `_standardize` can be called.

        Parameters
        ----------
        X
            Two-dimensional tensor whose first dimension ranges over training
            samples and whose second dimension ranges over input features.

        """

        self.means = X.mean(dim=0)
        self.stds = X.std(dim=0)
    
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

    def _standardize(self, X: torch.Tensor) -> torch.Tensor:
        """
        Standardize a tensor of input data using the precalculated statistics.

        Parameters
        ----------
        X
            Tensor to standardize.

        Returns
        -------
        torch.Tensor
            Standardized tensor.

        """

        sdx = self.stds > 0
        output = torch.zeros_like(X)
        output[:, sdx] = (X - self.means)[:, sdx] / self.stds[sdx]

        return output
