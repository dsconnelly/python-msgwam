import torch

class LossMonitor:
    def __init__(self, patience: int, min_decrease: float) -> None:
        """
        Initialize a `LossMonitor`, which will track the validation loss and
        determine when it has plateaued, so that the training loop can either
        adjust a parameter or exit.

        Parameters
        ----------
        patience
            How many epochs the validation loss can go without decreasing before
            the monitor considers it to have plateaued.
        min_decrease
            Percentage by which the validation loss must decrease for the epoch
            to reset the waiting count.

        """

        self.patience = patience
        self.min_decrease = min_decrease
        self._reset(torch.inf)

    def has_plateaued(self, mse_va: float) -> bool:
        """
        Determine whether the validation loss has plateaued, and adjust the
        internal state accordingly.

        Parameters
        ----------
        mse_va
            Validation loss from the current epoch.

        Returns
        -------
        bool
            Whether the validation loss has plateaued. If the validation loss is
            greater than unity, we do not consider the curve to be plateaued
            because the model tends to need several epochs with large losses
            before training starts to work well.

        """

        if mse_va <= self.threshold or mse_va > 1:
            self._reset(mse_va)
            return False
        
        self.n_waiting = self.n_waiting + 1
        if self.n_waiting == self.patience:
            self._reset(mse_va)
            return True
        
        return False

    def _reset(self, mse_va: float) -> None:
        """
        Reset the monitor to start waiting from the current validation loss.

        Parameters
        ----------
        mse_va
            Validation loss against which to check for decreases.
 
        """

        self.threshold = (1 - self.min_decrease) * mse_va
        self.n_waiting = 0