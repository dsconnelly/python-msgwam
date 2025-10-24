from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch, torch.nn as nn

from optuna.trial import Trial

class AbstractLoss(nn.Module, ABC):
    _scales_W: torch.Tensor

    def __init__(self, trial: Trial, W: torch.Tensor) -> None:
        """
        At initialization, calculates the scales (in transformed space) that
        will be used to weight loss in `W` later on.

        Parameters
        ----------
        trial
            Current trial, used to sample a `Y` bias.
        W
            `W` samples from the training data.
        bias_Y
            Bias towards `Y` to use when aggregating losses.

        """

        super().__init__()
        self._scale_Y = 0.2
        self._bias_Y = trial.suggest_float('bias_Y', 0.5, 0.98)

        self._set_buffers(W.cpu().numpy())
        scales_W = self._get_scales_W(W.cpu())
        self.register_buffer('_scales_W', scales_W)

    def forward(
        self,
        Y: torch.Tensor,
        W: torch.Tensor,
        Y_hat: torch.Tensor,
        W_hat: torch.Tensor,
        reduce: bool=True
    ) -> tuple[torch.tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the losses over both target types.

        Parameters
        ----------
        Y, Y_hat
            True and network-predicted shape profiles.
        W, W_hat
            True and network-predict scale profiles, where `W_hat` is already in
            the transformed space.
        reduce
            Whether to take a mean over all samples and entries (for training)
            or to leave the arrays intact (for plotting).
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            Losses for `Y` and `W` targets, respectively.
        Optional[torch.Tensor]
            If `reduce`, the combined loss.

        """

        mask = self._get_mask(W)
        mask_hat = self._get_mask(self._transform(W_hat, inverse=True))
        loss_W = ((self._transform(W) - W_hat) / self._scales_W) ** 2
        loss_W = loss_W * (mask | mask_hat).int()

        maxes, _ = Y.max(dim=-1, keepdim=True)
        mins, _ = Y.min(dim=-1, keepdim=True)

        scales_Y = (maxes - mins) / 2
        scales_Y[scales_Y < 1e-6] = 1        
        loss_Y = mask * ((Y - Y_hat) / scales_Y) ** 2

        if reduce:
            loss_Y, loss_W = loss_Y.mean(), loss_W.mean()
            return loss_Y, loss_W, self._combine(loss_Y, loss_W)
        
        return loss_Y, loss_W, None

    def _combine(
        self,
        loss_Y: torch.Tensor,
        loss_W: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the `Y` and `W` losses, possibly with an unequal waiting, and
        return an aggregate loss that can be used for training or evaluation.

        Paramters
        ---------
        loss_Y, loss_W
            Tensors of (reduced) losses for each target type.

        Returns
        -------
        torch.Tensor
            Aggregate loss.

        """

        b = self._bias_Y if self.training else 0.5
        return b * loss_Y + (1 - b) * loss_W

    @abstractmethod
    def _get_mask(self, W: torch.Tensor) -> torch.Tensor:
        """
        Get a mask indicating which scale parameters are active (that is, which
        shape profiles should be graded and not ignored).
        """
        ...

    def _get_scales_W(self, W: torch.Tensor) -> torch.Tensor:
        """
        Given the `W` data from the training set, calculate and set any buffers
        that will be needed to compute losses later.

        Parameters
        ----------
        W
            `W` samples from the training data, cast to `ndarray` but still
            sharing memory with the actual training data.

        """

        a = W.clone()
        a[~self._get_mask(a)] = torch.nan
        a = self._transform(a).numpy()

        scales = np.nanstd(a, axis=0)        
        return torch.as_tensor(scales)

    @abstractmethod
    def _set_buffers(self, W: np.ndarray) -> None:
        """Set any additional buffers needed before calculating scales."""

    @abstractmethod
    def _transform(self, W: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        Transform the weights into the space where their loss is calculated. If
        `inverse`, perform the opposite transformation.
        """
        ...

class BulkLoss(AbstractLoss):
    _threshold: torch.Tensor

    def _get_mask(self, W: torch.Tensor) -> torch.Tensor:
        """The active weights are simply those greater than the threshold."""

        return W > self._threshold

    def _set_buffers(self, W: np.ndarray) -> None:
        """
        The threshold is set as a quarter of the smallest nonzero scale
        parameter found in the training data.
        """

        W[W == 0] = np.nan
        threshold = np.nanquantile(W, q=0.02, axis=0)
        threshold = np.maximum(threshold, 0.00001)
        W[np.isnan(W)] = 0

        self.register_buffer('_threshold', torch.as_tensor(threshold))

    def _transform(self, W: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        The transformation is just the natural log, with clipping.
        """

        if inverse:
            return torch.exp(W)
        
        W = torch.clip(W, min=self._threshold)
        return torch.log(W)
