import torch, torch.nn as nn

class PartiallyMonotone(nn.Module):
    """
    Fully connected layer constrained to be monotonic in its last input. Can
    have multiple output channels. Inspired by Cannon (2018).
    """

    def __init__(
        self,
        n_free: int,
        n_outputs: int,
        decreasing: bool=False
    ) -> None:
        """
        Create a linear module, responsible for the non-monotonic behavior, and
        initialize the log of the slope applied to the monotonic input.

        Parameters
        ----------
        n_free
            Number of non-monotonic inputs to this module.
        n_outputs
            Number of output channels.
        decreasing
            Whether this module should be monotonically decreasing in its last
            input instead of increasing.

        """

        super().__init__()

        self._linear = nn.Linear(n_free, n_outputs)
        self._log_weight = nn.Parameter(torch.empty((n_outputs, 1)))
        self.sign = -1 if decreasing else 1

        xavier_init(self._linear)
        xavier_init(self._log_weight)

    def forward(
        self,
        X_free: torch.Tensor,
        X_mono: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the module to data.

        Parameters
        ----------
        X_free
            Tensor containing non-monotonic inputs. Should have the number of
            columns specified by `n_free` at initialization.
        X_mono
            Tensor containing monotonic inputs. Should have a single column.

        Returns
        -------
        torch.Tensor
            Output values, each one of which is monotonic in `X_mono`.

        """

        weight = self.sign * torch.exp(self._log_weight)
        output = nn.functional.linear(X_mono, weight)
        output = output + self._linear(X_free)

        return output

def xavier_init(a: nn.Module | torch.Tensor) -> None:
    """
    Apply Xavier initialization a linear layer or a weight matrix.

    Parameters
    ----------
    a
        Module or weight matrix to potentially initialize. If an `nn.Linear` is
        passed, its weight matrix will be extracted and initialized.

    """

    if isinstance(a, nn.Linear):
        a = a.weight

    if isinstance(a, torch.Tensor):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(a, gain=gain)
