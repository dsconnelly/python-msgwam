import torch

def root_transform(a: torch.Tensor, root: int) -> torch.Tensor:
    """
    Transform data by taking a sign-aware root.

    Parameters
    ----------
    a
        Data to be transformed.
    root
        Order of the root to take. For example, `root=3` takes a cube root.

    Returns
    -------
    torch.Tensor
        Transformed data.

    """

    return torch.sign(a) * (abs(a) ** (1 / root))

def standardize(
    a: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor
) -> torch.Tensor:
    """
    Standardize data given means and standard deviations. Columns where the
    standard deviation is zero will be set to zero.

    Parameters
    ----------
    a
        Data to standardize.
    means
        Precomputed means to use during standardization.
    stds
        Precomputed standard deviations to use during standardization.

    Returns
    -------
    torch.Tensor
        Standardized data.

    """

    sdx = stds != 0
    out = torch.zeros_like(a)
    out[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    return out
    