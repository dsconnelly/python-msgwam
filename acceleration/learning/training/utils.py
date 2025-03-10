from typing import Optional

import torch

def get_kinds(phase: str) -> list[str]:
    """
    Get the kinds of data that should be loaded for a given training phase.

    Parameters
    ----------
    phase
        Phase of training, as passed to `train_networks`.

    Returns
    -------
    list[str]
        List of kinds to pass to `get_loader`.

    """

    return {
        'encoding' : ['Ro', 'Fo'],
        'stepping' : ['u', 'S', 'Ri', 'Ro'],
        'joint' : ['u', 'S', 'Ri', 'Fo']
    }[phase]

def get_pipeline_spec(phase: str) -> dict[str, str]:
    """
    Get a dictionary describing the data pipeline for a given training phase.

    Parameters
    ----------
    phase
        Phase of training, as passed to `train_networks`.

    Returns
    -------
    dict[str, str]
        Dictionary whose keys indicate `BaseNet` subclasses and whose values are
        either `'new'`, in which case a new component will be created, `'load'`,
        in which case a previously-trained component will be loaded and trained,
        or `'frozen'`, in which case a previously-trained component will be
        loaded but its weights should be frozen.

    """
    
    if phase == 'encoding':
        return {'encoder' : 'new', 'observer' : 'new'}
    
    if phase == 'stepping':
        return {'encoder' : 'frozen', 'stepper' : 'new'}
    
    if phase == 'joint':
        return {
            'encoder' : 'loaded',
            'stepper' : 'loaded',
            'observer' : 'loaded'
        }
    
    raise ValueError(f'Unknown phase: {phase}')

def get_subsets(eval_type: str) -> tuple[list[str], list[str]]:
    """
    Return the subset suffixes for a given evaluation type.

    Parameters
    ----------
    eval_type
        Evaluation dataset specifier, as passed to `train_networks`.

    Returns
    -------
    list[str], list[str]
        Subsets to use for training and evaluation, respectively.

    """

    if eval_type == 'validation':
        return ['tr'], ['va']
    
    if eval_type == 'test':
        return ['tr', 'va'], ['te']
    
    raise ValueError(f'Unknown eval_type: {eval_type}')

def standardize(
    a: torch.Tensor,
    means: Optional[torch.Tensor]=None,
    stds: Optional[torch.Tensor]=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize a tensor along the first dimension.

    Parameters
    ----------
    a
        Tensor containing data to standardize.
    means
        Means to use during standardization. If `None`, the mean along the first
        dimension will be computed and used.
    stds
        Standard deviations to use during standardization. If `None`, the
        standard deviation along the first dimension will be computed and used.

    Returns
    -------
    torch.Tensor
        Standardized data.
    torch.Tensor, torch.Tensor
        Means and standard deviations used during standardization. If either of
        these statistics was provided, they will be returned as is.

    """

    if means is None:
        means = a.mean(dim=0)

    if stds is None:
        stds = a.std(dim=0)

    sdx = stds > 0
    output = torch.zeros_like(a)
    output[:, sdx] = (a - means)[:, sdx] / stds[sdx]

    return output, means, stds
