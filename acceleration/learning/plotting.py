from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from msgwam import config
from msgwam.utils import get_vertical_grids

from .architectures import BaseNet
from .training import get_loader

def plot_training_fluxes(tag: Optional[str]=None) -> None:
    """
    Plot examples of the training fluxes.
    
    Parameters
    ----------
    tag
        Indicates a trained model to plot outputs for. If `None`, only the
        targets themselves will be plotted.

    """

    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(4 * 3, 2 * 4.5)

    z = get_vertical_grids()[0] / 1000
    for i, subset in enumerate(['tr', 'te']):
        for *Xs, Y in get_loader(['R', 'F'], subset, 4096):
            if tag is not None:
                output = _apply_model(tag, *Xs)

            break

        kdx = np.random.permutation(Y.shape[0])
        for j, (k, ax) in enumerate(zip(kdx, axes[i])):
            ax.plot(1000 * Y[k], z, color='k')

            if tag is not None:
                ax.plot(1000 * output[k], z, color='royalblue')

            ax.set_xlim(-2, 2)
            ax.set_ylim(z.min(), z.max())
            ax.grid(color='lightgray')

            ax.set_xlabel('flux (mPa)')
            name = {'tr' : 'train', 'te' : 'test'}[subset]
            ax.set_title(f'sample {k} ({name})')

            if j == 0:
                ax.set_ylabel('height (km)')

    plt.tight_layout()
    plt.savefig(f'plots/{config.name}/training-fluxes.png')

def _apply_model(tag: str, *Xs: torch.Tensor) -> torch.Tensor:
    """
    Apply an encoder-observer pair saved to disk to loaded input data.

    Parameters
    ----------
    tag
        Trained model tag, as passed to `plot_training_fluxes`.
    Xs
        Model inputs to be passed to the neural network.

    Returns
    -------
    torch.Tensor
        Dimensionalized flux profiles.

    """

    encoder = BaseNet.from_kwargs(name='encoder', tag=tag)
    observer = BaseNet.from_kwargs(name='observer', tag=tag)
    encoder.eval(), observer.eval()

    kwargs = {'weights_only' : True}
    means = torch.load(f'data/{config.name}/models/means-{tag}.pkl', **kwargs)
    stds = torch.load(f'data/{config.name}/models/stds-{tag}.pkl', **kwargs)

    with torch.no_grad():
        output = observer(encoder(*Xs))
        return stds * output + means