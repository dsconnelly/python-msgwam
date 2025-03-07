import matplotlib.pyplot as plt
import torch

from msgwam import config

from ... import hyperparameters as hp
from ..architectures import Encoder, Observer
from .loops import _load_datasets

def plot_encoder() -> None:
    """
    
    """

    model_dir = f'data/{config.name}/models'
    loader_tr, loader_ev = _load_datasets(1, 'validation')
    enc_state = torch.load(f'{model_dir}/encoder-{hp.task_id}.pkl')
    obs_state = torch.load(f'{model_dir}/observer-{hp.task_id}.pkl')

    encoder, observer = Encoder(), Observer()
    encoder.load_state_dict(enc_state)
    observer.load_state_dict(obs_state)

    means = torch.load(f'data/{config.name}/models/flux-means.pkl')
    stds = torch.load(f'data/{config.name}/models/flux-stds.pkl')

    # encoder.eval()
    # observer.eval()

    for *Xs, Y in loader_tr:
        with torch.no_grad():
            output = observer(encoder(*Xs))
            break

    output = output * stds + means

    fig, axes = plt.subplots(ncols=4)
    fig.set_size_inches(12, 4.5)

    kdx = torch.randperm(Y.shape[0])[:4]
    z = torch.linspace(config.z_min, config.z_max, config.n_grid) / 1000

    for k, ax in zip(kdx, axes):
        ax.plot(Y[k], z, color='k')
        ax.plot(output[k], z, color='tab:red')

    plt.tight_layout()
    plt.show()