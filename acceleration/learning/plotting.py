from .training import get_loader

def plot_training_fluxes() -> None:
    """
    
    """

    for *_, Y in get_loader(1, ['tr', 'va']):
        break

    print(Y.shape)