import torch

from architectures import CoarseNet
from utils import integrate_batches

N_EPOCHS = 10

def train_network() -> None:
    """Train a CoarseNet instance."""

    wind = torch.load('data/coarsenet/wind.pkl')
    X = torch.load('data/coarsenet/spectra.pkl')
    Z = torch.load('data/coarsenet/targets.pkl')

    model = CoarseNet()
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=0.005)

    for n_epoch in range(N_EPOCHS):
        print('=' * 8, f'epoch {n_epoch + 1}', '=' * 8)

        for i in range(X.shape[0]):
            optim.zero_grad()
            output = model(X[i])
            grad = _grad_loss(X[i], wind[i], output.detach(), Z[i])
            output.backward(grad)
            optim.step()

            with torch.no_grad():
                model.eval()
                output = model(X[i])
                loss = _loss(X[i], wind[i], output, Z[i]).sum().item()
                print(f'    batch {i + 1}: loss = {loss:.4g}')
                model.train()

    torch.save(model.state_dict(), 'data/coarsenet/model.pkl')

def _grad_loss(
    X: torch.Tensor,
    wind: torch.Tensor,
    output: torch.Tensor,
    Z: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the online loss function with respect to the neural
    network-predicted replacement ray volume properties.

    Parameters 
    ----------
    model
        Neural network (only used for `build_spectrum`).
    wind
        Fixed zonal and meridional wind profiles to use during integration.
    output
        Replacement ray volume properties as returned by `model`. Make sure that
        this tensor is detached before being passed in.
    Z
        Correct online pseudomomentum fluxes for each packet.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as `output` containing the gradient of the
        loss function with respect to each replacement ray property.
    
    """

    grad = torch.zeros(output.shape)
    for k in range(output.shape[0]):
        dprop = torch.zeros(output.shape)
        dprop[k] = 0.1 * output[k]

        L_plus = _loss(X, wind, output + dprop, Z)
        L_minus = _loss(X, wind, output - dprop, Z)

        grad[k] = (L_plus - L_minus) / (2 * dprop[k])

    return grad

def _loss(
    X: torch.Tensor,
    wind: torch.Tensor,
    output: torch.Tensor,
    Z: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the squared error between the pseudomomentum fluxes associated
    with the original wave packet and those from the replacement ray volume,
    summed over the duration of the integration.

    Parameters
    ----------
    X
        Original wave packet properties.
    wind
        Fixed zonal and meridional wind profiles to use during integration.
    output
        Replacement ray volume properties as returned by a `CoarseNet`.
    Z
        Correct online pseudomomentum fluxes for each wave packet.

    Returns
    -------
    torch.Tensor
        Tensor giving the squared error for each wave packet.

    """
    
    spectrum = CoarseNet.build_spectrum(X, output)
    Z_hat = integrate_batches(wind, spectrum, 1).mean(dim=1)

    return ((Z - Z_hat) ** 2).sum(dim=1)

