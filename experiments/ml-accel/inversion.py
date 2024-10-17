import os

import torch

from torch.optim import Adam

from msgwam import config
from msgwam.mean import MeanState
from msgwam.utils import put, shapiro_filter

import hyperparameters as params

from architectures import Surrogate
from preprocessing import PACKETS_PER_BATCH
from utils import DATA_DIR, load_data, integrate_batch, nondimensionalize

def invert_surrogate(
    n_steps: int=100,
    n_print: int=10,
    conservative: bool=True
) -> None:
    """
    Inverts a coarse Surrogate to find ray volumes that should produce online
    flux or drag profiles as similar as possible to those associated with the
    underlying fine ray volumes.
    """

    model = _load_model()
    u, Y_all, Z_fine = load_data('fine')
    *_, Z_coarse = load_data('coarse')

    grid = MeanState().z_faces / 1000
    keep = (grid > 25) & (grid < 68)

    errors = _loss_func(
        Z_fine[:, keep],
        Z_coarse[:, keep],
        reduce=False,
        mode='max'
    )
    
    idx = torch.nonzero(errors > 0.3).flatten()
    mask = torch.isin(torch.arange(u.shape[0]), idx)
    mask = mask.reshape(-1, PACKETS_PER_BATCH)

    print(f'Suggesting changes to {len(idx)} samples')
    
    u, Y, Z_fine = u[idx], Y_all[idx], Z_fine[idx]
    signs = torch.sign(Y[:, 2])[:, None]

    n_z = config.n_grid - 1
    stacked = model._preprocess(u, Y)
    stacked = model._standardize(stacked)
    u, Y_hat = stacked[:, :n_z], stacked[:, n_z:]

    means, stds = model.means[n_z:], model.stds[n_z:]
    log_M = torch.log(abs(Y[:, 2]) * Y[:, 7] * Y[:, 8])
    mu_sum = means.sum() - means[1]
    s_1, _, s_2, s_3 = stds

    Y_hat.requires_grad = True
    optimizer = Adam([Y_hat], lr=1e-1)

    for n_step in range(1, n_steps + 1):
        optimizer.zero_grad()
        output = _forward(u, Y_hat, signs, model)
        loss = _loss_func(Z_fine[:, keep], output[:, keep])

        loss.backward()
        optimizer.step()

        if conservative:
            with torch.no_grad():
                _k, _, _dm, _ = Y_hat.T
                _dens = (log_M - s_1 * _k - s_2 * _dm - mu_sum) / s_3
                Y_hat[:, -1] = _dens

        if n_step % n_print == 0:
            print(f'step {n_step}: loss = {loss.item():.6g}')

    with torch.no_grad():
        output = _forward(u, Y_hat, signs, model)

    Y_hat = torch.exp(Y_hat.detach() * stds + means)
    Y_hat = Y_hat * torch.sign(Y[:, model.idx_in])
    Y_hat = put(Y.T, model.idx_in, Y_hat.T).T
    Y_hat = put(Y_all, idx, Y_hat)

    Y_hat = Y_hat.reshape(-1, PACKETS_PER_BATCH, 9).transpose(1, 2)
    torch.save(mask, f'{DATA_DIR}/mask-candidates.pkl')
    torch.save(Y_hat, f'{DATA_DIR}/Y-hat.pkl')

    return Z_fine, Z_coarse[idx], output

def validate_inversion() -> None:
    """
    Take the candidate ray volumes suggested by the inversion and integrate the
    actual solver with them. Then, adjust the mask to include only those that
    actually lead to better online behavior.
    """

    mask = torch.load(f'{DATA_DIR}/mask-candidates.pkl')
    wind = torch.load(f'{DATA_DIR}/wind.pkl').float()
    Y_hat = torch.load(f'{DATA_DIR}/Y-hat.pkl')

    *_, Z_fine = load_data('fine')
    *_, Z_coarse = load_data('coarse')

    Z_fine = Z_fine.reshape(-1, PACKETS_PER_BATCH, Z_fine.shape[-1])
    Z_coarse = Z_coarse.reshape(-1, PACKETS_PER_BATCH, Z_coarse.shape[-1])
    errors_coarse = _loss_func(Z_fine, Z_coarse, reduce=False)
    serrors_coarse = _loss_func(Z_fine, Z_coarse, reduce=False, mode='sum')

    vname = 'SLURM_ARRAY_TASK_COUNT'
    n_tasks = int(os.environ.get(vname, 1))
    # n_per_task = mask.shape[0] // n_tasks

    # start = params.task_id * n_per_task
    # end = (params.task_id + 1) * n_per_task
    # print(f'Task {params.task_id} integrating {start} to {end}.')

    for task_id in range(25):
        Z_hats = torch.load(f'{DATA_DIR}/Z-hats-{task_id}.pkl')
        start = task_id * 40
        end = (task_id + 1) * 40

        for i, Z_hat in zip(range(start, end), Z_hats):
            if mask[i].sum() == 0:
                continue

            if i <= start:
                continue

            rays = Y_hat[i, :, mask[i]].reshape(9, -1)
            # Z_hat = integrate_batch(wind[i], rays, 1, None).sum(1) / config.n_t_max
            # Z_hat = nondimensionalize(rays.T, Z_hat)
            # Z_hats.append(Z_hat)

            errors_hat = _loss_func(Z_fine[i, mask[i]], Z_hat, reduce=False)
            discard = errors_hat > errors_coarse[i, mask[i]]

            serrors_hat = _loss_func(Z_fine[i, mask[i]], Z_hat, reduce=False, mode='sum')
            discard = discard & (serrors_hat > serrors_coarse[i, mask[i]])

            # mask[i, torch.nonzero(mask[i])[discard, 0]] = False
            print(f'discarded {discard.sum()} out of {rays.shape[1]}')
            return Z_fine[i, mask[i]], Z_coarse[i, mask[i]], Z_hat, discard

        torch.save(mask[start:end], f'{DATA_DIR}/mask-accepted-{task_id}.pkl')

    # Z_hats = []
    # for i in range(start, end):
    #     if mask[i].sum() == 0:
    #         continue

    #     rays = Y_hat[i, :, mask[i]].reshape(9, -1)
    #     Z_hat = integrate_batch(wind[i], rays, 1, None).sum(1) / config.n_t_max
    #     Z_hat = nondimensionalize(rays.T, Z_hat)
    #     Z_hats.append(Z_hat)

    #     errors_hat = _loss_func(Z_fine[i, mask[i]], Z_hat, reduce=False)
    #     discard = errors_hat > errors_coarse[i, mask[i]]

    #     mask[i, torch.nonzero(mask[i])[discard, 0]] = False
    #     print(f'discarded {discard.sum()} out of {rays.shape[1]}')

    # suffix = '' if n_tasks == 1 else f'-{params.task_id}'
    # torch.save(mask[start:end], f'{DATA_DIR}/mask-accepted{suffix}.pkl')
    # torch.save(Z_hats, f'{DATA_DIR}/Z-hats{suffix}.pkl')

def finalize_validation() -> None:
    """
    
    """

    ks, datas = [], []
    for fname in os.listdir(DATA_DIR):
        if not fname.startswith('mask-accepted-'):
            continue

        ks.append(int(fname.split('.')[0].split('-')[-1]))
        datas.append(torch.load(f'{DATA_DIR}/{fname}'))

    idx = torch.argsort(torch.as_tensor(ks))
    mask = torch.vstack([datas[i] for i in idx.tolist()])
    torch.save(mask, f'{DATA_DIR}/mask-accepted.pkl')

    candidates = torch.load(f'{DATA_DIR}/mask-candidates.pkl')
    percent = 100 * mask.sum() / candidates.sum()
    print(f'Accepted {percent:.4f}% of candidates.')
    print(mask.sum())

def _forward(
    u: torch.Tensor,
    Y: torch.Tensor,
    signs: torch.Tensor,
    model: Surrogate
) -> torch.Tensor:
    """
    Apply the Surrogate to already-preprocessed input data. The inversion uses
    this function instead of `Surrogate.forward` because it should be easier to
    navigate the loss landscape of the nondimensional data.

    Parameters
    ----------
    u
        Two-dimensional tensor of standardized zonal wind data.
    Y
        Two-dimensional tensor of preprocessed and standardized ray volume data.
    signs
        Signs of the zonal wavenumbers of the packets contained in `Y`. Must be
        passed separately since `Y` is assumed to be preprocessed already.
    model
        Surrogate model to apply.

    Returns
    -------
    torch.Tensor
        Nondimensionalized flux profiles of the correct sign.

    """

    stacked = torch.hstack((u, Y))
    output = model._predict(stacked)
    output[:, 1:-1] = shapiro_filter(output.T).T

    return signs * output

def _load_model() -> Surrogate:
    """
    Load a trained `Surrogate` model. Note that this function changes the
    hyperparameter settings so that the trained model can be loaded.
    
    Returns
    -------
    Surrogate
        Best-performing coarse `Surrogate`, in evaluation mode.

    """

    path = f'{DATA_DIR}/surrogate-coarse/state-best-r0.pkl'
    state = torch.load(path)['model']

    model = Surrogate()
    model.load_state_dict(state)
    model.eval()

    return model

def _loss_func(
    target: torch.Tensor,
    output: torch.Tensor,
    reduce: bool=True,
    mode: str='mae'
) -> torch.Tensor:
    """
    
    """

    if mode == 'mse':
        loss = ((output - target) ** 2).mean(dim=-1)

    elif mode == 'mae':
        loss = abs(output - target).mean(dim=-1)
    
    elif mode == 'sum':
        loss = abs((output - target).sum(dim=-1))

    elif mode == 'max':
        loss = abs(output - target).max(dim=-1)[0]

    if reduce:
        return loss.mean()
    
    return loss
