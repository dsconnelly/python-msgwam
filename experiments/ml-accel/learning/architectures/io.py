from os import listdir

import numpy as np
import torch

from torch.optim import Adam

from msgwam import config

from .. import hyperparameters as hp

from .base import SourceNet

def get_model_dir(target_type: str) -> str:
    """
    Return the path to the directory containing models with the given targets.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.

    Returns
    -------
    str
        Path to the appropriate directory.

    """

    if target_type.startswith(('flux', 'proxies')):
        target_type, grain = target_type.split('-')

    cls_name = _get_class_name(target_type).lower()
    return f'data/{config.name}/{cls_name}-{grain}'

def load_model(
    target_type: str,
    eval_type: str,
    restart: bool
) -> tuple[SourceNet, Adam]:
    """
    Load a model and an associated optimizer. Can be used to initialize a new
    model or to load a trained model from disk. If loading a model trained on
    the full training and validation sets, the `task_id` will be changed so that
    the saved weights are compatible with the model structure.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.
    eval_type
        Evaluation dataset specifier, as passed to `train_network`. If `test`,
        the hyperparameter `task_id` may be modified.
    restart
        Whether to load saved state from disk.

    Returns
    -------
    SourceNet
        Requested subclass instance, with loaded state if necessary.
    Adam
        Associated optimizer, with loaded state if necessary.

    """

    if restart:
        model_dir = get_model_dir(target_type)
        tag = 'best' if eval_type == 'test' else hp.task_id
        state = torch.load(f'{model_dir}/state-{tag}.pkl')

        if eval_type == 'test':
            hp.load(hp.grid_path, state['task_id'])

    elif eval_type == 'test':
        task_id = _get_best_task_id(target_type)
        print(f'Selecting hyperparameter configuration {task_id}:')
        hp.load(hp.grid_path, task_id, verbose=True)
        print()

    cls_name = _get_class_name(target_type)
    model = SourceNet.from_name(cls_name.capitalize())
    weight_decay = hp.weight_decay * hp.learning_rate

    optimizer = Adam(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=weight_decay
    )

    if restart:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    return model, optimizer

def _get_best_task_id(target_type: str) -> int:
    """
    Get the task ID of the training run with the lowest validation score by
    reading the log files.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.
    
    Returns
    -------
    int
        Task ID of the best hyperparameter configuration.

    """

    best_id = None
    best_score = np.inf

    log_dir = 'logs/ml-accel'
    _, grain = target_type.split('-')

    for fname in listdir(log_dir):
        if not fname.startswith('train-'):
            continue

        if not grain in fname:
            continue

        with open(f'{log_dir}/{fname}') as f:
            for line in f:
                if not line.startswith('loss_ev'):
                    continue

                score = float(line.strip().split(' = ')[1])

        if score < best_score:
            best_id = int(fname.split('.')[0].split('-')[-1])
            best_score = score

    return best_id

def _get_class_name(target_type: str) -> str:
    """
    Return the name of the `SourceNet` subclass that can be trained to learn
    targets with the given specifier.

    Parameters
    ----------
    target_type
        Target specifier, as passed to `train_network`.

    Returns
    -------
    str
        Name of the appropriate `SourceNet` subclass.

    """

    if target_type.startswith(('flux', 'proxies')):
        return 'Surrogate'

    raise ValueError(f'Unknown target type: {target_type}')