import os
import sys
import yaml

# For the testing, the version of gumibair in the local codebase should be used instead of the installed package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

import pytest
from gumibair.dataset import FullMicrobiomeDataset
from gumibair.cmvib import CMVIB
from gumibair.trainer import Trainer

import torch
from torch.utils.data.sampler import SubsetRandomSampler


@pytest.fixture(scope='session')
def config():
    """
    Configuration used for all the unittests.
    """
    test_session_config=f"{os.path.dirname(__file__)}/test_config.yaml"

    with open(test_session_config) as stream:
        try:
            plain_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    hpo_config = plain_config.get('hpo')
    search_space = plain_config.get('search_space')
    general_config = plain_config.get('general_config')

    concat_config = dict()
    for config in [hpo_config, search_space, general_config]:
        if config:
            concat_config.update(config)

    return concat_config

@pytest.fixture(scope='session')
def model_input(config):
    """
    Function to instanciate an example FullMicrobiomeDataset object and
    create the suitable input for the unit tests.
    """
    path = config['path']
    device = torch.device('cpu')

    dataset = FullMicrobiomeDataset(
        path,
        device,
        data='joint',
        cohorts=config['cohorts_to_disease'].keys(),
        cohorts_to_disease=config['cohorts_to_disease'],
        conditioning_type=config['conditioning_type']
    )

    X = dataset[list(range(len(dataset)))] # abundance, markers, heads and conditions

    return dataset, X, device

@pytest.fixture(scope='session')
def train_test_split(config, model_input):
    dataset, _, __ = model_input

    seed=42
    inner_train_ids, val_ids, test_ids, y_inner_train, y_val, y_test = dataset.per_cohort_train_val_test_split(
        test_size=0.2, val_size=0.15, random_state=seed, test_stratify=False
    )

    train_sampler = SubsetRandomSampler(inner_train_ids)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=config['batch_size']
    )

    val_sampler = SubsetRandomSampler(val_ids)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=val_sampler,
        batch_size=config['batch_size']
    )

    loaders = (train_loader, val_loader)
    ids = (inner_train_ids, val_ids, test_ids)
    labels = (y_inner_train, y_val, y_test)

    return loaders, ids, labels

@pytest.fixture(scope='session')
def model(config, model_input):
    """
    Fixture returning an instance of CMVIB
    """
    dataset, _, device = model_input

    model = CMVIB(
        config['n_latents'],
        len(dataset[0][0]),
        len(dataset[0][1]),
        device,
        len(dataset[0][3]),
        config['hidden_encoder'],
        config['hidden_encoder'],
        config['hidden_decoder'],
        1,
        run_mode=config['mode']
    )

    model.to(device)

    return model

@pytest.fixture(scope='session')
def trainer(config, model):
    trainer = Trainer(
        model=model,
        epochs=config['epochs'],
        lr=config['lr'],
        beta=config['beta'],
        checkpoint_dir='.',
        monitor=config['monitor'],
        device=config['device']
    )

    return trainer

@pytest.fixture(scope='session')
def state(trainer, train_test_split):
    train_loader, val_loader = train_test_split[0]

    state = trainer.train(
        train_loader,
        val_loader,
        verbose=False
    )

    return state