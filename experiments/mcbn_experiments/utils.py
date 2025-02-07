# GUMIBAIR
#
#   Authors:  Moritz von Stetten (moritz@oncoimmunity.com)
#             Pierre Machart (pierre.machart@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

import numpy as np
import random
import yaml

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from mcbn.cmvib import CMVIB
from mcbn.trainer import Trainer
from ray import tune

from typing import Tuple
from statistics import mean
import pandas as pd

def load_config(config_path):
    """
    Function to parse configuration from a provided yaml file.
    Yaml should include HPO results as well as cohorts to be used for training.
    :param config_path: absolute path to the yaml file with the configuration
    :return: dictionary with concatenated configurations.
    """
    with open(config_path) as stream:
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


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def instantiate_cmvib(hp: dict, config: dict) -> CMVIB:
    """
    Function to instantiate a CMVIB object.
    :param config: dictionary with general configurations for the model.
    :hp: dictionarty with hyperparameters for the model.
    :return: instance of CMVIB class.
    """
    model = CMVIB(
        hp.get('n_latents'),
        config.get('abundance_dim'),
        config.get('marker_dim'),
        config.get('device'),
        config.get('cond_dim'),
        hp.get('hidden_encoder'),
        hp.get('hidden_encoder'),
        hp.get('hidden_decoder'),
        config.get('n_heads'),
        p_drop=hp.get('p_drop'),
        conditioning_mode=config.get('conditioning_mode', 'single')
    )
    return model

def instantiate_rf(config: dict) -> RandomForestClassifier:
    """
    Function to instantiate a RandomForestClassifier object.
    :param config: dictionary with configurations for the model.
    :return: instance of RandomForestClassifier class.
    """
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators'),
        criterion=config.get('criterion'),
        min_samples_leaf=config.get('min_samples_leaf'),
        max_features=config.get('max_features'),
        n_jobs=32
    )
    return model

def train_cmvib(dataset, ids, model, hp, config, **kwargs):
    """
    Function to train CMVIB, giving specific fmd objects for training and testing.
    :param seed: integer number of random seed
    :param dataset: FMD object with all cohorts
    :param ids: dictionary with lists of ids for training, testing and validation
    :param model: CMVIB instance initialized with a particular set of hyperparameters.
    :param hp: dict with hyperparameters to initialize the Trainer object.
    :param config: dictionary with additional configurations
    :return : tuple with test predictions, test ground truth and latent mean values
    """
    batch_size = config.get('batch_size')
    device = config.get('device')
    
    # create dataloaders
    train_sampler = SubsetRandomSampler(ids.get('inner_train_ids'))
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=config['drop_last_train'],
    )

    val_sampler = SubsetRandomSampler(ids.get('val_ids'))
    val_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        drop_last=config['drop_last_val'],
    )

    model.to(device)
    
    trainer = Trainer(
        model=model,
        epochs=config.get('max_epochs'),
        lr=hp.get('lr'),
        beta=hp.get('beta'),
        checkpoint_dir='.',
        monitor=config.get('monitor'),
        device=device,
        log=config.get('log'),
        log_dir=config.get('log_dir'),
        test_set=kwargs.get('test_set')
    )

    state = trainer.train(
        train_loader,
        val_loader,
        verbose=True
    )

    return state, trainer

def test_cmvib(state, dataset, ids, trainer, config):
    test_abundance, test_markers, test_gt, test_cond, test_heads = dataset[ids.get('test_ids')]

    device = config.get('device')

    test_abundance = test_abundance.to(device)
    test_markers = test_markers.to(device)
    test_gt = test_gt.to(device)
    test_cond = test_cond.to(device)
    test_heads = test_heads.to(device)

    model, _ = trainer.load_checkpoint(state)
    model.eval()
    model.to(device)
    prediction = model.classify((test_abundance, test_markers, test_heads, test_cond))
    mu, logvar = model.infer(test_abundance, test_markers, test_cond)

    return model, prediction, test_gt, mu, logvar

def hpo_run_cmvib(config, dataset, abundance_dim, marker_dim, device, cond_dim, n_heads, monitor, max_epochs, batch_size, gene_filter):
    """
    Method to train/validate (C)MVIB for HPO.
    :param seed: integer number of random seed
    :param dataset: FMD object with all cohorts
    :param ids: dictionary with lists of ids for training, testing and validation
    :param model: CMVIB instance initialized with a particular set of hyperparameters.
    :param hp: dict with hyperparameters to initialize the Trainer object.
    :param config: dictionary with additional configurations
    :return : tuple with test predictions, test ground truth and latent mean values
    """
    prediction_stack = np.array([])
    val_gt_stack = np.array([])

    if gene_filter is not 'None':
        dataset.subset_markers(cutoff=int(gene_filter))

    for idx, cohort in enumerate(dataset.cohorts):
        val_ids = dataset.cohorts_2_patients[idx]['patients']
        y_val = dataset.cohorts_2_patients[idx]['labels']
        inner_train_ids = [i for i in dataset.patients_ids if not i in val_ids]
        y_inner_train = [l for i, l in enumerate(dataset.labels) if i in inner_train_ids]

        ids = {
            'inner_train_ids': inner_train_ids,
            'val_ids': val_ids,
            'test_ids': val_ids,
            'y_inner_train': y_inner_train,
            'y_val': y_val,
            'y_test': y_val
        }

        model = torch.nn.DataParallel(CMVIB(
            n_latents=config['n_latents'],
            abundance_dim=abundance_dim,
            marker_dim=marker_dim,
            device=device,
            cond_dim=cond_dim,
            hidden_abundance=config['hidden_encoder'],
            hidden_markers=config['hidden_encoder'],
            hidden_decoder=config['hidden_decoder'],
            n_heads=n_heads,
            p_drop=config['p_drop'],
            conditioning_mode=config['conditioning_mode']
        ))

        train_sampler = SubsetRandomSampler(ids.get('inner_train_ids'))
        train_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
        )

        val_sampler = SubsetRandomSampler(ids.get('val_ids'))
        val_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=val_sampler,
            batch_size=batch_size,
        )

        trainer = Trainer(
            model=model,
            epochs=max_epochs,
            lr=config.get('lr'),
            beta=config.get('beta'),
            checkpoint_dir='.',
            monitor=monitor,
            device=device,
        )

        state = trainer.train(
            train_loader,
            val_loader,
            verbose=True
        )

        _, prediction, val_gt, __, ___ = test_cmvib(state, dataset, ids, trainer, config)
        if type(prediction) == torch.Tensor:
            prediction = prediction.cpu().detach().numpy().squeeze().tolist()
        if type(val_gt) == torch.Tensor:
            val_gt = val_gt.cpu().detach().numpy().squeeze().tolist()

        prediction_stack = np.concatenate(
            (prediction_stack, prediction),
                axis=0
        )

        val_gt_stack = np.concatenate(
            (val_gt_stack, val_gt),
                axis=0
        )
        
    if gene_filter is not 'None':
        dataset.markers = dataset.original_markers

    val_score = roc_auc_score(val_gt_stack, prediction_stack)

    tune.report(best_val_score=val_score)

def run_rf(dataset, train_ids, test_ids, y_train, y_test, seed, classifier):
    """
    Function to train and test Random Forest on a specific set of ids.
    :param dataset: FullMicrobiomeDataset object containing the input data.
    :param train_ids: list with train_ids
    :param test_ids: list with test_ids
    :param y_train: list with train ground truths
    :param y_test: list with test ground truths
    :param seed: integer number of random seed
    :param classifier: RandomForestClassifier instance initialized with a particular set of hyperparameters
    :return: test ground truth and predictions.
    """
    x_train = dataset.markers[train_ids].cpu().numpy()
    x_test = dataset.markers[test_ids].cpu().numpy()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier.set_params(random_state=seed)
    classifier.fit(x_train, y_train)

    y_true, y_pred = y_test, classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)[:, 1]

    return y_true, y_pred
    
def hpo_run_rf(config, dataset, train_ids, val_ids, y_train, y_val, seed):
    """
    Function for training and testing RF during HPO.
    Reports the auc of the model with the current set of parameters to ray tune.
    :param dataset: FullMicrobiomeDataset object containing the input data.
    :param train_ids: list with train_ids
    :param val_ids: list with val ids
    :param y_train: list with train ground truths
    :param y_val: list with val ground truths
    """
    x_train = dataset.markers[train_ids].cpu().numpy()
    x_val = dataset.markers[val_ids].cpu().numpy()

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    classifier = instantiate_rf(seed, config)

    classifier.fit(x_train, y_train)

    y_true, y_pred = y_val, classifier.predict(x_val)
    y_prob = classifier.predict_proba(x_val)[:, 1]

    val_auc = roc_auc_score(y_true, y_prob)
    tune.report(val_auc=val_auc)

def partial_hold_out_split(seed, dataset, lo_cohort, test_size, all_cohorts=True, stratify=True):
    """
    Function to create a train test split with only one cohort for testing
    :param seed: integer number of random seed
    :param dataset: FMD object with all cohorts
    :param lo_cohort: integer id of the cohort that should be left out for testing
    :param test_size: proportion of lo_cohort samples that should be used for testing
    """
    train_ids = list()
    y_train = list()

    for cohort, data in dataset.cohorts_2_patients.items():
        patient_ids = data['patients']
        labels = data['labels']
        if not cohort == lo_cohort:
            train_ids.extend(patient_ids)
            y_train.extend(labels)
        else:
            if stratify == True:
                stratify_label = labels
            else:
                stratify_label = None
                
            lo_train_ids, test_ids, y_lo_train, y_test = model_selection.train_test_split(
                patient_ids,
                labels,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_label
            )
            if all_cohorts == False:
                return lo_train_ids, test_ids, y_lo_train, y_test

            train_ids.extend(lo_train_ids)
            y_train.extend(y_lo_train)

    return train_ids, test_ids, y_train, y_test

def complete_hold_out_split(dataset, cohort):
    """
    Function to hold out one cohort completely for testing.
    All other cohorts should be used for training.
    :param dataset: FMD object with all cohorts
    :param cohort: Name of cohort to hold out for training
    """
    train_ids = list()
    y_train = list()

    test_ids = list()
    y_test = list()
    for c, _ in enumerate(dataset.cohorts_2_patients):
        if not c == cohort:
            train_ids.extend(dataset.cohorts_2_patients[c]['patients'])
            y_train.extend(dataset.cohorts_2_patients[c]['labels'])
        else:
            test_ids.extend(dataset.cohorts_2_patients[c]['patients'])
            y_test.extend(dataset.cohorts_2_patients[c]['labels'])

    return train_ids, test_ids, y_train, y_test
