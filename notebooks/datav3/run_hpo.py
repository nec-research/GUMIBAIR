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

from ray import tune
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
from run_experiments import import_data
from argparse import ArgumentParser
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from gumibair.hpo import hpo_train
from gumibair.dataset import FullMicrobiomeDataset
from mcbn_experiments.utils import hpo_run_rf, hpo_run_cmvib

from typing import Tuple
import os
import yaml

from datetime import date

def parse_args(description):
    parser = ArgumentParser(description=description)
    parser.add_argument('config_file',
                        help='Path to config file to be used')
    parser.add_argument('model',
                        choices=['CMVIB','MVIB', 'RF'],
                        help="Argument defining what model should be tuned. Either 'CMVIB' or 'RF'")
    parser.add_argument('mode',
                        choices=['heldout_validation', 'default'],
                        help="""Argument to decide how to train for each trial. 
                        'heldout_validation' iterates through all cohorts and makes each the validation set once.
                        'default' just does a random split into train/validation and test set.""")
    return parser.parse_args()

def prep_data(data: FullMicrobiomeDataset,
              train_ids: list, 
              val_ids: list, 
              config: dict) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    """
    Function to create the train and validation samplers based on the FMD object.
    :param train_ids: list with train ids
    :param val_ids: list with test_ids
    :param config: dict with hpo-related configs
    :return: tuple with two SubsetRandomSampler objects
    """
    train_sampler = SubsetRandomSampler(train_ids)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=config['batch_size'],
        sampler=train_sampler
    )

    val_sampler = SubsetRandomSampler(val_ids)
    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=config['batch_size'],
        sampler=val_sampler
    )

    return train_loader, val_loader

def create_search_space(model: str, config: dict) -> dict:
    """
    Function to create search space for CMVIB or RF.
    :param model: str defining what model should be used for tuning.
    :param config: dictionary with configs for the search space.
    :return: dictionary with model-specific search space
    """
    if model in ['CMVIB', 'MVIB']:
        search_space = { 
            'n_latents': tune.randint(config['n_latents'][0], config['n_latents'][1]),
            'lr': tune.loguniform(config['lr'][0], config['lr'][1]),
            'beta': tune.loguniform(config['beta'][0], config['beta'][1]),
            'p_drop': tune.uniform(config['p_drop'][0], config['p_drop'][1]),
            'hidden_encoder': tune.choice(config['hidden_encoder']),
            'hidden_decoder': tune.choice(config['hidden_decoder']),
            'conditioning_mode': tune.choice(config['conditioning_mode']),
            'gene_filter': tune.choice(config['gene_filter']),
        }
    elif model  == 'RF':
        search_space = {
            'n_estimators': tune.randint(config['n_estimators'][0], config['n_estimators'][-1]),
            'max_features': tune.choice(config['max_features']),
            'min_samples_leaf': tune.randint(config['min_samples_leaf'][0], config['min_samples_leaf'][-1]),
            'criterion': tune.choice(config['criterion']),
            'gene_filter': tune.choice(config['gene_filter'])
        }
    return search_space


def optimize_hp(opt_config: dict, resources_per_trial: dict, search_space: dict, trainable, metric: str, **trainable_args) -> ExperimentAnalysis:
    """
    Function that runs the HPO using HyperOptSearch.
    :param opt_config: dict with configurations for the HPO.
    :param resources_per_trial: dict with resource specs for gpu und cpu.
    :param search_space: dict with model-specific search space.
    :param trainable: function to be used for training during HPO.
    :param metric: str defining the mode of the search (either 'min' or 'max').
    :param trainable_args: dict with kwargs for the trainable function.
    :return: ExperimentAnalysis object.
    """
    search_alg = HyperOptSearch(
        metric=metric,
        mode='max',
    )

    search_alg.set_max_concurrency(opt_config['max_concurrent_trials'])

    analysis = tune.run(
        tune.with_parameters(
            trainable,
            **trainable_args,
        ),
        config=search_space,
        num_samples=opt_config['num_samples'],
        resources_per_trial=resources_per_trial,
        search_alg=search_alg,
    )

    return analysis

def write_output(model: str, best_config: dict):
    """
    Function to save the best config to a file.
    :param model: str defining the model used during HPO.
    :param best_config: dict with best configuration
    """
    today = str(date.today())
    today = today.replace("-", "_")

    fname = f"hpo_results_{today}_{model}.yaml"
    if os.path.isfile(fname):
        fname = fname.replace(today, f"{today}_2")

    with open(fname, 'w') as yaml_file:
        yaml.dump(best_config, yaml_file)
    
def main(args):
    config, data = import_data("/mnt/container-nle-microbiome/datav3/", args.config_file)

    if args.model == "MVIB":
        data.conditions[1:] = data.conditions[0]

    opt_config = {
        'abundance_dim':len(data[0][0]),
        'marker_dim':len(data[0][1]),
        'cond_dim':len(data[0][3]),
    }

    opt_config.update(config)

    search_space = create_search_space(args.model, opt_config)

    if args.model in ['CMVIB', 'MVIB']:
        metric='best_val_score'
        if args.mode == "heldout_validation":
            trainable = hpo_run_cmvib
            trainable_args = {
                'dataset': data,
                'abundance_dim': opt_config['abundance_dim'],
                'marker_dim': opt_config['marker_dim'],
                'device': opt_config['device'],
                'cond_dim': opt_config['cond_dim'],
                'n_heads': opt_config['n_heads'],
                'monitor': opt_config['monitor'],
                'max_epochs': opt_config['max_epochs'],
                'batch_size': opt_config['batch_size']
            }
        elif args.mode == "default":
            train_ids, val_ids, y_train, y_val = data.per_cohort_train_test_split(config['test_size'], config['hpo_seed'])
            train_loader, val_loader = prep_data(data, train_ids, val_ids, opt_config)
            trainable = hpo_train
            trainable_args = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'abundance_dim': opt_config['abundance_dim'],
                'marker_dim': opt_config['marker_dim'],
                'device': opt_config['device'],
                'cond_dim': opt_config['cond_dim'],
                'n_heads': opt_config['n_heads'],
                'monitor': opt_config['monitor'],
                'max_epochs': opt_config['max_epochs'],
            }

        resources_per_trial = {'gpu':0.25, 'cpu':1}

    elif args.model == "RF":
        metric='val_auc'
        trainable = hpo_run_rf
        trainable_args = {
            'dataset': data,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'y_train': y_train,
            'y_val':  y_val,
            'seed': opt_config['hpo_seed']
        }

        resources_per_trial = {'gpu':0, 'cpu':1}

    results = optimize_hp( 
        opt_config,
        resources_per_trial,
        search_space,
        trainable,
        metric,
        **trainable_args
    )

    best_config = results.get_best_config(metric=metric, mode='max')
    write_output(args.model, best_config)
    

if __name__ == "__main__":
    description="""
    Python script to run Hyperparameter optimization.
    """
    args = parse_args(description)
    main(args)