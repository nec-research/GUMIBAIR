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

from gumibair_experiments.cross_cohort_exp import CrossCohortExperiment
from gumibair.dataset import FullMicrobiomeDataset
from gumibair.cond_norm import CondBatchNorm1d, CondBatchNorm1dExtend
from gumibair.cmvib import CMVIB
from run_experiments import import_data, write_output
from argparse import ArgumentParser
from sklearn import model_selection
import copy
import pandas as pd
import torch
import sys


def parse_args(description):
    parser = ArgumentParser(description=description)
    parser.add_argument('config_file',
                        help='Path to config file to be used')
    parser.add_argument('data_dir',
                        help='Location of the data to be used')
    parser.add_argument('-o', '--output-path',
                        type=str,
                        default='./',
                        help="Path to output location")
    parser.add_argument('-r', '--replicates',
                        type=int,
                        default=5,
                        help="Specifies the number of replicates that should be run per experiment.")
    parser.add_argument('-s', '--set-single',
                        default=False,
                        action='store_true',
                        help="""If set, the weight matrix of the CBN layers is set to an identity matrix.
                        This is supposed to represent the 'single' conditioning mode.""")
    parser.add_argument('-a', '--all-cohorts',
                        default=False,
                        action='store_true',
                        help="""If set, the weight matrix of the CBN layers is set to an identity matrix.
                        This is supposed to represent the 'single' conditioning mode.""")
    parser.add_argument('-bm', '--benchmark',
                        default=None,
                        type=str,
                        help="""Set either MVIB or Mean. The script will then run the same procedure as with GUMBAIR, 
                        but instead train MVIB or GUMIBAIR with mean-conditioning. Fine-tuning is then disabled.""")
    return parser.parse_args()

def init_experiment(fmd: FullMicrobiomeDataset, config: dict) -> CrossCohortExperiment:
    experiment = CrossCohortExperiment(
        fmd,
        config,
        mode='complete',
        benchmark=False,
        all_cohorts=True,
        monitor_test_auc=True,
        cross_validate=False,
    )

    return experiment

def set_cbn_to_single_layer(model: CMVIB):
    """
    Method to set the weight matrix of the weight and bias layer in the CondEncoder to an identity matrix.
    To represent the single-layer setting (mainly with the purpose of benchmarking single-layer against multi-layer)
    """
    # Set weights layer to identity matrix
    weights_layer = model.markers_encoder.encoder.encoder_layer_2.weight[0]
    weights_layer_matrix_dim = weights_layer.weight.shape

    # set the weights matrix of the weight layer to an identity matrix
    weights_layer.weight = torch.nn.Parameter(torch.eye(weights_layer_matrix_dim[0], weights_layer_matrix_dim[1]))
    weights_layer.weight.requires_grad = False

    # Set bias layer to matrix of zeros
    bias_layer = model.markers_encoder.encoder.encoder_layer_2.bias[0]
    bias_layer_matrix_dim = bias_layer.weight.shape

    # set the weights matrix of the weight layer to an identity matrix
    bias_layer.weight = torch.nn.Parameter(torch.eye(bias_layer_matrix_dim[0], bias_layer_matrix_dim[1]))
    bias_layer.weight.requires_grad = False

def pretrain_model(experiment: CrossCohortExperiment, config: dict, seed: int, heldout_idx: int, set_single=False, benchmark=None) -> CMVIB:
    """
    Function to pretrain the model on all cohorts but the heldout.
    """
    experiment.set_ids(seed, heldout_idx)

    if benchmark == "MVIB":
        mvib_condition = torch.zeros(experiment.dataset.conditions[0].shape)
        mvib_condition[0] = 1
        experiment.dataset.conditions[1:] = mvib_condition

    elif benchmark == "Mean":
        mean_condition = torch.tensor([1/len(experiment.dataset.cohorts) for _ in range(len(experiment.dataset.cohorts))])
        experiment.dataset.conditions[experiment.ids['test_ids']] = mean_condition

    if set_single:
        set_cbn_to_single_layer(experiment.cmvib)

    experiment.initialize_models(seed)

    if benchmark == 'MVIB':
        hpo_config = config['MVIB']
    else:
        hpo_config = config["CMVIB"] 
    state, trainer = experiment.run_training(seed, experiment.ids, experiment.cmvib, hpo_config)

    pretrained_model, _ =  trainer.load_checkpoint(state)
    return pretrained_model, state, trainer

def freeze_model(pretrained_model: CMVIB):
    """
    Function to freeze all layers in the pretrained model except the CondNorm layers.
    """
    # Freeze everything in the encoder but the conditioning layers
    for layer in pretrained_model.markers_encoder.encoder:
        if not isinstance(layer, CondBatchNorm1d) and not isinstance(layer, CondBatchNorm1dExtend):
            for params in layer.parameters():
                params.requires_grad = False

    # Freeze all layers of the decoder
    for layer in pretrained_model.multihead_decoder.decoder:
        for params in layer.parameters():
            params.requires_grad = False

def get_cond_train_ids(experiment, seed, val_size, all_cohorts=False):
    """
    Function that creates the list of ids for the training step after freezing the model.
    Should be either just the test cohort (When conditioning with the cohort index),
    or the train ids + a subset of the test cohort to fine tune the CBN parameters.
    :param experiment: Experiment object that is used.
    :param seed: int defining the random seed of the run.
    :param all_cohorts: bool that defines whether training cohorts should be included after pre-training or not.
    """
    if not all_cohorts:
        train_ids = experiment.ids['test_ids']
        y_train = experiment.ids['y_test']

        inner_train_ids, val_ids, y_inner_train, y_val = model_selection.train_test_split(
                train_ids,
                y_train,
                test_size=val_size,
                random_state=seed,
                stratify=y_train
            )
        
        cond_training_ids = {
            'inner_train_ids': inner_train_ids,
            'val_ids': val_ids,
            'y_inner_train': y_inner_train,
            'y_val': y_val,
            'test_ids': experiment.ids['test_ids'],
            'y_test': experiment.ids['y_test']
        }

    else:
        # split the test cohort into test_train and test_test
        test_ids = experiment.ids['test_ids']
        y_test = experiment.ids['y_test']

        test_train_ids, test_test_ids, y_test_train, y_test_test = model_selection.train_test_split(
                test_ids,
                y_test,
                test_size=0.3,
                random_state=seed,
                stratify=y_test
            )

        # concatenate test_train with the inner_train_ids --> training samples
        test_train_ids.extend(experiment.ids['inner_train_ids'])
        y_test_train.extend(experiment.ids['y_inner_train'])

        inner_train_ids, val_ids, y_inner_train, y_val = model_selection.train_test_split(
                test_train_ids,
                y_test_train,
                test_size=val_size, # should be defined either as a argparse arg or in the config for the future
                random_state=seed,
                stratify=y_test_train
        )

        # # add the original validation set to the final test samples
        # test_val_ids = test_test_ids.extend(experiment.ids['val_ids'])
        # y_test_val = y_test_test.extend(experiment.ids['y_val'])

        # concatenate the test_test and val ids --> test samples
        cond_training_ids = {
            'inner_train_ids': inner_train_ids,
            'val_ids': val_ids,
            'y_inner_train': y_inner_train,
            'y_val': y_val,
            'test_ids': test_test_ids,
            'y_test': y_test_test
        }
    
    return cond_training_ids

def train_frozen_model(experiment: CrossCohortExperiment, 
                       pretrained_model: CMVIB, 
                       config: dict, 
                       seed: int, 
                       heldout_idx: int, 
                       all_cohorts: bool =  False):
    """
    Function to train the frozen model on the heldout cohort.
    """
    cond_training_ids = get_cond_train_ids(experiment, seed, config['val_size'], all_cohorts=all_cohorts)

    rep_model = copy.deepcopy(pretrained_model)
    state, trainer = experiment.run_training(seed, cond_training_ids, rep_model, config['CMVIB'])
    prediction, test_gt = experiment.run_testing(state, trainer, cond_training_ids, heldout_idx)
  
    return prediction, test_gt

def main(args):
    if args.benchmark == "None":
        args.benchmark = None
    config, fmd = import_data(args.data_dir,args.config_file)

    fmd.subset_markers(cutoff=5)
    config['marker_dim'] = fmd.markers.shape[1]

    experiment = init_experiment(fmd, config)

    
    if args.benchmark:
        if args.benchmark == "MVIB":
            output_id = "MVIB"
        elif args.benchmark == "Mean":
            output_id = "GB_MEAN"
    else:
        output_id = "GB"

    output_fname = f"{args.output_path}/cross_c_complete_frozen_results_{output_id}_per_cohort.csv"

    for heldout_idx, cohort in enumerate(fmd.cohorts):
        scores = list()
        print(f"Current held-out cohort: {cohort}")
        for seed in range(args.replicates):
            print(f"Replicate {seed}:")
            print("Pretraining the model:")
            pretrained_model, pre_state, pre_trainer = pretrain_model(experiment, config, seed, heldout_idx, set_single=args.set_single, benchmark=args.benchmark)

            if not args.benchmark:
                print("Freezing the model:")
                freeze_model(pretrained_model)
                print("Training frozen model on the test set samples")
                prediction, test_gt = train_frozen_model(experiment,
                                                    pretrained_model, 
                                                    config, 
                                                    seed, 
                                                    heldout_idx, 
                                                    args.all_cohorts)
            else:
                prediction, test_gt = experiment.run_testing(pre_state, pre_trainer, experiment.ids, heldout_idx)
            
            condensed_scores = experiment.get_condensed_metrics(prediction, test_gt)
            condensed_scores['exp'] = seed
            condensed_scores['heldout_cohort'] = cohort
            print("Current Replicate auROC:")
            print(condensed_scores[condensed_scores['metrics'] == "auROC"])
            scores.append(condensed_scores)

        concatenated_scores = pd.concat(scores)

        if heldout_idx == 0:
            concatenated_scores.to_csv(output_fname, mode='a', index=False)
        else:
            concatenated_scores.to_csv(output_fname, mode='a', index=False, header=False)
        print(f"Results for cohort {cohort} are saved to {output_fname}")

if __name__ == "__main__":
    description="""
    Python script to train CMVIB on completely-heldout setting and freeze everything but the CondNorm Layers afterwards.
    The pretrained model is then trained on the heldout set, only updating the conditioning weights.
    Finally computes the AUC on the test set.
    """
    args = parse_args(description)
    main(args) 