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

from gumibair.dataset import FullMicrobiomeDataset
from gumibair.trainer import Trainer
from gumibair.utils import get_scores, per_cohort_metrics
from gumibair_experiments.utils import set_seed, instantiate_cmvib, instantiate_rf, train_cmvib, test_cmvib, run_rf
import torch
import pandas as pd
from timeit import default_timer as timer

class _BaseExperiment:
    """ Base class for (C)MVIB-based experiments."""
    def __init__(self, 
                 dataset:FullMicrobiomeDataset, 
                 config: dict, 
                 replicates: int = 5,
                 benchmark: bool = True,
                 monitor_test_auc: bool = False):
        """
        :param dataset: FullMicroBiomeDataset object linked to the experiment
        :param config: dictionary with basic configurations from HPO etc.
        :param replicates: int defining how often the experiment should be executed.
        """
        self.dataset = dataset
        self.replicates = replicates
        self.config = config
        self.benchmark = benchmark
        self.ids = {}
        self.monitor_test_auc = monitor_test_auc

    def initialize_models(self, seed: int):
        """
        Method to initialize all models for  this experiment with the particular hyperparameters.
        :param seed: int with random seed used for current replicate.
        """
        set_seed(seed)

        cmvib_hp = self.config['CMVIB']
        mvib_hp = self.config['MVIB']
        rf_hp = self.config['RF']

        self.cmvib = instantiate_cmvib(cmvib_hp, self.config)
        self.mvib = instantiate_cmvib(mvib_hp, self.config)
        self.rf = instantiate_rf(rf_hp)

    def check_last_batch(self, id_list: list, key: str):
        """
        Method to check wether there is a residual sample with the used batch size.
        This will cause issues with teh BatchNorm (not possible with batches containing only one sample).
        Therefore, if there is one batch with only one sample, the drop_last argument of the dataloaders should be set to true.
        :param id_list: list with the ids to be checked.
        :param key: str with the key to the boolean in the config that should be set.
        """
        if len(id_list) % self.config['batch_size'] == 1:
            self.config[key] = True
        else:
            self.config[key] = False

    def get_condensed_metrics(self, prediction , test_gt):
        """
        Method getting the different performance scores.
        Uses the get_scores() method from gumibair.utils.
        :param prediction: tensor with predictions on test samples.
        :param test_gt: tensor with gt labels from test set.
        :return scores: pd.DataFrame with the different scores.
        """
        if type(prediction) == torch.Tensor:
            prediction = prediction.cpu().detach().numpy().squeeze().tolist()

        if type(test_gt) == torch.Tensor:
            test_gt = test_gt.cpu().detach().numpy().squeeze().tolist()

        results_df = pd.DataFrame(data={
            'predictions':prediction,
            'gt':test_gt,
        })

        condensed_metrics_df = get_scores(results_df['gt'], results_df['predictions'], results_df['predictions'].round())
                
        return condensed_metrics_df

    def get_per_cohort_metrics(self, prediction: torch.Tensor, test_gt: torch.Tensor, test_ids: list, seed: int) -> tuple:
        """
        Function to parse results and create output to write to file (both for the entire test set and per-cohort)
        :param prediction: array-like object with predictions of classifier
        :param test_gt: array-like object with predictions of classifier
        :param test_ids: list of ids included in the test set
        :param seed: int with random seed of particular replicate that is added to the results df.
        """
        cohorts_to_disease = self.config['cohorts_to_disease']

        if not isinstance(prediction, torch.Tensor):
            prediction = torch.from_numpy(prediction)
        if not isinstance(test_gt, torch.Tensor):
            test_gt = torch.tensor(test_gt)
                
        per_cohort_metrics_df = per_cohort_metrics(
            list(cohorts_to_disease.keys()),
            cohorts_to_disease,
            prediction,
            test_gt,
            test_ids,
            self.dataset,
            exp=seed
        )

        return per_cohort_metrics_df

    def set_ids(self, seed, *args, **kwargs):
        pass

    def run_training(self, seed: int, ids: dict, model, hp: dict) -> tuple:
        """
        Method that runs testing and returns a state and trainer.
        :return: tuple with state dict and gumibair.Trainer object.
        """
        set_seed(seed)
        self.check_last_batch(ids['inner_train_ids'], "drop_last_train")
        self.check_last_batch(ids['val_ids'], "drop_last_val")


        if self.monitor_test_auc:
            test_set = self.dataset[ids['test_ids']]
        else:
            test_set=None

        state, trainer = train_cmvib(self.dataset, ids, model, hp, self.config, test_set=test_set)
        return state, trainer

    def run_testing(self, state: dict, trainer: Trainer, ids: dict, *args, **kwargs) -> tuple:
        """
        Method that tests a state of the model.
        :param state: state dict of the trained model.
        :param trainer: gumibair.Trainer object
        :return: tuple with prediction and test_gt tensors.
        """
        _, prediction, test_gt, __, ___ = test_cmvib(state, self.dataset, ids, trainer, self.config)
        return prediction, test_gt

    def run(self, seed: int, ids: dict, hp, *args, **kwargs) -> pd.DataFrame:
        """
        Method to run one replicate of an experiment.
        :param seed: int with random seed for that specific replicate.
        :return: tuple of two pd.DataFrames with condensed scores and per-cohort scores.
        """
        print(f"Logged in {self.config['log_dir']}")
        state, trainer = self.run_training(seed, ids, self.cmvib, hp)
        prediction, test_gt = self.run_testing(state, trainer, ids, *args, **kwargs)

        return prediction, test_gt

    def benchmark_rf(self, seed: int, ids: dict):
        """
        Method to benchmark the experiment against Random Forest.
        :param seed: int with random seed used for current replicate.
        :return: tuple with a pd.DataFrame containing performance metrics, plus the predictions  and test ground truth.
        """
        if all(value in ids.keys() for value in ['inner_train_ids', 'val_ids', 'y_inner_train', 'y_val']):
            train_ids = ids['inner_train_ids'] + ids['val_ids']
            y_train = ids['y_inner_train'] + ids['y_val']
        else:
            train_ids = ids['train_ids']
            y_train = ids['y_train']
        test_ids = ids['test_ids']
        y_test = ids['y_test']

        start = timer()
        test_gt, prediction = run_rf(self.dataset, train_ids, test_ids, y_train, y_test, seed, self.rf)
        end = timer()
        print(f"Completed! Time (s): {end - start}")

        return prediction, test_gt

    def benchmark_mvib(self, seed: int, ids: dict, hp: dict, *args, **kwargs):
        """
        Method to benchmark the experiment against MVIB.
        If self.benchmark, then for every replicate,  self.dataset.conditions is set to the same index for all samples.
        run() is called again, afterwards the self.dataset.conditions is set back to the original.
        """
        print(f"Logged in {self.config['log_dir']}")
        original_conditions = self.dataset.conditions.clone()
        mvib_condition = torch.zeros(original_conditions[0].shape)
        mvib_condition[0] = 1
        self.dataset.conditions[1:] = mvib_condition

        state, trainer = self.run_training(seed, ids, self.mvib, hp)
        prediction, test_gt = self.run_testing(state, trainer, ids, *args, **kwargs)

        self.dataset.conditions = original_conditions

        return prediction, test_gt
    
    def run_replicates(self, *args, **kwargs) -> list:
        """
        Method to run all the replicates of an experiment and return the obtained metrics.
        """
        pass