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

from mcbn_experiments.base_exp import _BaseExperiment
from gumibair.dataset import FullMicrobiomeDataset
from gumibair.trainer import Trainer
from mcbn_experiments.utils import partial_hold_out_split, complete_hold_out_split
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import inspect
from typing import Tuple
import numpy as np
import torch

class CrossCohortExperiment(_BaseExperiment):
    """Class for cross-cohort experiments."""
    def __init__(self, 
                 dataset: FullMicrobiomeDataset, 
                 config: dict, 
                 mode: str = "partial", 
                 replicates: int = 5,
                 benchmark: bool = True,
                 all_cohorts: bool = False,
                 test_prop: float = 0.9,
                 test_condition: str = None,
                 monitor_test_auc: bool = False,
                 cross_validate: bool = False):
        """
        :param dataset: FullMicroBiomeDataset object linked to the experiment
        :param config: dictionary with basic configurations from HPO etc.
        :param mode: string defining wether to hold out test cohort partially or completely.
        :param replicates: int defining how often the experiment should be executed.
        :param all_cohorts: bool deciding wether to train on all cohorts in dataset or only on the test cohort.
        :param test_prop: float representing the percentage of held-out cohort to be contained in testing.
        :param test_condition: str deciding how to condition the test samples. Right now either 'validation_set' or 'mean'
        :param cross_validate: bool deciding wether the experiment should be cross-validated during training.  
        """

        _BaseExperiment.__init__(self, 
                            dataset, 
                            config, 
                            replicates=replicates, 
                            benchmark=benchmark,
                            monitor_test_auc=monitor_test_auc)
        
        self.mode = mode
        self.all_cohorts = all_cohorts
        self.test_prop = test_prop
        self.test_condition = test_condition
        self.cross_validate = cross_validate
        self.best_condition = None
    
    def set_ids(self, seed: int, heldout_cohort_idx: int):
        """
        Class specific method to define train/val/test ids and gt_labels for each replicate.
        Sampling method is chosen based on self.mode. Either partial holdout or complete holdout.
        :param seed: int representing random seed
        :param heldout_cohort_index: int represent index of the cohort to be held out for testing.
        """
        if self.mode == "partial":
            train_ids, test_ids, y_train, y_test = partial_hold_out_split(
                seed,
                self.dataset, 
                heldout_cohort_idx,
                self.test_prop,
                all_cohorts=self.all_cohorts, 
                stratify=True
            )

        elif self.mode == "complete":
            train_ids, test_ids, y_train, y_test = complete_hold_out_split(
                self.dataset,
                heldout_cohort_idx
            )

        else:
            raise NotImplementedError("Currently only 'partial' or 'complete' are accepted as mode.")
        
        if self.cross_validate:
            ids = {
                'train_ids': train_ids,
                'test_ids': test_ids,
                'y_train': y_train,
                'y_test': y_test
            }
        
        else:
            inner_train_ids, val_ids, y_inner_train, y_val = model_selection.train_test_split(
                    train_ids,
                    y_train,
                    test_size=self.config.get('val_size'),
                    random_state=seed,
                    stratify=y_train
                )

            ids = {
                "inner_train_ids": inner_train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "y_inner_train": y_inner_train,
                "y_val": y_val,
                "y_test": y_test,
            }

        self.ids = ids

    def _cross_validate(self,
                        func,
                        seed: int, 
                        ids: dict,
                        hp: dict,
                        heldout_cohort_idx: int,
                        *args, **kwargs) -> Tuple[np.array, np.array]:
        """
        Method to cross validate the experiment, 
        which in this context means that each cohort in the training set is used as validation set once.
        Afterwards a majority vote is made by averaging all predictions of one sample to obtain one final prediction for each sample in the test set.
        :param seed: int with random seed for a particular replicate
        :param ids: dict with train/test ids and labels.
        :param hp: dict with hyperparameters for the particular model to be trained and tested.
        :return: tuple with predicitons stack and test_gt stack from all models in the ensemble.
        """
        predictions = list()

        counter = 0
        original_logdir = self.config['log_dir']
        for idx, cohort in enumerate(self.dataset.cohorts):
            if idx == heldout_cohort_idx:
                continue
            self.config['log_dir'] = f"{original_logdir}{counter}"
            counter += 1
            print(f"Run {counter}, current validation set: {cohort}")
            val_ids = self.dataset.cohorts_2_patients[idx]['patients']
            y_val = self.dataset.cohorts_2_patients[idx]['labels']
            inner_train_ids = [i for i in ids['train_ids'] if not i in val_ids]
            y_inner_train = [l for i, l in enumerate(self.dataset.labels) if i in inner_train_ids]

            self.ids = {
                'inner_train_ids': inner_train_ids,
                'val_ids': val_ids,
                'test_ids': ids['test_ids'],
                'y_inner_train': y_inner_train,
                'y_val': y_val,
                'y_test': ids['y_test']
            }

            self.initialize_models(seed)
            prediction, _ = func(seed, self.ids, hp, heldout_cohort_idx, *args, **kwargs)
            if type(prediction) == np.ndarray:
                prediction = np.reshape(prediction, (prediction.shape[0],1))
                prediction = torch.tensor(prediction, dtype=torch.float32)
                
            predictions.append(prediction)
   
        prediction_stack = torch.stack(predictions).mean(dim=0)
        
        test_gt = self.ids['y_test']
        self.config['log_dir'] = original_logdir

        if self.benchmark:
            self.set_ids(seed, heldout_cohort_idx)

        return prediction_stack, test_gt

    def change_condition_from_idx(self, ids: list, cohort_idx: int):
        """
        Method to change the condition of a set of ids based on a cohort index
        Sets the condition on the dataset attribute of the instance.
        :param ids: list of ids for which the condition should be changed
        :param cohort_index: int with index of cohort to whichs condition the ids should be changed
        """
        set_condition = self.dataset.conditions[self.dataset.cohorts_2_patients[cohort_idx]['patients'][0]].clone()
        self.dataset.conditions[ids] = set_condition

    def set_condition_to_mean(self, ids: list):
        """
        Method that sets the cohort idx array of the test samples to something like [1/C, 1/C, .../ 1/C],
        where C is the total number of conditions (length of self.dataset.conditions)
        :param ids: list with ids for which the condition should be changed (test samples)
        """
        mean_condition = torch.tensor([1/len(self.dataset.cohorts) for _ in range(len(self.dataset.cohorts))])
        self.dataset.conditions[ids] = mean_condition

    def get_best_condition(self, state: dict, trainer: Trainer, heldout_cohort_idx: int):
        """
        Method to find the best working cohort index to condition held-out cohort on.
        Based on validation set.
        If no samples of the heldout cohort are contained in the validation set, the method just returns None.
        :param state: state dict of the trained model.
        :param trainer: gumibair.Trainer object.
        :return: either the index of the best performing cohort or None.
        """
        val_ids = self.ids['val_ids']
        y_val = self.ids['y_val']

        val_abundance, val_markers, _, val_cond, val_heads = self.dataset[val_ids]
        device = self.config['device']

        val_abundance = val_abundance.to(device)
        val_markers = val_markers.to(device)
        val_cond = val_cond.to(device)
        val_heads = val_heads.to(device)

        model, _ = trainer.load_checkpoint(state)
        model.eval()
        model.to(device)

        cond_performance = dict()
        heldout_val_ids = [x for x in val_ids if x in self.dataset.cohorts_2_patients[heldout_cohort_idx]['patients']]
        if heldout_val_ids:
            for idx, cohort in enumerate(self.dataset.cohorts):
                if idx == heldout_cohort_idx:
                    continue
                heldout_val_ids = [x for x in val_ids if x in self.dataset.cohorts_2_patients[heldout_cohort_idx]['patients']]
                original_condition = self.dataset.conditions[self.ids['test_ids'][0]].clone()
                self.change_condition_from_idx(heldout_val_ids, idx)
                prediction = model.classify((val_abundance, val_markers, val_heads, val_cond))
                prediction = prediction.cpu().detach().numpy().squeeze().tolist()
                cond_performance[idx] = roc_auc_score(y_val, prediction)

                self.dataset.conditions[heldout_val_ids] = original_condition

            return max(cond_performance, key=cond_performance.get)
        else:
            return None
    
    def run_testing(self, state: dict, trainer: Trainer, ids: dict, heldout_cohort_idx: int):
        """
        Method that overrides `run_testing()` from the parent Experiment class.
        Implements picking the best-working condition for the heldout cohort,
        based on validation set performance.
        :param state: state dict of the trained model.
        :param trainer: gumibair.Trainer object.
        :param ids: dict with train/val/test ids and labels.
        :param heldout_cohort_idx: int representing the cohort index of the heldout cohort.
        :return: tuple with prediction and test_gt tensors.
        """
        if self.test_condition is not None and trainer.model is self.cmvib:
            original_condition = self.dataset.conditions[ids['test_ids'][0]].clone()
            if self.test_condition == "validation_set":
                print("Getting best condition:")
                best_condition_idx = self.get_best_condition(state, trainer, heldout_cohort_idx)
                if best_condition_idx is not None:
                    self.best_condition = list(self.dataset.cohorts)[best_condition_idx]
                    print(f"Best Condition is Cohort idx {best_condition_idx}")
                    self.change_condition_from_idx(ids['test_ids'], best_condition_idx)
                    prediction, test_gt = super().run_testing(state, trainer, ids)
                    self.dataset.conditions[ids['test_ids']] = original_condition
                else:
                    print("No samples from held-out cohort found in val-set. Continuing without changing condition.")
                    self.best_condition = 'unchanged'
                    prediction, test_gt = super().run_testing(state, trainer, ids)
            elif self.test_condition == "mean":
                print("Setting condition to average for the test set:")
                self.set_condition_to_mean(ids['test_ids'])
                prediction, test_gt = super().run_testing(state, trainer, ids)
                self.dataset.conditions[ids['test_ids']] = original_condition
        else:
            prediction, test_gt = super().run_testing(state, trainer, ids)

        return prediction, test_gt

    def run(self, seed: int, ids: dict, hp: dict, heldout_cohort_idx: int) -> Tuple:
        """
        Method overriding the run() method from the parent class, to pass the heldout_cohort_idx to run_testing.
        This is mainly to enable the run_testing() method to figure out the best test condition, when test_on_best_condition == True.
        Also, the overriding method enables using the _cross_validate() wrapper for training.
        :param seed: int with random seed for a particular replicate.
        :param ids: dict with train/val/test ids and labels.
        :param model: instance of the model to be trained and tested.
        :param hp: dict with hyperparameters for the particular model to be trained and tested.
        :param heldout_cohort_idx: int representing the cohort index of the heldout cohort.
        return: tuple with array-like objects containing the predictions and test_gt
        """
        if self.cross_validate:
            prediction, test_gt = self._cross_validate(super().run, seed, ids, hp, heldout_cohort_idx)
        else:
            prediction, test_gt = super().run(seed, ids, hp, heldout_cohort_idx)
        return prediction, test_gt
    
    def benchmark_mvib(self, seed: int, ids: dict, hp: dict, heldout_cohort_idx: int):
        """
        Method overriding the benchmark_rf() method from the parent class, to pass the heldout_cohort_idx to run_testing.
        This is mainly to enable the run_testing() method to figure out the best test condition, when test_on_best_condition == True.
        Also, the overriding method enables using the _cross_validate() wrapper for training.
        :param seed: int with random seed for a particular replicate.
        :param ids: dict with train/val/test ids and labels.
        :param model: instance of the model to be trained and tested.
        :param hp: dict with hyperparameters for the particular model to be trained and tested.
        :param heldout_cohort_idx: int representing the cohort index of the heldout cohort.
        return: tuple with array-like objects containing the predictions and test_gt
        """
        if self.cross_validate:
            prediction, test_gt = self._cross_validate(super().benchmark_mvib, seed, ids, hp, heldout_cohort_idx)
        else:
            prediction, test_gt = super().benchmark_mvib(seed, ids, hp, heldout_cohort_idx)
        return prediction, test_gt
    
    def benchmark_rf(self, seed: int, ids: dict, hp: dict, heldout_cohort_idx: int):
        """
        Method overriding the benchmark_rf() method from the parent class, to pass the heldout_cohort_idx to run_testing.
        Also, the overriding method enables using the _cross_validate() wrapper for training.
        :param seed: int with random seed for a particular replicate.
        :param ids: dict with train/val/test ids and labels.
        :param model: instance of the model to be trained and tested.
        :param hp: dict with hyperparameters for the particular model to be trained and tested.
        :param heldout_cohort_idx: int representing the cohort index of the heldout cohort.
        return: tuple with array-like objects containing the predictions and test_gt
        """
        if self.cross_validate:
            prediction, test_gt = self._cross_validate(super().benchmark_rf, seed, ids, hp, heldout_cohort_idx)
        else:
            prediction, test_gt = super().benchmark_rf(seed, self.ids, hp, heldout_cohort_idx)
        return prediction, test_gt
    
    
    def run_replicates(self) -> list:
        """
        method that implements running all replicates of the current experiment and collecting the results.
        :return: list with pd.DataFrame objects containing the results from the individual replicates.
        """
        condensed_metrics = list()
        original_logdir = self.config['log_dir']
        for idx, cohort in enumerate(self.dataset.cohorts): 
            cohort_metrics = list()

            print(f"Current heldout-cohort: {cohort}")
            for seed in range(self.replicates):
                self.config['log_dir'] = f"{original_logdir}CMVIB/{cohort}/{seed}/"

                if self.config['gene_filter'] is not 'None':
                    self.dataset.subset_markers(cutoff=int(self.config['gene_filter']))
                    self.config['marker_dim'] = self.dataset.markers.shape[1]

                self.initialize_models(seed)
                self.set_ids(seed, idx)

                print(f"Replicate {seed}:")
                prediction, test_gt = self.run(seed, self.ids, self.config['CMVIB'], idx)
                condensed_scores = self.get_condensed_metrics(prediction, test_gt)
                condensed_scores['model'] = 'GUMIBAIR'
                condensed_scores['exp'] = seed
                if self.test_condition == 'validation_set':
                    condensed_scores['best_condition'] = self.best_condition

                print(condensed_scores)
                cohort_metrics.append(condensed_scores)

                if self.benchmark:
                    print("Running MVIB benchmark:")
                    self.config['log_dir'] = f"{original_logdir}MVIB/{cohort}/{seed}/"
                    mvib_prediction, mvib_test_gt = self.benchmark_mvib(seed, self.ids, self.config['MVIB'], idx)
                    mvib_condensed_scores = self.get_condensed_metrics(mvib_prediction, mvib_test_gt)
                    mvib_condensed_scores['model'] = 'MVIB'
                    mvib_condensed_scores['exp'] = seed
                    
                    cohort_metrics.append(mvib_condensed_scores)

                    print(mvib_condensed_scores)

                    if self.config['gene_filter'] is not 'None':
                        self.dataset.markers = self.dataset.original_markers

                    print("Running RF benchmark:")
                    rf_prediction, rf_test_gt = self.benchmark_rf(seed, self.ids, self.config['RF'], idx)
                    rf_condensed_scores = self.get_condensed_metrics(rf_prediction, rf_test_gt)
                    rf_condensed_scores['model'] = 'RF'
                    rf_condensed_scores['exp'] = seed

                    cohort_metrics.append(rf_condensed_scores)

            self.config['log_dir'] = original_logdir

            for df in cohort_metrics:
                df['heldout_cohort'] = cohort
                if self.mode == "partial":
                    df['test_prop'] = self.test_prop
            condensed_metrics.extend(cohort_metrics)

        return condensed_metrics