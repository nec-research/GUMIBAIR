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
from mcbn.dataset import FullMicrobiomeDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import pandas as pd
import numpy as np
from typing import Tuple

class InCohortExperiment(_BaseExperiment):
    """Class for in-cohort experiments."""
    def __init__(self, 
                 dataset: FullMicrobiomeDataset, 
                 config: dict, 
                 replicates: int = 5,
                 benchmark: bool = True,
                 cross_validation: bool = False,
                 n_splits: int = 5,
                 monitor_test_auc: bool = False):

        _BaseExperiment.__init__(self, 
                            dataset, 
                            config, 
                            replicates=replicates, 
                            benchmark=benchmark, 
                            monitor_test_auc=monitor_test_auc)
        
        self.cross_validation = cross_validation
        self.n_splits=n_splits


    def set_ids(self, seed: int) -> dict:
        """
        Class specific method to define train/val/test ids and gt_labels for each replicate.
        :param seed: int with random seed of a particular replicate
        :return: either a dictionary with train/val/test ids or a list of cohort-specific generators for the cross validation splits.
        Returns a dict with train/val/test ids based on stratified sampling method from the FMD class.
        """
        if not self.cross_validation:
            inner_train_ids, val_ids, test_ids, y_inner_train, y_val, y_test = self.dataset.per_cohort_train_val_test_split(
                test_size=self.config.get('test_size'),
                val_size=self.config.get('val_size'), 
                random_state=seed, 
                test_stratify=False
            )

            self.ids = {
                "inner_train_ids": inner_train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "y_inner_train": y_inner_train,
                "y_val": y_val,
                "y_test": y_test,
            }
        else:
            cohort_splits = list()
            for cdx, _ in enumerate(self.dataset.cohorts):
                k_folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                splits = k_folds.split(self.dataset.cohorts_2_patients[cdx]['patients'], self.dataset.cohorts_2_patients[cdx]['labels'])
                cohort_splits.append(splits)
            self.ids = cohort_splits

    def _cross_validate(self, func, seed, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        A wrapper that should decorate the run() method as well as the run_rf method, 
        when self.cross_validation == true.
        :param func: method that trains and test a model.
        :param seed: int with random seed for a particular replicate.
        :return: tuple of pd.DataFrames with condensed scores and per-cohort scores.
        """
        prediction_stack = np.array([])
        test_gt_stack = np.array([])
        test_ids_stack = list()

        original_log_dir = self.config['log_dir']

        for i in range(self.n_splits):
            self.config['log_dir'] = f"{original_log_dir}cv_{i}"
            self.initialize_models(seed)

            print(f"CV SPLIT {i}")

            train_ids = list()
            test_ids = list()
            for cdx, generator in enumerate(self.ids):
                train_idx, test_idx = next(generator)
                c_train_ids = [x for i, x in enumerate(self.dataset.cohorts_2_patients[cdx]['patients']) if i in train_idx]
                c_test_ids = [x for i, x in enumerate(self.dataset.cohorts_2_patients[cdx]['patients']) if i in test_idx]
                train_ids.extend(c_train_ids)
                test_ids.extend(c_test_ids)
    
            y_train = [l for i, l in enumerate(self.dataset.labels) if i in train_ids]
            y_test = [l for i, l in enumerate(self.dataset.labels) if i in test_ids]

            inner_train_ids, val_ids, y_inner_train, y_val = train_test_split(
                train_ids,
                y_train,
                test_size=self.config['val_size'],
                stratify=y_train
            )

            split_ids = {
                'inner_train_ids': inner_train_ids,
                'val_ids': val_ids,
                'test_ids': test_ids,
                'y_inner_train': y_inner_train,
                'y_val': y_val,
                'y_test': y_test,
            }

            prediction, test_gt = func(seed, split_ids, *args, **kwargs)
            if type(prediction) == torch.Tensor:
                prediction = prediction.cpu().detach().numpy().squeeze().tolist()
            if type(test_gt) == torch.Tensor:
                test_gt = test_gt.cpu().detach().numpy().squeeze().tolist()

            prediction_stack = np.concatenate(
                (prediction_stack, prediction),
                axis=0
            )

            test_gt_stack = np.concatenate(
                (test_gt_stack, test_gt),
                axis=0
            )

            test_ids_stack.extend(split_ids['test_ids'])

        condensed_scores = self.get_condensed_metrics(prediction_stack, test_gt_stack)
        per_cohort_scores = self.get_per_cohort_metrics(prediction_stack, test_gt_stack, test_ids_stack, seed)

        if self.benchmark:
            self.set_ids(seed)

        self.config['log_dir'] = original_log_dir
        return condensed_scores, per_cohort_scores


    def run(self, seed: int, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method that overrides the `run()` method from the Parent class and cross validates each replicate, 
        if self.cross_validation == True.
        :param seed: int with random seed for particular replicate.
        :return: tuple of pd.DataFrames with condensed scores and per-cohort scores.
        """
        if self.cross_validation:
            condensed_scores, per_cohort_scores = self._cross_validate(super().run, seed, *args, **kwargs)
        else:
            prediction, test_gt = super().run(seed, self.ids, *args, **kwargs)
            condensed_scores = self.get_condensed_metrics(prediction, test_gt)
            per_cohort_scores = self.get_per_cohort_metrics(prediction, test_gt, self.ids['test_ids'], seed)
            
        condensed_scores['exp'] = seed
        per_cohort_scores['exp'] = seed
        return condensed_scores, per_cohort_scores
    
    def benchmark_mvib(self, seed: int, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method that overrides the `benchmark_mvib()` method from the Parent class and cross validates each replicate, 
        if self.cross_validation == True.
        :param seed: int with random seed for particular replicate.
        :return: tuple of pd.DataFrames with condensed scores and per-cohort scores.
        """
        if self.cross_validation:
            condensed_scores, per_cohort_scores = self._cross_validate(super().benchmark_mvib, seed, *args, **kwargs)
        else:
            prediction, test_gt = super().benchmark_mvib(seed, self.ids, *args, **kwargs)
            condensed_scores = self.get_condensed_metrics(prediction, test_gt)
            per_cohort_scores = self.get_per_cohort_metrics(prediction, test_gt, self.ids['test_ids'], seed)
            
        condensed_scores['exp'] = seed
        per_cohort_scores['exp'] = seed
        return condensed_scores, per_cohort_scores

    def benchmark_rf(self, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method that overrides the `benchmark_rf()` method from the Parent class and cross validates each replicate, 
        if self.cross_validation == True.
        :param seed: int with random seed for particular replicate.
        :return: tuple of pd.DataFrames with condensed scores and per-cohort scores.
        """
        if self.cross_validation:
            condensed_scores, per_cohort_scores = self._cross_validate(super().benchmark_rf, seed)
        else:
            prediction, test_gt = super().benchmark_rf(seed, self.ids)
            condensed_scores = self.get_condensed_metrics(prediction, test_gt)
            per_cohort_scores = self.get_per_cohort_metrics(prediction, test_gt, self.ids['test_ids'], seed)

        condensed_scores['exp'] = seed
        per_cohort_scores['exp'] = seed
        return condensed_scores, per_cohort_scores
        
    def run_replicates(self, *args, **kwargs):
        """
        Method to run all the replicates of an experiment and return the obtained metrics.
        If self.benchmark, adds benchmark scores for RF and MVIB.
        :param replicate_ids: dictionary with train/val/test ids and labels.
        :return: tuple containing list of pd.DataFrames with performance metrics.
        """
        condensed_metrics = list()
        per_cohort_metrics = list()

        original_log_dir = self.config['log_dir']
        for seed in range(self.replicates):
            if self.config['gene_filter'] is not 'None':
                self.dataset.subset_markers(cutoff=int(self.config['gene_filter']))
                self.config['marker_dim'] = self.dataset.markers.shape[1]

            self.initialize_models(seed)

            self.set_ids(seed, *args, **kwargs)

            print(f"Replicate {seed}:")
            self.config['log_dir'] = f"{original_log_dir}cmvib/{seed}/"
            condensed_scores, per_cohort_scores = self.run(seed, self.config['CMVIB'], *args, **kwargs)
            condensed_scores['model'] = 'GUMIBAIR'
            per_cohort_scores['model'] = 'GUMIBAIR'

            condensed_metrics.append(condensed_scores)
            per_cohort_metrics.append(per_cohort_scores)
            print(condensed_scores)

            if self.benchmark:
                print("Running MVIB benchmark:")
                self.config['log_dir'] = f"{original_log_dir}mvib/{seed}/"
                mvib_condensed_scores, mvib_per_cohort_scores = self.benchmark_mvib(seed, self.config['MVIB'], *args, **kwargs)
                mvib_condensed_scores['model'] = 'MVIB'
                mvib_per_cohort_scores['model'] = 'MVIB'
                
                condensed_metrics.append(mvib_condensed_scores)
                per_cohort_metrics.append(mvib_per_cohort_scores)

                if self.config['gene_filter'] is not 'None':
                    self.dataset.markers = self.dataset.original_markers

                print("Running RF benchmark:")
                rf_condensed_scores, rf_per_cohort_scores = self.benchmark_rf(seed)
                rf_condensed_scores['model'] = 'RF'
                rf_per_cohort_scores['model'] = 'RF'

                condensed_metrics.append(rf_condensed_scores)
                per_cohort_metrics.append(rf_per_cohort_scores)

        return condensed_metrics, per_cohort_metrics