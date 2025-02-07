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

"""A Dataset class for the microbiome data.
"""
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale

def get_cohorts_to_disease_idx(cohort_dict):
    disease_to_id = {
        k: v for v, k in enumerate(set(cohort_dict.values()))
    }
    return {
        k: disease_to_id[v] for k, v in cohort_dict.items()
    }


class FullMicrobiomeDataset(Dataset):
    """Create dataset for all cohorts.
    """
    def __init__(self, data_dir, device, label_dict, scale_abundance=True, scale_markers=False, data='default', cohorts=None, start_idx_df=210,
                 conditioning_type='cohort', conditioning_labels: dict = {}, cohorts_to_disease=None, 
                 drop_samples: dict = None):
        super(FullMicrobiomeDataset, self).__init__()
        self.data_dir = data_dir
        self.cohorts = cohorts
        self.device = device
        self.label_dict = label_dict
        self.data = data
        self.start_idx_df = start_idx_df
        self.conditioning_type = conditioning_type
        self.conditioning_labels = conditioning_labels
        self.drop_samples = drop_samples

        abundance_stack = None
        markers_stack = None
        labels_stack = None
        conditions_stack = None
        heads_stack = None
        label_conditions_stack = None

        cohorts_to_disease_idx = get_cohorts_to_disease_idx(cohorts_to_disease)
        self.cohorts_2_patients = {}
        patients_running_sum = 0

        self.metadata = self._get_metadata()

        for d_idx, d in enumerate(cohorts):
            raw_data = self.load_disease_data(d, data_dir, data, drop_samples=self.drop_samples)
            abundance = raw_data['abundance'].iloc[start_idx_df:, :].T.to_numpy().astype(np.float)
            markers = raw_data['marker'].iloc[start_idx_df:, :].T.to_numpy().astype(np.float)
            labels = self._get_labels(raw_data, cohorts_to_disease[d])

            self.cohorts_2_patients[d_idx] = {
                'patients': list(range(patients_running_sum, patients_running_sum+len(labels))),
                'labels': list(labels.cpu().numpy().squeeze().astype('int'))
            }
                     
            patients_running_sum += len(labels)

            if self.conditioning_type == 'cohort':
                conditions = np.zeros((len(labels), len(cohorts)))
                conditions[:,d_idx]=1
            elif self.conditioning_type == 'disease':
                conditions = np.zeros((len(labels), len(set(cohorts_to_disease.values()))))
                conditions[:, cohorts_to_disease_idx[d]] = 1
            elif self.conditioning_type == 'multi':
                if self.conditioning_labels == {}:
                    raise Exception("'conditioning_labels' seems to be empty. Ensure that it's set during instantiating of the dataset object!")
                conditions = self._encode_conditions(raw_data, labels)
            else:
                raise NotImplementedError
            
            heads = np.zeros((len(labels)))
            heads[:] = cohorts_to_disease_idx[d]

            if abundance_stack is None:
                abundance_stack = abundance
                markers_stack = markers
                labels_stack = labels
                conditions_stack = conditions
                heads_stack = heads

            else:
                abundance_stack = np.concatenate((abundance_stack, abundance), axis=0)
                markers_stack = np.concatenate((markers_stack, markers), axis=0)
                labels_stack = torch.cat((labels_stack, labels), dim=0)
                conditions_stack = np.concatenate((conditions_stack, conditions), axis=0)
                heads_stack = np.concatenate((heads_stack, heads), axis=0)

        abundance_stack = torch.tensor(abundance_stack, dtype=torch.float32)#.to_dense()
        markers_stack = torch.tensor(markers_stack, dtype=torch.float32)#.to_dense()
        conditions_stack = torch.tensor(conditions_stack, dtype=torch.float32)#.to_dense()
        heads_stack = torch.tensor(heads_stack, dtype=torch.float32)#.to_dense()

        self.unscaled_markers = markers_stack.clone()

        if scale_abundance:
            ab_scaler = StandardScaler()
            abundance_stack = ab_scaler.fit_transform(abundance_stack)
            abundance_stack = torch.from_numpy(abundance_stack)

        if scale_markers:
            markers_scaler = StandardScaler()
            markers_stack = markers_scaler.fit_transform(markers_stack)
            markers_stack = torch.from_numpy(markers_stack)

        self.abundance = abundance_stack#.to(device)
        self.markers = markers_stack#.to(device)
        self.labels_gpu = labels_stack
        self.labels = labels_stack.cpu().numpy().squeeze().astype('int')
        self.conditions = conditions_stack#.to(device)
        self.heads = heads_stack#.to(device)


        self.patients_ids = np.array([i for i in range(len(self.abundance))])

    def __getitem__(self, idx):
        return self.abundance[idx], self.markers[idx], self.labels_gpu[idx], self.conditions[idx], self.heads[idx]

    def __len__(self):
        return len(self.patients_ids)

    @staticmethod
    def load_disease_data(cohort, data_dir, data, drop_samples=None):
        """Load the data of a specified disease.
        """
        raw_data = dict()

        if data == 'joint':
            raw_data['abundance'] = pd.read_csv(
                os.path.join(data_dir, data, 'abundance/abundance_{}.txt'.format(cohort))
            )
            raw_data['marker'] = pd.read_csv(
                os.path.join(data_dir, data, 'marker/marker_{}.txt'.format(cohort))
            )
            # fix the name of the first data column
            raw_data['abundance'] = raw_data['abundance'].rename(columns={'Unnamed: 0': 'sampleID'})
            raw_data['marker'] = raw_data['marker'].rename(columns={'Unnamed: 0': 'sampleID'})
        else:
            raw_data['abundance'] = pd.read_csv(
                os.path.join(data_dir, 'default', 'abundance/abundance_{}.txt'.format(cohort)),
                sep="\t",
                skiprows=1
            )
            raw_data['marker'] = pd.read_csv(
                os.path.join(data_dir, data, 'marker/marker_{}.txt'.format(cohort)),
                sep="\t",
                skiprows=1
            )

        # drop non-patient column
        raw_data['abundance'] = raw_data['abundance'].set_index('sampleID')
        raw_data['marker'] = raw_data['marker'].set_index('sampleID')

        if drop_samples and drop_samples.get(cohort):
            raw_data['abundance'].drop(columns=drop_samples[cohort], inplace=True)
            raw_data['marker'].drop(columns=drop_samples[cohort], inplace=True)

        assert len(raw_data['abundance'].columns) == len(raw_data['marker'].columns)
        return raw_data

    def _get_labels(self, raw_data, disease):
        """Create labels.
        """
        labels = np.array([[self.label_dict[disease][i]] for i in raw_data['marker'].loc['disease'].to_list()])
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(self.device)
        return labels
    
    def _get_metadata(self):
        """
        Method to get the union of the metadata labels from all cohorts in the dataset.
        :return: dict with label as key and list of unique (if not numerical) values in the dataset as value.
        """
        metadata = list()
        for cohort in self.cohorts:
            df = pd.read_csv(
                os.path.join(self.data_dir, self.data, f"marker/marker_{cohort}.txt"),
                nrows=self.start_idx_df - 1
            )
            df = df[df['Unnamed: 0'].str.contains('field_') == False]

            if self.drop_samples and self.drop_samples.get(cohort):
                df.drop(columns=self.drop_samples[cohort], inplace=True)
            metadata.append(df)

        unionized_metadata = pd.DataFrame(columns=['Unnamed: 0'])
        for df in metadata:
            unionized_metadata = pd.merge(unionized_metadata, df, on='Unnamed: 0', how='outer')
        unionized_metadata.fillna('nd', inplace=True)

        unionized_metadata.set_index('Unnamed: 0', inplace=True, drop=True)
        unionized_T = unionized_metadata.T

        metadata = dict()
        for column in unionized_T.columns:
            if self.conditioning_labels: 
                if column in self.conditioning_labels.keys() and self.conditioning_labels[column] == 'numerical':
                    metadata[column] = unionized_T[column].tolist()
                if column in self.conditioning_labels.keys() and self.conditioning_labels[column] == 'categorical':
                    metadata[column] = unionized_T[column].unique().tolist()
            else:
                metadata[column] = unionized_T[column].tolist()
        return metadata

    def _encode_conditions(self, raw_data, labels):
        """
        Method to create a binary encoding of the conditions based on the metadata.
        :param raw_data: pd.DataFrame with raw data from a specific cohort.
        :param labels: torch.Tensor with gt labels, used to infer on the number of samples.
        :return: numpy array of all conditions concatenated on the last axis.
        """
        conditions = list()
        for label, kind in self.conditioning_labels.items():
            if kind == 'categorical':
                condition = np.zeros((len(labels), len(self.metadata[label])))
                for i, _ in enumerate(labels):
                    sample_cond = raw_data['marker'].iloc[:, i].loc[label]
                    sample_cond_idx = self.metadata[label].index(sample_cond)
                    condition[i, sample_cond_idx] = 1
            elif kind  == 'numerical':
                condition  = np.zeros((len(labels), 1))
                normalized_data = minmax_scale(self.metadata[label])
                for i, _ in enumerate(labels):
                    sample_cond = normalized_data[i]
                    condition[i, 0] = sample_cond
            else:
                raise NotImplementedError("Only 'categorical' or 'numerical' are accepted!")
            if len(self.conditioning_labels) == 1:
                return condition
            else:
                conditions.append(condition)
                
        conditions = np.concatenate(conditions, axis=-1) 
        return conditions
    
    def subset_markers(self, cutoff=5):
        """
        Method to subset the gene markers based on sample prevalence cutoff.
        Identifies all the marker genes that are prevalent in <= cutoff samples based on the unscaled marker abundance.
        markers are then filtered out on the values that are used as final inputs (self.markers).
        """
        self.original_markers = self.markers.clone()
        binary_markers_stack = torch.where(self.unscaled_markers.clone() > 0, 1.0, 0.0)
        sample_prevalence = torch.sum(binary_markers_stack, 0)
        filtered_indices = (sample_prevalence >= cutoff).nonzero().flatten().numpy()
        self.markers = self.markers[:, filtered_indices]

    def train_test_split(self, test_size, random_state):
        """Do stratified train/test split.
        """
        train_ids, test_ids, y_train, y_test = model_selection.train_test_split(
            self.patients_ids,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        return train_ids, test_ids, y_train, y_test

    def per_cohort_train_test_split(self, test_size, random_state, stratify=False):
        """Do stratified train/test split for each cohort.
        """
        train_ids, test_ids, y_train, y_test = [], [], [], []
        
        if stratify:
            stratify_labels = self.cohorts_2_patients[c_idx]['labels']
        else:
            stratify_labels = None

        for c_idx, c in enumerate(self.cohorts):                           
            temp1, temp2, temp3, temp4 = model_selection.train_test_split(
                self.cohorts_2_patients[c_idx]['patients'],
                self.cohorts_2_patients[c_idx]['labels'],
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            train_ids += temp1
            test_ids += temp2
            y_train += temp3
            y_test += temp4
            
        return train_ids, test_ids, y_train, y_test

    def per_cohort_train_val_test_split(self, test_size, val_size, random_state, test_stratify=False):
        """Do stratified train/val/test split for each cohort.
        """
        inner_train_ids, val_ids, test_ids, y_inner_train, y_val, y_test = [], [], [], [], [], []

        if test_stratify:
            stratify_labels = self.cohorts_2_patients[c_idx]['labels']
        else:
            stratify_labels = None

        for c_idx, c in enumerate(self.cohorts):                           
            X_train_c, X_test_c, y_train_c, y_test_c = model_selection.train_test_split(
                self.cohorts_2_patients[c_idx]['patients'],
                self.cohorts_2_patients[c_idx]['labels'],
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )

            X_train_c, X_val_c, y_train_c, y_val_c = model_selection.train_test_split(
                X_train_c,
                y_train_c,
                test_size=val_size,
                random_state=random_state,
                stratify=y_train_c
            )
    
            inner_train_ids += X_train_c
            val_ids += X_val_c
            test_ids += X_test_c
            y_inner_train += y_train_c
            y_val += y_val_c
            y_test += y_test_c
            
        return inner_train_ids, val_ids, test_ids, y_inner_train, y_val, y_test
