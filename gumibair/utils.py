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

"""Utilities.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, precision_recall_curve, auc



_METRICS = [
    'auROC',
    'accuracy',
    'f1',
    'auPR',
    'precision',
    'recall',
]


def save_results(disease_dir, val_mean, test_mean, test_std, args):
    """Save the overall results for all diseases and all experiments.
    """
    results = {
        'val_mean': val_mean,
        'test_mean': test_mean,
        'test_std': test_std,
    }
    results = {**results, **vars(args)}

    with open(os.path.join(disease_dir, 'overall_results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


def save_experiment(downstream_metrics, val_mean, test_mean, disease, result_dir, run_id, tuning_id):
    """Save the the results of one experiment for one disease.
    """
    result_dir = os.path.join(result_dir, run_id, disease, tuning_id)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    downstream_metrics['val'] = val_mean
    downstream_metrics['test'] = test_mean
    with open(os.path.join(result_dir, 'experiment.json'), 'w') as fp:
        json.dump(downstream_metrics, fp, indent=4)


def save_latent(experiment_dir, mu, logvar, y_test, test_ids, fold):
    """Save latent space embeddings and labels.
    """
    np.save(os.path.join(experiment_dir, 'mu_{}'.format(fold)), mu[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'logvar_{}'.format(fold)), logvar[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'label_{}'.format(fold)), y_test)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """Compute AUPR.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_scores(y_true, y_prob, y_pred):
    """Compute a dictionary with all classification metrics and respective scores.
    """
    scores = [
        roc_auc_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred)
    ]

    scores_df = pd.DataFrame(data={
    'scores': scores,
    'metrics': _METRICS
    })
    
    return scores_df


def per_cohort_metrics(cohorts, cohorts_2_disease, predictions, test_gt, test_ids, fmd, exp):
    """Compute the metrics for each cohort (or disease).
    """
    per_cohort_results = {}
    patients_2_cohort = {}
    for c_idx, c in enumerate(cohorts):
        for p in fmd.cohorts_2_patients[c_idx]['patients']:
            patients_2_cohort[p] = c_idx

    results_df = pd.DataFrame(data={
        'predictions': predictions.cpu().detach().numpy().squeeze().tolist(),
        'gt': test_gt.cpu().detach().int().numpy().squeeze().tolist(),
        'patient_id': test_ids,
    })
    results_df['cohort'] = results_df['patient_id'].apply(lambda x: cohorts[patients_2_cohort[x]])
    results_df['disease'] = results_df['patient_id'].apply(lambda x: cohorts_2_disease[cohorts[patients_2_cohort[x]]])
    
    metrics = []
    for c in cohorts:
        tmp = results_df[results_df['cohort']==c]
        scores_df = get_scores(tmp['gt'], tmp['predictions'], tmp['predictions'].round())
        scores_df['cohort'] = c
        scores_df['exp'] = exp
        metrics.append(scores_df)

    return pd.concat(metrics)