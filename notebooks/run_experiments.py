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
from gumibair_experiments.in_cohort_exp import InCohortExperiment
from gumibair_experiments.cross_cohort_exp import CrossCohortExperiment
from gumibair_experiments.utils import load_config
import torch
import pandas as pd
import os
from argparse import ArgumentParser

def parse_args(description: str) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument('experiment_type',
                        choices=['in_cohort', 'cross_cohort'],
                        help="Type of experiment. Either 'in_cohort' or 'cross_cohort'")
    parser.add_argument('config_file',
                        help='Path to config file to be used')
    parser.add_argument('data_dir',
                        help='Location of the data to be used')
    parser.add_argument('-o', '--output-path',
                        type=str,
                        default='./',
                        help="Path to output location")
    parser.add_argument('-t', '--test-prop',
                        type=str,
                        default='0.9',
                        help="Proportion of held-out cohort to be included in testing for partial holdout")
    parser.add_argument('-tc', '--test-condition',
                        help="""can be set to either `validation_set` (only makes sense with 'partial') or `mean` ('only makes sense with complete'). 
                        If not set, the condition of the test set stays unchanged.""")
    parser.add_argument('-m', '--mode',
                        default='partial',
                        choices=['partial', 'complete'],
                        help="""
                        Option to choose training mode for cross-cohort experiments.
                        Either 'partial' or 'complete'
                        """)
    parser.add_argument('-a', '--all-cohorts',
                        default=False,
                        action='store_true',
                        help="""Only for CrossCohortExperiment. 
                        If set, the models are trained on all cohorts + the proportion of samples from the heldout cohort contained in training""")
    parser.add_argument('-r', '--replicates',
                        type=int,
                        default=5,
                        help="Specifies the number of replicates that should be run per experiment.")
    parser.add_argument('-b', '--benchmark',
                        default=False,
                        action='store_true',
                        help="If set, experiment will be benchnmarked against MVIB and RF"
                        )
    parser.add_argument('-c', '--cross-validate',
                        default=False,
                        action='store_true',
                        help="""For InCohorExperiment: If set, each replicate runs in a 5-fold CV scheme, creating 5 different train and test splits.
                        For CrossCohortExperiment: If set, each replicate runs with each cohort in training being the validation set once.""")
    parser.add_argument('-mt', '--monitor-test-auc',
                        default=False,
                        action='store_true',
                        help="""If set, the model is evaluated on the test set at every epoch and the AUC is logged to the tensorboard.""")

    return parser.parse_args()

def import_data(data_dir, config_file: str) -> tuple:
    """
    Function to parse config from yaml file and import the data.
    :param config_file: str with path to config file to be used.
    :return: tuple with config dictionary and FullMicrobiomeDataset object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_config = load_config(config_file)
    COHORTS_TO_DISEASE = base_config['cohorts_to_disease']
    COHORTS = tuple(COHORTS_TO_DISEASE.keys())

    fmd = FullMicrobiomeDataset(
        data_dir,
        device=device,
        label_dict=base_config.get('label_dict'),
        data="joint",
        cohorts=COHORTS, 
        conditioning_type=base_config['conditioning_type'],
        conditioning_labels=base_config.get('conditioning_labels'),
        cohorts_to_disease=COHORTS_TO_DISEASE,
        drop_samples=base_config.get('drop_samples'),
        scale_markers=True
    )

    config = {
        'abundance_dim':len(fmd[0][0]),
        'marker_dim':len(fmd[0][1]),
        'cond_dim':len(fmd[0][3]),
        'device': device,
    }

    for key, value in base_config.items():
        config[key] = value

    return config, fmd

def init_experiment(parsed_args, config: dict, fmd: FullMicrobiomeDataset, **kwargs):
    """
    Function to instantiate the experiment object according to config file and arguments from the commandline.
    :param args: parsed arguments
    :param config: dict with configs from yaml file
    :param fmd: FullMicrobiomeDataset object to be used for the experiment.
    """
    if parsed_args.experiment_type == "in_cohort":
        experiment = InCohortExperiment(
            fmd,
            config,
            replicates=parsed_args.replicates,
            benchmark=parsed_args.benchmark,
            cross_validation=parsed_args.cross_validate,
            monitor_test_auc=parsed_args.monitor_test_auc
        )
    else:
        experiment = CrossCohortExperiment(
            fmd,
            config,
            mode=parsed_args.mode,
            replicates=parsed_args.replicates,
            benchmark=parsed_args.benchmark,
            all_cohorts=parsed_args.all_cohorts,
            test_prop=kwargs.get('test_prop'),
            test_condition=parsed_args.test_condition,
            cross_validate=parsed_args.cross_validate,
            monitor_test_auc=parsed_args.monitor_test_auc,
        )

    return experiment

def write_output(results_df: pd.DataFrame, fname: str):
    """
    Function to write results to csv file.
    Tries saving the file to the specified output path.
    If it doesn't work, the file is saved in the current working directory.
    """
    try:
        results_df.to_csv(fname, mode='a', index=False)
    except FileNotFoundError:
        fname = fname.rsplit('/', 1)[-1]
        results_df.to_csv(fname, mode='a', index=False)

def main(description: str):
    """
    Main execution of the experiment.
    Includes initializing the object, executing the experiment and parsing & saving the results.
    For in_cohort, a condensed result as well as a per-cohort result file is saved.
    For cross_cohort, only the condensed result is created directly from the metrics that are returned from the experiment object.
    (since the test-set contains only one cohort anyways.)
    """
    args = parse_args(description)

    if args.benchmark:
        template_string = "_results_GB_RF_MVIB_"
    else:
        template_string = "_results_GB_"

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    config, fmd = import_data(args.data_dir, args.config_file)
    
    if args.experiment_type == 'in_cohort':
        experiment = init_experiment(args, config, fmd)
        results = experiment.run_replicates()
        condensed_scores, per_cohort_scores = results
        condensed_scores_df = pd.concat(condensed_scores)
        per_cohotr_scores_df = pd.concat(per_cohort_scores)

        condensed_name = f"{args.output_path}{args.experiment_type}{template_string}condensed.csv"
        write_output(condensed_scores_df, condensed_name)

        per_cohort_name = f"{args.output_path}{args.experiment_type}{template_string}per_cohort.csv"
        write_output(per_cohotr_scores_df, per_cohort_name)

    elif args.experiment_type == 'cross_cohort':
        if args.mode == "partial":
            for prop in args.test_prop.split(','):
                print(f"\nTest Proportion: {prop}")
                experiment = init_experiment(args, config, fmd, test_prop=float(prop))
                results = experiment.run_replicates()
                condensed_scores_df = pd.concat(results)

                condensed_name = f"{args.output_path}{args.experiment_type}{template_string}condensed.csv"
                write_output(condensed_scores_df, condensed_name)
        else:
            experiment = init_experiment(args, config, fmd)
            results = experiment.run_replicates()
            condensed_scores_df = pd.concat(results)

            condensed_name = f"{args.output_path}{args.experiment_type}{template_string}condensed.csv"
            write_output(condensed_scores_df, condensed_name)
    else:
        raise NotImplementedError

    print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    description="""Python script to run experiments. 
    Can be configured to run in-cohort or cross-cohort. 
    Either only with GUMIBAIR or GUMIBAIR benchmarked against RF and MVIB"""
    main(description)