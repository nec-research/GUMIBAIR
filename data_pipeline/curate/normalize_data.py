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

import argparse
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import shutil
from functions import get_fnames

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path',
                        help='Path to directory with data')
    parser.add_argument('-d', '--data_modality', 
                        help="Data modality keyword. If set, will only consider filenames containing this word.")
    parser.add_argument('-s', '--skiprows', 
                        type=int,
                        help="Number of rows to skip (Length of the metadata header, if present)")
    
    return parser.parse_args()

def check_for_existing_output(fnames, data_modality):
    """
    Check if desired output file exists already.
    If so, use that file to renormalize with new studies.
    """
    old_output = list()
    for i, fname in enumerate(fnames):
        output_name = fname.replace(data_modality, f"normalized/{args.data_modality}_normalized")
        if os.path.isfile(output_name):
            old_output.append(output_name)
            fnames[i] = output_name
    return old_output

def import_data(fname):
    """
    Import data.
    """
    df = pd.read_csv(fname, sep="\t", dtype=str)
    return df

def get_unique_index(data, start_range=None, end_range=None):
    """
    Get list of unique index values shared between all files.
    :param data: dictionary with file locations as keys and dataframe as values
    :param start_range: startpoint for slice operation
    :param end_range: endpoint for slice operation
    :return unique_index: list of all abundance profiles or marker IDs
    """
    total_index = list()
    for df in data.values():
        index = df.iloc[start_range:end_range, 0].tolist()
        total_index.extend(index)
    unique_index = list(set(total_index))

    return unique_index

def normalize_data(df, unique_index):
    """
    Normalize dataset rows based unique index.
    :param df: dataframe containing individual dataset
    :param unique_index: list of all abundance profiles or marker IDs
    :return normalized_df: normalized dataset with additional zero-rows
    """
    df_index = df.iloc[:, 0]
    additional_items = list(set(unique_index) - set(df_index))
    additional_rows = list()
    for item in additional_items:
        row = [item]
        row.extend([0 for i in range(len(df.columns) - 1)])
        additional_rows.append(row)
    additional_data = pd.DataFrame(additional_rows, columns=df.columns)
    normalized_df = pd.concat([df, additional_data], axis=0)
    normalized_df.reset_index(drop=True, inplace=True)
    return normalized_df

def sort_df_subset(df, skiprows):
    """
    Sort part of the dataframe containing numerical rows (exclude metadata)
    :param df: individual dataset
    :param skiprows: number of rows to be skipped (length of metadata header, if present)
    :return:
    """
    header = df.iloc[:skiprows].copy()
    data = df.iloc[skiprows:].copy()
    data.sort_values('Unnamed: 0', inplace=True)    
    df = pd.concat([header, data], ignore_index=True)
    return df
    
def test_success(comparing_indices):
    """
    Test if normalization was successfull.
    :param comparing indices: lists of index columns from all datasets
    """
    if all(x == comparing_indices[0] for x in comparing_indices):
        print("Data successfully normalized!")
    else:
        print("There was a problem, please check the data!")
    
def main(args):
    fnames = get_fnames(args.path, args.data_modality)
    old_output = check_for_existing_output(fnames, args.data_modality)

    data = dict()
    print("Importing Data\n")
    for fname in tqdm(fnames):
        df = import_data(fname)
        data[fname] = df

    unique_index = get_unique_index(data, start_range=args.skiprows)

    print("\nNormalizing Data\n")
    comparing_indices = list()
    for fname, df in tqdm(data.items()):
        normalized_df = normalize_data(df, unique_index)

        output_path = args.path.replace(args.path, f"{args.path}normalized/")
        if not os.path.exists(output_path):
            Path(f"{output_path}old/").mkdir(parents=True, exist_ok=True)

        if not fname in old_output:
            output_name = fname.rsplit("/")[-1].replace(args.data_modality, f"{args.data_modality}_normalized")
            filename = f"{output_path}{output_name}"
        else:
            filename = fname
            fname_new = fname.replace(".txt", f"_{datetime.today().strftime('%Y-%m-%d')}.txt")
            fname_new = fname_new.replace(output_path, f"{output_path}old/")
            shutil.move(fname, fname_new)

        sorted_df = sort_df_subset(normalized_df, args.skiprows)
        sorted_df.to_csv(filename, index=False, sep="\t")
        comparing_indices.append(sorted_df.iloc[args.skiprows:, 0].tolist())

    test_success(comparing_indices)


if __name__ == "__main__":
    description="""
    Python script to find the union of all cohorts for a specific data modality. 
    Either gene marker IDs or species of abundance profiles.
    Sets all files to that union (i.e. adds zero rows back in again).
    """
    args = parse_args(description)
    main(args)