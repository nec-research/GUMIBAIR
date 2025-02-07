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
from normalize_data import get_fnames
from functions import concatenate_batch

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path',
                        help='Path to directory with data')
    parser.add_argument('-d', '--data_modality',
                        default='abundance',
                        help="Data modality keyword. If set, will only consider filenames containing this word.")
    parser.add_argument('-sk','--skiprows',
                         type=int,
                         help='Number of rows to skip (Length of the metadata header, if present)')
    parser.add_argument('-rt', '--round-to',
                        type=int,
                        default=1,
                        help='Number of decimals that the abundace values are rounded to for duplicate checking')
    parser.add_argument('-s', '--save-output',
                        action='store_true',
                        default=False,
                        help='If set, sample IDs and abundance profiles will be saved as files')
    return parser.parse_args()

def import_data(path, skiprows=None, delimiter="\t"):
    data = pd.read_csv(path, delimiter=delimiter, dtype=str)
    if skiprows:
        data.drop(data.index[:skiprows], inplace=True)
    return data

def convert_to_binary(df):
    """
    Convert the relative abundance values to a binary indicator.
    :param df: dataframe with relative abundance values
    :return df: dataframe with binary indicators for species presence
    """
    df = df.astype(float)
    df = df.mask(df>0,1)
    return df

def convert_to_string(df):
    """
    Convert the relative abundance values to a string and only keep
    the integer values
    :param df: dataframe with relative abundance values
    :return df: dataframe with only integers
    """
    df = df.astype(str)
    for col in df:
        df[col] = (df[col].str.split('.').str[0])
    return df

def check_duplicates(df):
    """
    find columns that have the exact same content.
    Extract column names (sample IDs)
    :param data: dataframe containing either abundance or gene markers
    :return: list of sample IDs that are duplicated within the data.
    """
    df_T = df.T
    # create a sort column composed of the first 10 columns
    df_T.loc[:, 'sort_col'] = df_T.loc[:, df_T.columns[0:10].tolist()].astype(str).agg(''.join, axis=1)

    duplicates = df_T[df_T.duplicated(keep=False)].sort_values('sort_col')
    duplicates.drop(columns=['sort_col'], inplace=True)
    return duplicates.index.tolist()

def save_duplicates(data, sample_ids, output_name):
    """
    Safe duplicate IDs to .txt file as well as abundance profiles for
    corresponding samples.
    :param data: abundance dataframe
    :param sample_ids: list of duplicate IDs
    :param output_name: path and name to output file
    """
    data.loc[:, sample_ids].to_csv(output_name, sep="\t", index=False)

    sample_ID_file = output_name.replace('.txt', '_IDs.txt')
    with open(sample_ID_file, 'w') as tsv:
        tsv.write("\t".join(str(sample_id) for sample_id in sample_ids))
         
def main(args):
    fnames = get_fnames(args.path, args.data_modality)
    data = dict()
    for fname in fnames:
        df = import_data(fname, args.skiprows)
        data[fname] = df

    merged_data = concatenate_batch(data.values(), "Unnamed: 0", keep_merge_col=False).astype(float)

    abundance_duplicates = check_duplicates(merged_data.round(args.round_to))
    binary_merged_data = convert_to_binary(merged_data)
    presence_duplicates = check_duplicates(binary_merged_data)

    integer_merged_data = convert_to_string(merged_data)
    integer_duplicates = check_duplicates(integer_merged_data)

    if not abundance_duplicates and not presence_duplicates:
        print("No duplicates found in the batch!")
    else:
        print("Following sample IDs seem to be duplicates of other samples in the batch:\n")
        print(f"""Duplicates based on species relative abundance:\n
              {abundance_duplicates}\n
              Duplicates based on species presence:\n
              {presence_duplicates}\n""")
        print("Consider dropping those samples!")
    
    if args.save_output:
        output_path = f"{args.path}duplicates/"
        print(f"Output is being saved to {output_path}")
        if not os.path.exists(output_path):
            Path(output_path).mkdir()

        save_duplicates(merged_data, abundance_duplicates, f"{output_path}abundance.txt")
        save_duplicates(merged_data, presence_duplicates, f"{output_path}presence.txt")
        save_duplicates(merged_data, integer_duplicates, f"{output_path}integer.txt")
        
if __name__ == "__main__":
    description = "Python script with checks to run on all output from curatedMetagenomicData"
    args = parse_args(description)
    main(args)