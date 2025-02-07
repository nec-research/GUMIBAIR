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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_color_palette(
        color: str,
        start_idx: int = 0,
        num_paired_colors: int = 2,
        step: int = 2) -> list:
    """
    :param color: str representing the base color. has to be included in `sns.color_palette()`.
    :param num_paired_colors: number of colors that should be paired for each model.
    :param step: int defining the distance between the colors in the original palette.
    :return: list of hexcodes with colors to be used for a particular model.
    """
    input_palette = list(sns.color_palette(color))
    input_palette.reverse()

    custom_palette = input_palette[0+start_idx:(num_paired_colors+start_idx)*step:step]
    return custom_palette

def subset_metric(dataset: pd.DataFrame, metric: str = "auROC") -> pd.DataFrame:
    """
    Function to subset the imported results with the goal to visualize a particular metric.
    :param dataset: pd.DataFrame with total results of a particular experiment.
    :param metric:
    :return: pd.DataFrame with subset only containing the metric of interest
    """
    metric_subset = dataset[dataset['metrics'] == metric].copy()
    metric_subset['scores'] = metric_subset['scores'].astype(float)
    return metric_subset

def get_cohort_info(path: str) -> dict:
    """
    Function to get the cohort info (name and len) from a file.
    :param path: string with absolute path to the file containing the info.
    """
    cohort_info = dict()
    with open(path, 'r') as fname:
        reader = csv.reader(fname)
        for row in reader:
            cohort_info[row[0]] = int(row[1])

    return cohort_info

def create_labels(cohort_info: dict) -> list:
    """
    Function to create the per cohort labels for the plots.
    Each label is a string of format 'Cohort_Name Samples: Cohort_Size'. Labels are sorted by cohort size.
    :param cohort_info: dict with cohort name as key and cohort size as value.
    :return: list of formatted labels.
    """
    cohorts_sorted  = {cohort:samples for cohort, samples in sorted(cohort_info.items(), key=lambda item: item[1])}

    labels = [f"{cohort}\n Samples: {cohorts_sorted[cohort]}" for cohort in cohorts_sorted.keys()]
    return labels, cohorts_sorted

def create_barplot(
        metric_subset: pd.DataFrame,
        order: list,
        labels: list,
        x: str = "cohort",
        hue: str = "model",
        palette: list = list(sns.color_palette()),
        rotation: int = 15,
        ylim: list = [0.3, 1.05],
        width: float = 0.8,
        dodge: bool = True,
        **kwargs,
        ):
    """
    Function to create a sns.barplot for a particular result based on a config dict.
    :param metric_subset: pd.DataFrame containing the data to be plotted.
    :param order: list with ordered values of the variable to be placed on the x axis.
    :param labels: list with formatted x-tick labels
    :param x: string referring to the variable to be put on the x-axis
    :param hue: string referring to the variable to color multiple bars by.
    :param palette: list with RGB tuples of custom colors.
    :param rotation: int referring to the degrees by which the x-tick labels should be rotated.
    :param ylim: list with float limits for the y-axis.
    """
    plt.figure()

    ax = sns.barplot(
        data=metric_subset,
        x=x,
        y="scores",
        palette=palette,
        hue=hue,
        order=order,
        width=width,
        dodge=dodge
    )

    ax.set_ylim(ylim)
    ax.grid(axis='y')
    ax.set_title(kwargs.get('title'))
    ax.set_xticklabels(labels, rotation=rotation)
    plt.ylabel(kwargs.get('ylabel'))
    plt.xlabel(kwargs.get('xlabel'))

    return ax

def plot_parallel_coordinates(
        df: pd.DataFrame,
        categorical_params: list = [],
        metric:str = 'auROC', 
        colorscale=px.colors.diverging.RdBu,
        title: str = ""):
    """
    Function to create a parallel coordinates plot from a dataframe with HPO results.
    """
    dimensions = list()
    for param in df.columns:
        if param in categorical_params:
            unique_dict = {value:i for i, value in enumerate(df[param].unique().tolist())}
            values = [unique_dict[value] for value in df[param].tolist()]
            dimension = dict(
                range=[0, len(unique_dict.values())],
                tickvals=list(range(0, len(unique_dict))),
                label=param,
                values=values,
                ticktext=list(unique_dict.keys())
            )
            dimensions.append(dimension)

        elif not param == metric:
            dimension = dict(
                range=[df[param].min(), df[param].max()],
                values=df[param].tolist(),
                label=param
            )
            dimensions.append(dimension)
                
    plot = go.Figure(
        data=go.Parcoords(
            line= dict(
                color=df[metric],
                colorscale='RdBu',
                showscale=True,
                cmin=df[metric].min(),
                cmax=df[metric].max()
            ),
            dimensions=dimensions
        ),
        layout=go.Layout(
            title=go.layout.Title(text=title)
        )
    )

    return plot