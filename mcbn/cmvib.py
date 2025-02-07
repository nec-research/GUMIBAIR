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

"""Conditional Multimodal Variational Information Bottleneck (CMVIB).

Implements MVIB (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010050)
with conditional batch normalization.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from collections import OrderedDict
from mcbn.cond_norm import CondBatchNorm1d, CondBatchNorm1dExtend
from typing import Tuple


class CMVIB(nn.Module):
    """Conditional Multimodal Variational Information Bottleneck.
    """
    def __init__(self, n_latents, abundance_dim, marker_dim, device, cond_dim, hidden_abundance, hidden_markers,
                 hidden_decoder, n_heads, p_drop=0.0, out_dim=1, run_mode='unimodal', conditioning_mode='single'):
        """
        :param n_latents: latent dimension
        :param abundance_dim: number of microbial species
        :param marker_dim: number of strain-level markers
        :param device: GPU or CPU for Torch
        :param cond_dim: dimension for the conditional batch norm
        :param hidden_abundance : a list for the hidden neurons of the abundance encoder MLP
        :param hidden_markers: a list for the hidden neurons of the marker encoder MLP
        :param hidden_decoder: a list for the hidden neurons of the decoder MLP
        :param n_heads: number of output heads for CMVIB
        :param p_drop: drop-out probability
        :param out_dim: output neurons
        :param run_mode: defines wether to use both data-modalities (and the PoE layer) or only the gene markers
        :param conditioning_mode: string definng wether to use CondBatchNorm1d (in 'single') or CondBatchNorm1dExtend (in 'multi')
        """
        super(CMVIB, self).__init__()

        self.abundance_encoder = CondEncoder(x_dim=abundance_dim, z_dim=n_latents, cond_dim=cond_dim,
                                             hidden_features_list=hidden_abundance, p_drop=p_drop, conditioning_mode=conditioning_mode)
        self.markers_encoder = CondEncoder(x_dim=marker_dim, z_dim=n_latents, cond_dim=cond_dim,
                                           hidden_features_list=hidden_markers, p_drop=p_drop, conditioning_mode=conditioning_mode)

        self.multihead_decoder = MultiHeadDecoder(in_features=n_latents, out_features=out_dim, n_heads=n_heads,
                                                  hidden_features_list=hidden_decoder, p_drop=p_drop)
            
        self.experts = ProductOfExperts()

        self.n_latents = n_latents
        self.abundance_dim = abundance_dim
        self.marker_dim = marker_dim
        self.device = device
        self.cond_dim = cond_dim
        self.hidden_abundance = hidden_abundance
        self.hidden_markers = hidden_markers
        self.hidden_decoder = hidden_decoder
        self.n_heads = n_heads
        self.p_drop = p_drop
        self.out_dim = out_dim
        self.run_mode = run_mode
        self.conditioning_mode = conditioning_mode

    def reparametrize(self, mu, logvar):
        """Reparameterization trick.
        Samples z from its posterior distribution.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, X):
        abundance, markers, X_head_idx, X_cond = X
        # infer joint posterior
        mu, logvar = self.infer(abundance, markers, X_cond)

        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # classification
        classification_logits = self.multihead_decoder((z, X_head_idx))

        return mu, logvar, classification_logits

    def infer(self, abundance=None, markers=None, X_cond=None):
        """Infer joint posterior q(z|x).
        """
        if self.run_mode == "multimodal":
            if abundance is not None:
                batch_size = abundance.size(0)
            elif markers is not None:
                batch_size = markers.size(0)
            else:
                batch_size = X_cond.size(0)

            # initialize the universal prior expert
            mu, logvar = self.prior_expert((1, batch_size, self.n_latents))
            if abundance is not None:
                a_mu, a_logvar = self.abundance_encoder((abundance, X_cond))
                mu = torch.cat((mu, a_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, a_logvar.unsqueeze(0)), dim=0)

            if markers is not None:
                m_mu, m_logvar = self.markers_encoder((markers, X_cond))
                mu = torch.cat((mu, m_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, m_logvar.unsqueeze(0)), dim=0)

            # product of experts to combine gaussians
            mu, logvar = self.experts(mu, logvar)
            return mu, logvar
        elif self.run_mode == "unimodal":
            mu, logvar = self.markers_encoder((markers, X_cond))
            return mu, logvar
        else:
            raise NotImplementedError

    def classify(self, X):
        """Classification - Compute p(y|x).
        """
        abundance, markers, X_head_idx, X_cond = X
        mu, logvar = self.infer(abundance, markers, X_cond)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        classification_logits = self.multihead_decoder((z, X_head_idx))
        prediction = torch.sigmoid(classification_logits)
        return prediction

    def prior_expert(self, size):
        """
        Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        :param size: dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        mu, logvar = mu.to(self.device), logvar.to(self.device)
        return mu, logvar


class TupleWrapper(nn.Module):
    """A simple wrapper for tuples.
    """
    def __init__(self, layer: nn.Module):
        super(TupleWrapper, self).__init__()
        self.layer = layer

    def forward(self, X: Tuple[torch.Tensor]):
        x, x_cond = X
        return self.layer(x), x_cond


class CondEncoder(nn.Module):
    """Parametrizes q(z|x).
    :param x_dim: input dimension
    :param z_dim: latent dimension
    """
    def __init__(self, x_dim, z_dim, cond_dim, hidden_features_list: list = [], p_drop: float = 0.0, conditioning_mode='single'):
        super(CondEncoder, self).__init__()

        in_dim = x_dim
        layers = []
        for i in hidden_features_list:
            layers.append(TupleWrapper(nn.Linear(in_dim, i)))
            layers.append(TupleWrapper(nn.ReLU()))
            if conditioning_mode == 'single':
                layers.append(CondBatchNorm1d(i, cond_dim))
            elif conditioning_mode == 'multi':
                layers.append(CondBatchNorm1dExtend(i, cond_dim))
            else:
                raise NotImplementedError("conditioning_mode has to be either 'single' or 'multi'")
            if p_drop > 0:
                layers.append(TupleWrapper(nn.Dropout(p=p_drop)))
            in_dim = i
        self.encoder = nn.Sequential(OrderedDict([
            (f"encoder_layer_{i}", l)
            for i, l in enumerate(layers)
        ]))
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def forward(self, x: Tuple[torch.Tensor]):
        h, _ = self.encoder(x)
        return self.mu(h), self.logvar(h)


class ProductOfExperts(nn.Module):
    """Compute parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def forward(self, mu, logvar, eps=1e-8):
        """
        :param mu: M x D for M experts
        :param logvar: M x D for M experts
        :param eps: constant for stability
        """
        # do computation in double to avoid nans
        mu = mu.double()
        logvar = logvar.double()

        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)

        # back to float
        pd_mu = pd_mu.float()
        pd_logvar = pd_logvar.float()

        return pd_mu, pd_logvar


class SelectiveLinear(nn.Module):
    """
    #TODO
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SelectiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((n_heads, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_heads,out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: tuple) -> tuple:
        X, X_head_idx = X
        weight_stack = torch.stack([self.weight[int(i.item()), :, :] for i in X_head_idx])
        out = torch.einsum("ij,ijk->ik", X, weight_stack)
        if self.bias is not None:
            bias_stack = torch.stack([self.bias[int(i.item()), :] for i in X_head_idx])
            out += bias_stack
        return out, X_head_idx


class MultiHeadDecoder(nn.Module):
    """A decoder which supports multiple output heads.
    In the microbiome context, heads can be e.g. disease-specific or cohort-specific.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, hidden_features_list: list = [],
                 p_drop: float = 0.0, bias=True):
        super(MultiHeadDecoder, self).__init__()
        in_dim = in_features
        layers = []
        drop = TupleWrapper(nn.Dropout(p=p_drop))
        for i in hidden_features_list:
            layers.append(SelectiveLinear(in_dim, i, n_heads, bias))
            layers.append(TupleWrapper(nn.ReLU()))
            if p_drop > 0:
                layers.append(drop)
            in_dim = i
        layers.append(SelectiveLinear(in_dim, out_features, n_heads, bias))
        self.decoder = nn.Sequential(OrderedDict([
            (f"decoder_layer_{i}", l)
            for i, l in enumerate(layers)
        ]))

    def forward(self, X: tuple) -> torch.Tensor:
        out, _ = self.decoder(X)
        return out

