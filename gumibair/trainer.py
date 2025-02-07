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

"""Trainer class.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import functools
import torch
import numpy as np
import timeit
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from gumibair.cmvib import CMVIB


class Trainer:
    """A class to automate training.
    It stores the weights of the model at the epoch where a validation score is maximized.
    """
    def __init__(
            self, model, epochs, lr, beta, checkpoint_dir, monitor, device, log=False, log_dir='./',
            test_set: tuple = None
    ):
        """
        :param model: Torch MVIB model object
        :param epochs: max training epochs
        :param lr: learning rate
        :param beta: multiplier for KL divergence
        :param checkpoint_dir: directory for saving model checkpoints
        :param monitor: `min` minimize loss; `max` maximize ROC AUC
        :param log: boolean deciding, if the Val Loss and Val ROC AUC should be logged
        :param log_dir: location of the logfiles for Val Loss and Val ROC AUC
        :param test_set: tuple with abundance, marker, heads and conditions of the test samples
        """
        self.model = model
        self.epochs = epochs
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.logits_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.triplet_loss = torch.nn.TripletMarginLoss()
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, monitor, patience=10, cooldown=10, factor=0.5)

        self.log = log
        self.log_dir = log_dir
        if self.log:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.test_set = test_set

    def _log_training(train_func):
        """
        A conditional decorator wrapping the train() method when self.log is True.
        Basically flushes and closes the SummaryWriter(), after all epochs have been completed.
        """
        @functools.wraps(train_func)
        def wrapper(self, *args, **kwargs):
            state = train_func(self, *args, **kwargs)
            if self.log:
                self.writer.flush()
                self.writer.close()
                return state
            else:
                return state
        return wrapper
    
    def _write_output(func):
        """
        A conditinoal decorator wrapping the evaluate() method when self.log is True.
        Adds the loss and auROC to the writer.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            loss, roc_auc, logging_dict = func(self, *args, **kwargs)
            if self.log:
                epoch = logging_dict['epoch']
                for key, value in logging_dict.items():
                    if not key == 'epoch':
                        self.writer.add_scalar(key, value, epoch)
                if self.test_set:
                    test_roc_auc = self._test_while_training(self.test_set)
                    self.writer.add_scalar("test_auc", test_roc_auc, epoch)
            return loss, roc_auc, None
        return wrapper

    def _test_while_training(self, test_set) -> float:
        """
        A method that allows to predict on the test set during training.
        The main purpose is to monitor the test performance during training,
        to check wether it makes sense to stop early.
        :param test_set: tuple with inputs, ground_truth, heads and conditions
        :return: float of test_auc of the current epoch.
        """
        test_abundance, test_markers, test_gt, test_cond, test_heads = test_set

        test_abundance = test_abundance.to(self.device)
        test_markers = test_markers.to(self.device)
        test_gt = test_gt.to(self.device)
        test_cond = test_cond.to(self.device)
        test_heads = test_heads.to(self.device)

        self.model.eval()

        test_predictions = self.model.classify((test_abundance, test_markers, test_heads, test_cond))


        test_roc_auc = roc_auc_score(
            test_gt.cpu().detach().numpy().squeeze().tolist(), 
            test_predictions.cpu().detach().numpy().squeeze().tolist()
        )
        
        return test_roc_auc

    @_log_training
    def train(self, train_loader, val_loader, verbose=True):
        """The main training loop.

        :param train_loader: the training Torch data loader
        :param val_loader: the validation Torch data loader
        """
        if self.monitor == 'max':
            best_val_score = 0
        else:
            best_val_score = float('inf')
        is_best = False
        state = None

        t_0 = timeit.default_timer()

        for epoch in range(1, self.epochs + 1):
            self.train_step(train_loader, epoch)
            val_loss, val_roc_auc, _ = self.evaluate(val_loader, epoch)
            if self.monitor == 'max':
                self.scheduler.step(val_roc_auc)
                is_best = val_roc_auc > best_val_score
                best_val_score = max(val_roc_auc, best_val_score)
            elif self.monitor == 'min':
                self.scheduler.step(val_loss)
                is_best = val_loss < best_val_score
                best_val_score = min(val_loss, best_val_score)

            if is_best or state is None:
                if isinstance(self.model, CMVIB):
                    state = {
                        'state_dict': self.model.state_dict(),
                        'best_val_score': best_val_score,
                        'n_latents': self.model.n_latents,
                        'abundance_dim': self.model.abundance_dim,
                        'marker_dim': self.model.marker_dim,
                        'device': self.model.device,
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'cond_dim': self.model.cond_dim,
                        'hidden_abundance': self.model.hidden_abundance,
                        'hidden_markers': self.model.hidden_markers,
                        'hidden_decoder': self.model.hidden_decoder,
                        'n_heads': self.model.n_heads,
                        'p_drop': self.model.p_drop,
                        'out_dim': self.model.out_dim,
                        'run_mode': self.model.run_mode,
                        'conditioning_mode': self.model.conditioning_mode
                    }
                elif isinstance(self.model, torch.nn.DataParallel):
                    state = {
                        'state_dict': self.model.module.state_dict(),
                        'best_val_score': best_val_score,
                        'n_latents': self.model.module.n_latents,
                        'abundance_dim': self.model.module.abundance_dim,
                        'marker_dim': self.model.module.marker_dim,
                        'device': self.model.module.device,
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'cond_dim': self.model.module.cond_dim,
                        'hidden_abundance': self.model.module.hidden_abundance,
                        'hidden_markers': self.model.module.hidden_markers,
                        'hidden_decoder': self.model.module.hidden_decoder,
                        'n_heads': self.model.module.n_heads,
                        'p_drop': self.model.module.p_drop,
                        'out_dim': self.model.module.out_dim,
                        'run_mode': self.model.module.run_mode,
                        'conditioning_mode': self.model.module.conditioning_mode
                    }
                else:
                    raise NotImplementedError
                
                if verbose:
                    print('====> Epoch: {} | DeltaT: {} | Val score: {:.4f} | Best model'.format(
                        state['epoch'], timeit.default_timer() - t_0, best_val_score)
                    )
        return state


    @_write_output
    def train_step(self, train_loader, epoch):
        """Compute loss and do backpropagation for a single epoch.
        """
        self.model.train()
        train_loss_meter = AverageMeter()
        gt_stack = np.array([])
        prediction_stack = np.array([])

        if isinstance(self.model, CMVIB):
            run_mode = self.model.run_mode
        elif isinstance(self.model, torch.nn.DataParallel):
            run_mode = self.model.module.run_mode

        for batch_idx, (abundance, markers, gt, cond, heads) in enumerate(train_loader):
            #abundance = Variable(abundance)
            #markers = Variable(markers)
            #cond = Variable(cond)
            #heads = Variable(heads)

            abundance = Variable(abundance).to(self.device, non_blocking=True)
            markers = Variable(markers).to(self.device, non_blocking=True)
            cond = Variable(cond).to(self.device, non_blocking=True)
            heads = Variable(heads).to(self.device, non_blocking=True)
            gt = Variable(gt).to(self.device, non_blocking=True)
            batch_size = len(abundance)

            # refresh the optimizer
            self.optimizer.zero_grad()

            # pass data through model
            if run_mode == "multimodal":
                mu_1, logvar_1, cls_logits_1 = self.model((abundance, markers, heads, cond))
                mu_2, logvar_2, cls_logits_2 = self.model((abundance, None, heads, cond))
                mu_3, logvar_3, cls_logits_3 = self.model((None, markers, heads, cond))

                # compute KL divergence for each data combination
                kld = self.KL_loss(
                    mu_1, logvar_1, self.beta
                )

                kld += self.KL_loss(
                    mu_2, logvar_2, self.beta
                )

                kld += self.KL_loss(
                    mu_3, logvar_3, self.beta
                )

                # compute binary cross entropy for each data combination
                classification_loss = self.logits_bce(cls_logits_1, gt) + \
                                    self.logits_bce(cls_logits_2, gt) + \
                                    self.logits_bce(cls_logits_3, gt)
                classification_loss = torch.mean(classification_loss)
                train_loss = kld + classification_loss

                train_loss_meter.update(train_loss.item(), batch_size)

                # compute gradients and take step
                train_loss.backward()
                self.optimizer.step()
            elif run_mode == "unimodal":
                mu, logvar, cls_logits_1 = self.model((abundance, markers, heads, cond))
                kld = self.KL_loss(
                    mu, logvar, self.beta
                )
                classification_loss = self.logits_bce(cls_logits_1, gt)
                classification_loss = torch.mean(classification_loss)
                train_loss = kld + classification_loss
                train_loss_meter.update(train_loss.item(), batch_size)
                train_loss.backward()
                self.optimizer.step()
            else:
                raise NotImplementedError
            
            gt_stack = np.concatenate(
                (gt_stack, gt.cpu().numpy().squeeze().astype('int')),
                axis=0)
            prediction_stack = np.concatenate(
                (prediction_stack, torch.sigmoid(cls_logits_1).detach().cpu().numpy().squeeze()),
                axis=0)
            
        train_auc_roc = roc_auc_score(gt_stack, np.nan_to_num(prediction_stack))
        train_accuracy = accuracy_score(gt_stack, np.nan_to_num(prediction_stack).round())

        logging_dict = {
            "epoch":epoch,
            "train_loss":train_loss_meter.avg,
            "train_roc_auc": train_auc_roc,
            "train_kld": kld,
            "train_bce": classification_loss,
            "train_accuracy": train_accuracy
        }

        return None, None, logging_dict

    @_write_output
    def evaluate(self, val_loader, epoch):
        """Evaluate model performance on validation (or test) set.
        """
        self.model.eval()
        val_loss_meter = AverageMeter()
        gt_stack = np.array([])
        prediction_stack = np.array([])

        if isinstance(self.model, CMVIB):
            run_mode = self.model.run_mode
        elif isinstance(self.model, torch.nn.DataParallel):
            run_mode = self.model.module.run_mode

        with torch.no_grad():
            for batch_idx, (abundance, markers, gt, cond, heads) in enumerate(val_loader):
                #abundance = Variable(abundance)
                #markers = Variable(markers)
                #cond = Variable(cond)
                #heads = Variable(heads)
                
                abundance = Variable(abundance).to(self.device)
                markers = Variable(markers).to(self.device)
                cond = Variable(cond).to(self.device)
                heads = Variable(heads).to(self.device)
                gt = Variable(gt).to(self.device)
                batch_size = len(abundance)

                if run_mode == "multimodal":
                    # pass data through model
                    mu_1, logvar_1, cls_logits_1 = self.model((abundance, markers, heads, cond))
                    mu_2, logvar_2, cls_logits_2 = self.model((abundance, None, heads, cond))
                    mu_3, logvar_3, cls_logits_3 = self.model((None, markers, heads, cond))

                    # compute KL divergence for each data combination
                    kld = self.KL_loss(
                        mu_1, logvar_1, self.beta
                    )

                    kld += self.KL_loss(
                        mu_2, logvar_2, self.beta
                    )

                    kld += self.KL_loss(
                        mu_3, logvar_3, self.beta
                    )

                    # compute binary cross entropy for each data combination
                    classification_loss = self.logits_bce(cls_logits_1, gt) + \
                                        self.logits_bce(cls_logits_2, gt) + \
                                        self.logits_bce(cls_logits_3, gt)
                    classification_loss = torch.mean(classification_loss)
                    val_loss = kld + classification_loss
                    val_loss_meter.update(val_loss.item(), batch_size)


                elif run_mode == "unimodal":
                    mu_1, logvar_1, cls_logits_1 = self.model((abundance, markers, heads, cond))
                                         # compute KL divergence for each data combination
                    kld = self.KL_loss(
                        mu_1, logvar_1, self.beta
                    )

                    # compute binary cross entropy for each data combination
                    classification_loss = self.logits_bce(cls_logits_1, gt)
                    classification_loss = torch.mean(classification_loss)
                    val_loss = kld + classification_loss
                    val_loss_meter.update(val_loss.item(), batch_size)
                else:
                    raise NotImplementedError

                gt_stack = np.concatenate(
                    (gt_stack, gt.cpu().numpy().squeeze().astype('int')),
                    axis=0)
                prediction_stack = np.concatenate(
                    (prediction_stack, torch.sigmoid(cls_logits_1).detach().cpu().numpy().squeeze()),
                    axis=0)

        val_roc_auc = roc_auc_score(gt_stack, np.nan_to_num(prediction_stack))
        val_accuracy = accuracy_score(gt_stack, np.nan_to_num(prediction_stack).round())

        logging_dict = {
            "epoch":epoch,
            "val_loss":val_loss_meter.avg,
            "val_roc_auc": val_roc_auc,
            "val_kld": kld,
            "val_bce": classification_loss,
            "val_accuracy": val_accuracy
        }

        return val_loss_meter.avg, val_roc_auc, logging_dict

    def KL_loss(self, mu, logvar, beta=1):
        """Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        """
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD = torch.mean(beta * KLD)
        return KLD

    @staticmethod
    def save_checkpoint(state, folder='./', filename='model_best.pth.tar'):
        """Save model checkpoints.
        """
        if not os.path.isdir(folder):
            os.mkdir(folder)
        print('Saving best model: epoch {}'.format(state['epoch']))
        torch.save(state, os.path.join(folder, filename))

    @staticmethod
    def load_checkpoint(checkpoint):
        """Load model checkpoints.
        """
        model = CMVIB(
            checkpoint['n_latents'],
            checkpoint['abundance_dim'],
            checkpoint['marker_dim'],
            checkpoint['device'],
            checkpoint['cond_dim'],
            checkpoint['hidden_abundance'],
            checkpoint['hidden_markers'],
            checkpoint['hidden_decoder'],
            checkpoint['n_heads'],
            checkpoint['p_drop'],
            checkpoint['out_dim'],
            run_mode=checkpoint['run_mode'],
            conditioning_mode=checkpoint['conditioning_mode']
        )
        model.load_state_dict(checkpoint['state_dict'])

        return model, checkpoint['best_val_score']


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
