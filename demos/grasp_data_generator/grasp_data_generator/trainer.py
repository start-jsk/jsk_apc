import collections
import copy
import os
import os.path as osp
import time

import chainer
import fcn.datasets
import fcn.utils
import numpy as np
import skimage.io
import skimage.util
import tqdm

from. import utils


class Trainer(object):

    def __init__(
            self, device, model,
            optimizer, iter_train, iter_valid,
            out, max_iter, interval_validate=4000
    ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.iter_train = iter_train
        self.iter_valid = iter_valid
        self.out = out
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.interval_validate = interval_validate
        self.stamp_start = None

    def _write_log(self, **kwargs):
        log = collections.defaultdict(str)
        log.update(kwargs)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            f.write(','.join(str(log[h]) for h in self.log_headers) + '\n')

    def _save_model(self):
        out_model_dir = osp.join(self.out, 'models')
        if not osp.exists(out_model_dir):
            os.makedirs(out_model_dir)
        model_name = self.model.__class__.__name__
        out_model = osp.join(out_model_dir, '%s_iter%08d.npz' %
                             (model_name, self.iteration))
        chainer.serializers.save_npz(out_model, self.model)


class FCNTrainer(Trainer):

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000):
        super(FCNTrainer, self).__init__(
            device, model, optimizer, iter_train, iter_valid,
            out, max_iter, interval_validate)
        # for logging
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
        ]
        if not osp.exists(self.out):
            os.makedirs(self.out)
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def validate(self, n_viz=9):
        """Validate current model using validation dataset.

        Parameters
        ----------
        n_viz: int
            Number fo visualization.

        Returns
        -------
        log: dict
            Log values.
        """
        iter_valid = copy.copy(self.iter_valid)
        losses, lbl_trues, lbl_preds = [], [], []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            img, lbl_true = list(zip(*batch))
            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
                loss = self.model(*in_vars)
            losses.append(float(loss.data))
            score = self.model.score
            lbl_pred = chainer.functions.argmax(score, axis=1)
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
            for im, lt, lp in zip(img, lbl_true, lbl_pred):
                lbl_trues.append(lt)
                lbl_preds.append(lp)
                if len(vizs) < n_viz:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt,
                        img=im, n_class=self.model.n_class)
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        acc = fcn.utils.label_accuracy_score(
            lbl_trues, lbl_preds, self.model.n_class)
        self._write_log(**{
            'epoch': self.epoch,
            'iteration': self.iteration,
            'elapsed_time': time.time() - self.stamp_start,
            'valid/loss': np.mean(losses),
            'valid/acc': acc[0],
            'valid/acc_cls': acc[1],
            'valid/mean_iu': acc[2],
            'valid/fwavacc': acc[3],
        })
        self._save_model()

    def train(self):
        self.stamp_start = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # validate #
            ############

            if self.interval_validate and \
                    self.iteration % self.interval_validate == 0:
                self.validate()

            #########
            # train #
            #########

            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()

                _, lbl_true = list(zip(*batch))
                lbl_pred = chainer.functions.argmax(self.model.score, axis=1)
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                acc = fcn.utils.label_accuracy_score(
                    lbl_true, lbl_pred, self.model.n_class)
                self._write_log(**{
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'elapsed_time': time.time() - self.stamp_start,
                    'train/loss': float(loss.data),
                    'train/acc': acc[0],
                    'train/acc_cls': acc[1],
                    'train/mean_iu': acc[2],
                    'train/fwavacc': acc[3],
                })

            if iteration >= self.max_iter:
                self._save_model()
                break


class GraspTrainer(Trainer):

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000):

        super(GraspTrainer, self).__init__(
            device, model, optimizer, iter_train, iter_valid,
            out, max_iter, interval_validate)
        # for logging
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
            'train/loss',
            'train/seg/loss',
            'train/seg/acc',
            'train/seg/acc_cls',
            'train/seg/mean_iu',
            'train/seg/fwavacc',
            'train/grasp/loss',
            'train/grasp/acc',
            'train/grasp/precision',
            'train/grasp/recall',
            'valid/loss',
            'valid/seg/loss',
            'valid/seg/acc',
            'valid/seg/acc_cls',
            'valid/seg/mean_iu',
            'valid/seg/fwavacc',
            'valid/grasp/loss',
            'valid/grasp/acc',
            'valid/grasp/precision',
            'valid/grasp/recall',
        ]
        if not osp.exists(self.out):
            os.makedirs(self.out)
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def validate(self, n_viz=9):
        iter_valid = copy.copy(self.iter_valid)
        losses, lbl_trues, lbl_preds = [], [], []
        grasp_trues, grasp_preds = [], []
        seg_losses, grasp_losses = [], []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            img, lbl_true, grasp_true = list(zip(*batch))
            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
                loss = self.model(*in_vars)

            with chainer.cuda.get_device_from_array(loss.data):
                losses.append(float(loss.data))
                score = self.model.score
                lbl_pred = chainer.functions.argmax(score, axis=1)
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                grasp_score = self.model.grasp_score
                grasp_pred = chainer.functions.argmax(grasp_score, axis=1)
                grasp_pred = chainer.cuda.to_cpu(grasp_pred.data)

                seg_loss = self.model.seg_loss
                seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                seg_losses.append(seg_loss)
                grasp_loss = self.model.grasp_loss
                grasp_loss = chainer.cuda.to_cpu(grasp_loss.data)
                grasp_losses.append(grasp_loss)

            for im, lt, lp, grt, grp, in zip(img, lbl_true, lbl_pred,
                                             grasp_true, grasp_pred):
                lbl_trues.append(lt)
                lbl_preds.append(lp)
                grasp_trues.append(grt)
                grasp_preds.append(grp)
                if len(vizs) < n_viz:
                    viz = utils.visualize(
                        lbl_pred=lp, lbl_true=lt,
                        single_grasp_pred=grp, single_grasp_true=grt,
                        img=im, n_class=self.model.n_class)
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        seg_acc = fcn.utils.label_accuracy_score(
            lbl_trues, lbl_preds, self.model.n_class)
        grasp_acc = utils.grasp_accuracy(
            grasp_trues, grasp_preds)
        self._write_log(**{
            'epoch': self.epoch,
            'iteration': self.iteration,
            'elapsed_time': time.time() - self.stamp_start,
            'valid/loss': np.mean(losses),
            'valid/seg/loss': np.mean(seg_losses),
            'valid/seg/acc': seg_acc[0],
            'valid/seg/acc_cls': seg_acc[1],
            'valid/seg/mean_iu': seg_acc[2],
            'valid/seg/fwavacc': seg_acc[3],
            'valid/grasp/loss': np.mean(grasp_losses),
            'valid/grasp/acc': grasp_acc[0],
            'valid/grasp/precision': grasp_acc[1],
            'valid/grasp/recall': grasp_acc[2],
        })
        self._save_model()

    def train(self):
        self.stamp_start = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # validate #
            ############

            if self.interval_validate and \
                    self.iteration % self.interval_validate == 0:
                self.validate()

            #########
            # train #
            #########

            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()

                _, lbl_true, grasp_true = list(zip(*batch))
                with chainer.cuda.get_device_from_array(loss.data):
                    lbl_pred = chainer.functions.argmax(
                        self.model.score, axis=1)
                    lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                    grasp_pred = chainer.functions.argmax(
                        self.model.grasp_score, axis=1)
                    grasp_pred = chainer.cuda.to_cpu(grasp_pred.data)
                    seg_acc = fcn.utils.label_accuracy_score(
                        lbl_true, lbl_pred, self.model.n_class)
                    grasp_acc = utils.grasp_accuracy(
                        grasp_true, grasp_pred)
                    grasp_loss = self.model.grasp_loss
                    grasp_loss = chainer.cuda.to_cpu(grasp_loss.data)
                    seg_loss = self.model.seg_loss
                    seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                self._write_log(**{
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'elapsed_time': time.time() - self.stamp_start,
                    'train/loss': float(loss.data),
                    'train/seg/loss': float(seg_loss),
                    'train/seg/acc': seg_acc[0],
                    'train/seg/acc_cls': seg_acc[1],
                    'train/seg/mean_iu': seg_acc[2],
                    'train/seg/fwavacc': seg_acc[3],
                    'train/grasp/loss': float(grasp_loss),
                    'train/grasp/acc': grasp_acc[0],
                    'train/grasp/precision': grasp_acc[1],
                    'train/grasp/recall': grasp_acc[2],
                })

            if iteration >= self.max_iter:
                self._save_model()
                break


class DualarmGraspTrainer(Trainer):

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000):

        super(DualarmGraspTrainer, self).__init__(
            device, model, optimizer, iter_train, iter_valid,
            out, max_iter, interval_validate)
        # for logging
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
            'train/loss',
            'train/seg/loss',
            'train/seg/acc',
            'train/seg/acc_cls',
            'train/seg/mean_iu',
            'train/seg/fwavacc',
            'train/grasp/single/loss',
            'train/grasp/single/acc',
            'train/grasp/single/precision',
            'train/grasp/single/recall',
            'train/grasp/dual/loss',
            'train/grasp/dual/acc',
            'train/grasp/dual/precision',
            'train/grasp/dual/recall',
            'valid/loss',
            'valid/seg/loss',
            'valid/seg/acc',
            'valid/seg/acc_cls',
            'valid/seg/mean_iu',
            'valid/seg/fwavacc',
            'valid/grasp/single/loss',
            'valid/grasp/single/acc',
            'valid/grasp/single/precision',
            'valid/grasp/single/recall',
            'valid/grasp/dual/loss',
            'valid/grasp/dual/acc',
            'valid/grasp/dual/precision',
            'valid/grasp/dual/recall',
        ]
        if not osp.exists(self.out):
            os.makedirs(self.out)
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def validate(self, n_viz=9):
        iter_valid = copy.copy(self.iter_valid)
        losses, lbl_trues, lbl_preds = [], [], []
        single_grasp_trues, single_grasp_preds = [], []
        dual_grasp_trues, dual_grasp_preds = [], []
        seg_losses, single_grasp_losses, dual_grasp_losses = [], [], []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            in_data = list(zip(*batch))
            img, lbl_true, single_grasp_true, dual_grasp_true = in_data
            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
                loss = self.model(*in_vars)

            with chainer.cuda.get_device_from_array(loss.data):
                losses.append(float(loss.data))
                score = self.model.score
                lbl_pred = chainer.functions.argmax(score, axis=1)
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                single_grasp_score = self.model.single_grasp_score
                single_grasp_pred = chainer.functions.argmax(
                    single_grasp_score, axis=1)
                single_grasp_pred = chainer.cuda.to_cpu(single_grasp_pred.data)
                dual_grasp_score = self.model.dual_grasp_score
                dual_grasp_pred = chainer.functions.argmax(
                    dual_grasp_score, axis=1)
                dual_grasp_pred = chainer.cuda.to_cpu(dual_grasp_pred.data)

                seg_loss = self.model.seg_loss
                seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                seg_losses.append(seg_loss)
                single_grasp_loss = self.model.single_grasp_loss
                single_grasp_loss = chainer.cuda.to_cpu(single_grasp_loss.data)
                single_grasp_losses.append(single_grasp_loss)
                dual_grasp_loss = self.model.dual_grasp_loss
                dual_grasp_loss = chainer.cuda.to_cpu(dual_grasp_loss.data)
                dual_grasp_losses.append(dual_grasp_loss)

            for im, lt, lp, sgrt, sgrp, dgrt, dgrp in zip(
                    img, lbl_true, lbl_pred,
                    single_grasp_true, single_grasp_pred,
                    dual_grasp_true, dual_grasp_pred):
                lbl_trues.append(lt)
                lbl_preds.append(lp)
                single_grasp_trues.append(sgrt)
                single_grasp_preds.append(sgrp)
                dual_grasp_trues.append(dgrt)
                dual_grasp_preds.append(dgrp)
                if len(vizs) < n_viz:
                    viz = utils.visualize(
                        lbl_pred=lp, lbl_true=lt,
                        single_grasp_pred=sgrp, single_grasp_true=sgrt,
                        dual_grasp_pred=dgrp, dual_grasp_true=dgrt,
                        img=im, n_class=self.model.n_class)
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        seg_acc = fcn.utils.label_accuracy_score(
            lbl_trues, lbl_preds, self.model.n_class)
        single_grasp_acc = utils.grasp_accuracy(
            single_grasp_trues, single_grasp_preds)
        dual_grasp_acc = utils.grasp_accuracy(
            dual_grasp_trues, dual_grasp_preds)
        self._write_log(**{
            'epoch': self.epoch,
            'iteration': self.iteration,
            'elapsed_time': time.time() - self.stamp_start,
            'valid/loss': np.mean(losses),
            'valid/seg/loss': np.mean(seg_losses),
            'valid/seg/acc': seg_acc[0],
            'valid/seg/acc_cls': seg_acc[1],
            'valid/seg/mean_iu': seg_acc[2],
            'valid/seg/fwavacc': seg_acc[3],
            'valid/grasp/single/loss': np.mean(single_grasp_losses),
            'valid/grasp/single/acc': single_grasp_acc[0],
            'valid/grasp/single/precision': single_grasp_acc[1],
            'valid/grasp/single/recall': single_grasp_acc[2],
            'valid/grasp/dual/loss': np.mean(dual_grasp_losses),
            'valid/grasp/dual/acc': dual_grasp_acc[0],
            'valid/grasp/dual/precision': dual_grasp_acc[1],
            'valid/grasp/dual/recall': dual_grasp_acc[2],
        })
        self._save_model()

    def train(self):
        self.stamp_start = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # validate #
            ############

            if self.interval_validate and \
                    self.iteration % self.interval_validate == 0:
                self.validate()

            #########
            # train #
            #########

            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()

                in_data = list(zip(*batch))
                lbl_true, single_grasp_true, dual_grasp_true = in_data[1:]
                with chainer.cuda.get_device_from_array(loss.data):
                    lbl_pred = chainer.functions.argmax(
                        self.model.score, axis=1)
                    lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                    single_grasp_pred = chainer.functions.argmax(
                        self.model.single_grasp_score, axis=1)
                    single_grasp_pred = chainer.cuda.to_cpu(
                        single_grasp_pred.data)
                    dual_grasp_pred = chainer.functions.argmax(
                        self.model.dual_grasp_score, axis=1)
                    dual_grasp_pred = chainer.cuda.to_cpu(
                        dual_grasp_pred.data)

                    seg_acc = fcn.utils.label_accuracy_score(
                        lbl_true, lbl_pred, self.model.n_class)
                    single_grasp_acc = utils.grasp_accuracy(
                        single_grasp_true, single_grasp_pred)
                    dual_grasp_acc = utils.grasp_accuracy(
                        dual_grasp_true, dual_grasp_pred)
                    seg_loss = self.model.seg_loss
                    seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                    single_grasp_loss = self.model.single_grasp_loss
                    single_grasp_loss = chainer.cuda.to_cpu(
                        single_grasp_loss.data)
                    dual_grasp_loss = self.model.dual_grasp_loss
                    dual_grasp_loss = chainer.cuda.to_cpu(dual_grasp_loss.data)
                self._write_log(**{
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'elapsed_time': time.time() - self.stamp_start,
                    'train/loss': float(loss.data),
                    'train/seg/loss': float(seg_loss),
                    'train/seg/acc': seg_acc[0],
                    'train/seg/acc_cls': seg_acc[1],
                    'train/seg/mean_iu': seg_acc[2],
                    'train/seg/fwavacc': seg_acc[3],
                    'train/grasp/single/loss': float(single_grasp_loss),
                    'train/grasp/single/acc': single_grasp_acc[0],
                    'train/grasp/single/precision': single_grasp_acc[1],
                    'train/grasp/single/recall': single_grasp_acc[2],
                    'train/grasp/dual/loss': float(dual_grasp_loss),
                    'train/grasp/dual/acc': dual_grasp_acc[0],
                    'train/grasp/dual/precision': dual_grasp_acc[1],
                    'train/grasp/dual/recall': dual_grasp_acc[2],
                })

            if iteration >= self.max_iter:
                self._save_model()
                break


class OccludedDualarmGraspTrainer(Trainer):

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000):

        super(OccludedDualarmGraspTrainer, self).__init__(
            device, model, optimizer, iter_train, iter_valid,
            out, max_iter, interval_validate)
        # for logging
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
            'train/loss',
            'train/seg/loss',
            'train/seg/acc',
            'train/seg/acc_cls',
            'train/seg/mean_iu',
            'train/seg/fwavacc',
            'train/grasp/single/loss',
            'train/grasp/single/acc',
            'train/grasp/single/precision',
            'train/grasp/single/recall',
            'train/grasp/dual/loss',
            'train/grasp/dual/acc',
            'train/grasp/dual/precision',
            'train/grasp/dual/recall',
            'train/graph/loss',
            'train/graph/acc',
            'train/graph/precision',
            'train/graph/recall',
            'valid/loss',
            'valid/seg/loss',
            'valid/seg/acc',
            'valid/seg/acc_cls',
            'valid/seg/mean_iu',
            'valid/seg/fwavacc',
            'valid/grasp/single/loss',
            'valid/grasp/single/acc',
            'valid/grasp/single/precision',
            'valid/grasp/single/recall',
            'valid/grasp/dual/loss',
            'valid/grasp/dual/acc',
            'valid/grasp/dual/precision',
            'valid/grasp/dual/recall',
            'valid/graph/loss',
            'valid/graph/acc',
            'valid/graph/precision',
            'valid/graph/recall',
        ]
        if not osp.exists(self.out):
            os.makedirs(self.out)
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def validate(self, n_viz=9):
        iter_valid = copy.copy(self.iter_valid)
        losses, lbl_trues, lbl_preds = [], [], []
        single_grasp_trues, single_grasp_preds = [], []
        dual_grasp_trues, dual_grasp_preds = [], []
        graph_trues, graph_preds = [], []
        seg_losses, single_grasp_losses, dual_grasp_losses = [], [], []
        graph_losses = []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            in_data = list(zip(*list(batch)))
            img, lbl_true, single_grasp_true = in_data[:3]
            dual_grasp_true, graph_true = in_data[3:]
            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
                loss = self.model(*in_vars)

            with chainer.cuda.get_device_from_array(loss.data):
                losses.append(float(loss.data))
                score = self.model.score
                lbl_pred = chainer.functions.argmax(score, axis=1)
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                single_grasp_score = self.model.single_grasp_score
                single_grasp_pred = chainer.functions.argmax(
                    single_grasp_score, axis=1)
                single_grasp_pred = chainer.cuda.to_cpu(single_grasp_pred.data)
                dual_grasp_score = self.model.dual_grasp_score
                dual_grasp_pred = chainer.functions.argmax(
                    dual_grasp_score, axis=1)
                dual_grasp_pred = chainer.cuda.to_cpu(dual_grasp_pred.data)
                graph_score = self.model.graph_score
                graph_pred = chainer.functions.argmax(
                    graph_score, axis=1)
                graph_pred = chainer.cuda.to_cpu(graph_pred.data)

                seg_loss = self.model.seg_loss
                seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                seg_losses.append(seg_loss)
                single_grasp_loss = self.model.single_grasp_loss
                single_grasp_loss = chainer.cuda.to_cpu(single_grasp_loss.data)
                single_grasp_losses.append(single_grasp_loss)
                dual_grasp_loss = self.model.dual_grasp_loss
                dual_grasp_loss = chainer.cuda.to_cpu(dual_grasp_loss.data)
                dual_grasp_losses.append(dual_grasp_loss)
                graph_loss = self.model.graph_loss
                graph_loss = chainer.cuda.to_cpu(graph_loss.data)
                graph_losses.append(graph_loss)

            for im, lt, lp, sgrt, sgrp, dgrt, dgrp, grht, grhp in zip(
                    img, lbl_true, lbl_pred,
                    single_grasp_true, single_grasp_pred,
                    dual_grasp_true, dual_grasp_pred,
                    graph_true, graph_pred):
                lbl_trues.append(lt)
                lbl_preds.append(lp)
                single_grasp_trues.append(sgrt)
                single_grasp_preds.append(sgrp)
                dual_grasp_trues.append(dgrt)
                dual_grasp_preds.append(dgrp)
                graph_preds.append(grhp)
                graph_trues.append(grht)
                if len(vizs) < n_viz:
                    viz = utils.visualize(
                        lbl_pred=lp, lbl_true=lt,
                        single_grasp_pred=sgrp, single_grasp_true=sgrt,
                        dual_grasp_pred=dgrp, dual_grasp_true=dgrt,
                        img=im, n_class=self.model.n_class)
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        seg_acc = fcn.utils.label_accuracy_score(
            lbl_trues, lbl_preds, self.model.n_class)
        single_grasp_acc = utils.grasp_accuracy(
            single_grasp_trues, single_grasp_preds)
        dual_grasp_acc = utils.grasp_accuracy(
            dual_grasp_trues, dual_grasp_preds)
        graph_acc = utils.grasp_accuracy(
            graph_trues, graph_preds)

        self._write_log(**{
            'epoch': self.epoch,
            'iteration': self.iteration,
            'elapsed_time': time.time() - self.stamp_start,
            'valid/loss': np.mean(losses),
            'valid/seg/loss': np.mean(seg_losses),
            'valid/seg/acc': seg_acc[0],
            'valid/seg/acc_cls': seg_acc[1],
            'valid/seg/mean_iu': seg_acc[2],
            'valid/seg/fwavacc': seg_acc[3],
            'valid/grasp/single/loss': np.mean(single_grasp_losses),
            'valid/grasp/single/acc': single_grasp_acc[0],
            'valid/grasp/single/precision': single_grasp_acc[1],
            'valid/grasp/single/recall': single_grasp_acc[2],
            'valid/grasp/dual/loss': np.mean(dual_grasp_losses),
            'valid/grasp/dual/acc': dual_grasp_acc[0],
            'valid/grasp/dual/precision': dual_grasp_acc[1],
            'valid/grasp/dual/recall': dual_grasp_acc[2],
            'valid/graph/loss': np.mean(graph_losses),
            'valid/graph/acc': graph_acc[0],
            'valid/graph/precision': graph_acc[1],
            'valid/graph/recall': graph_acc[2],
        })
        self._save_model()

    def train(self):
        self.stamp_start = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # validate #
            ############

            if self.interval_validate and \
                    self.iteration % self.interval_validate == 0:
                self.validate()

            #########
            # train #
            #########

            batch = list(map(fcn.datasets.transform_lsvrc2012_vgg16, batch))
            in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()

                in_data = list(zip(*batch))
                lbl_true, single_grasp_true, dual_grasp_true, graph_true \
                    = in_data[1:]
                with chainer.cuda.get_device_from_array(loss.data):
                    lbl_pred = chainer.functions.argmax(
                        self.model.score, axis=1)
                    lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                    single_grasp_pred = chainer.functions.argmax(
                        self.model.single_grasp_score, axis=1)
                    single_grasp_pred = chainer.cuda.to_cpu(
                        single_grasp_pred.data)
                    dual_grasp_pred = chainer.functions.argmax(
                        self.model.dual_grasp_score, axis=1)
                    dual_grasp_pred = chainer.cuda.to_cpu(
                        dual_grasp_pred.data)
                    graph_pred = chainer.functions.argmax(
                        self.model.graph_score, axis=1)
                    graph_pred = chainer.cuda.to_cpu(graph_pred.data)

                    seg_acc = fcn.utils.label_accuracy_score(
                        lbl_true, lbl_pred, self.model.n_class)
                    single_grasp_acc = utils.grasp_accuracy(
                        single_grasp_true, single_grasp_pred)
                    dual_grasp_acc = utils.grasp_accuracy(
                        dual_grasp_true, dual_grasp_pred)
                    graph_acc = utils.grasp_accuracy(
                        graph_true, graph_pred)

                    seg_loss = self.model.seg_loss
                    seg_loss = chainer.cuda.to_cpu(seg_loss.data)
                    single_grasp_loss = self.model.single_grasp_loss
                    single_grasp_loss = chainer.cuda.to_cpu(
                        single_grasp_loss.data)
                    dual_grasp_loss = self.model.dual_grasp_loss
                    dual_grasp_loss = chainer.cuda.to_cpu(dual_grasp_loss.data)
                    graph_loss = self.model.graph_loss
                    graph_loss = chainer.cuda.to_cpu(graph_loss.data)

                self._write_log(**{
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'elapsed_time': time.time() - self.stamp_start,
                    'train/loss': float(loss.data),
                    'train/seg/loss': float(seg_loss),
                    'train/seg/acc': seg_acc[0],
                    'train/seg/acc_cls': seg_acc[1],
                    'train/seg/mean_iu': seg_acc[2],
                    'train/seg/fwavacc': seg_acc[3],
                    'train/grasp/single/loss': float(single_grasp_loss),
                    'train/grasp/single/acc': single_grasp_acc[0],
                    'train/grasp/single/precision': single_grasp_acc[1],
                    'train/grasp/single/recall': single_grasp_acc[2],
                    'train/grasp/dual/loss': float(dual_grasp_loss),
                    'train/grasp/dual/acc': dual_grasp_acc[0],
                    'train/grasp/dual/precision': dual_grasp_acc[1],
                    'train/grasp/dual/recall': dual_grasp_acc[2],
                    'train/graph/loss': float(graph_loss),
                    'train/graph/acc': graph_acc[0],
                    'train/graph/precision': graph_acc[1],
                    'train/graph/recall': graph_acc[2],
                })

            if iteration >= self.max_iter:
                self._save_model()
                break
