import collections
import copy
import os
import os.path as osp
import time

import chainer
import fcn
import numpy as np
import pandas
import skimage.io
import skimage.util
import tqdm

import grasp_fusion_lib


class FCNTrainer(object):

    """Training class for FCN models.

    Parameters
    ----------
    device: int
        GPU id, negative values represents use of CPU.
    model: chainer.Chain
        NN model.
    optimizer: chainer.Optimizer
        Optimizer.
    iter_train: chainer.Iterator
        Dataset itarator for training dataset.
    iter_valid: chainer.Iterator
        Dataset itarator for validation dataset.
    out: str
        Log output directory.
    max_iter: int
        Max iteration to stop training iterations.
    interval_validate: None or int
        If None, validation is never run. (default: 4000)
    """

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000,
            interval_save=None,
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
        self.interval_save = interval_save
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
            'train/loss',
            'train/lbl_cls/loss',
            'train/lbl_cls/acc_cls',
            'train/lbl_cls/mean_iu',
            'train/lbl_suc/loss',
            'train/lbl_suc/acc_cls',
            'train/lbl_suc/mean_iu',
            'valid/loss',
            'valid/lbl_cls/loss',
            'valid/lbl_cls/acc_cls',
            'valid/lbl_cls/mean_iu',
            'valid/lbl_suc/loss',
            'valid/lbl_suc/acc_cls',
            'valid/lbl_suc/mean_iu',
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
        losses = []
        lbl_cls_trues, lbl_cls_preds = [], []
        lbl_suc_trues, lbl_suc_preds = [], []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            img, lbl_cls_true, lbl_suc_true = zip(*batch)
            batch = map(fcn.datasets.transform_lsvrc2012_vgg16, batch)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
                loss_cls, loss_suc = self.model(*in_vars)
            loss = loss_cls + 100 * loss_suc
            losses.append({'__sum__': float(loss.data),
                           'lbl_cls': float(loss_cls.data),
                           'lbl_suc': float(loss_suc.data)})
            score_cls = self.model.score_cls
            lbl_cls_pred = chainer.functions.argmax(score_cls, axis=1)
            lbl_cls_pred = chainer.cuda.to_cpu(lbl_cls_pred.data)
            score_suc = self.model.score_suc
            lbl_suc_pred = chainer.functions.argmax(score_suc, axis=1)
            lbl_suc_pred = chainer.cuda.to_cpu(lbl_suc_pred.data)
            hmp_suc_pred = chainer.functions.softmax(score_suc)
            hmp_suc_pred = chainer.cuda.to_cpu(hmp_suc_pred.data)[:, 1, :, :]
            for im, lct, lcp, lst, lsp, hsp in \
                    zip(img, lbl_cls_true, lbl_cls_pred,
                        lbl_suc_true, lbl_suc_pred, hmp_suc_pred):
                lbl_cls_trues.append(lct)
                lbl_cls_preds.append(lcp)
                lbl_suc_trues.append(lst)
                lbl_suc_preds.append(lsp)
                if len(vizs) < n_viz:
                    viz_cls = fcn.utils.visualize_segmentation(
                        lbl_pred=lcp, lbl_true=lct,
                        img=im, n_class=self.model.n_class)
                    viz_suc = fcn.utils.visualize_segmentation(
                        lbl_pred=lsp, lbl_true=lst,
                        img=im, n_class=self.model.n_class)
                    hst = np.zeros_like(hsp)
                    hst.fill(0.5)
                    hst[lst == 1] = 1
                    viz_suc = np.hstack([
                        viz_suc,
                        np.vstack([
                            grasp_fusion_lib.image.colorize_heatmap(hst),
                            grasp_fusion_lib.image.colorize_heatmap(hsp),
                        ]),
                    ])
                    viz = np.hstack((viz_cls, viz_suc))
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        acc_lbl_cls = fcn.utils.label_accuracy_score(
            lbl_cls_trues, lbl_cls_preds, self.model.n_class)
        acc_lbl_suc = fcn.utils.label_accuracy_score(
            lbl_suc_trues, lbl_suc_preds, 2)
        loss = pandas.DataFrame(losses).mean()
        log = {
            'valid/loss': loss['__sum__'],
            'valid/lbl_cls/loss': loss['lbl_cls'],
            'valid/lbl_cls/acc': acc_lbl_cls[0],
            'valid/lbl_cls/acc_cls': acc_lbl_cls[1],
            'valid/lbl_cls/mean_iu': acc_lbl_cls[2],
            'valid/lbl_cls/fwavacc': acc_lbl_cls[3],
            'valid/lbl_suc/loss': loss['lbl_suc'],
            'valid/lbl_suc/acc': acc_lbl_suc[0],
            'valid/lbl_suc/acc_cls': acc_lbl_suc[1],
            'valid/lbl_suc/mean_iu': acc_lbl_suc[2],
            'valid/lbl_suc/fwavacc': acc_lbl_suc[3],
        }
        # finalize
        return log

    def save_model(self):
        out_model_dir = osp.join(self.out, 'models')
        if not osp.exists(out_model_dir):
            os.makedirs(out_model_dir)
        out_model = osp.join(
            out_model_dir, '%s_iter%08d.npz' %
            (self.model.__class__.__name__, self.iteration))
        chainer.serializers.save_npz(out_model, self.model)

    def train(self):
        """Train the network using the training dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        stamp_start = time.time()
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
                log = collections.defaultdict(str)
                log_valid = self.validate()
                log.update(log_valid)
                log['epoch'] = self.iter_train.epoch
                log['iteration'] = iteration
                log['elapsed_time'] = time.time() - stamp_start
                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    f.write(','.join(str(log[h]) for h in self.log_headers) +
                            '\n')
                self.save_model()

            #########
            # train #
            #########

            batch = map(fcn.datasets.transform_lsvrc2012_vgg16, batch)
            in_vars = fcn.utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss_cls, loss_suc = self.model(*in_vars)
            loss = loss_cls + 100 * loss_suc

            if loss_cls is not None and loss_suc is not None:
                loss.backward()
                self.optimizer.update()

                _, lbl_cls_true, lbl_suc_true = zip(*batch)
                lbl_cls_pred = chainer.functions.argmax(
                    self.model.score_cls, axis=1)
                lbl_cls_pred = chainer.cuda.to_cpu(lbl_cls_pred.data)
                lbl_suc_pred = chainer.functions.argmax(
                    self.model.score_suc, axis=1)
                lbl_suc_pred = chainer.cuda.to_cpu(lbl_suc_pred.data)
                acc_lbl_cls = fcn.utils.label_accuracy_score(
                    lbl_cls_true, lbl_cls_pred, self.model.n_class)
                acc_lbl_suc = fcn.utils.label_accuracy_score(
                    lbl_suc_true, lbl_suc_pred, 2)
                log = collections.defaultdict(str)
                log_train = {
                    'train/loss': float(loss.data),
                    'train/lbl_cls/loss': float(loss_cls.data),
                    'train/lbl_cls/acc': acc_lbl_cls[0],
                    'train/lbl_cls/acc_cls': acc_lbl_cls[1],
                    'train/lbl_cls/mean_iu': acc_lbl_cls[2],
                    'train/lbl_cls/fwavacc': acc_lbl_cls[3],
                    'train/lbl_suc/loss': float(loss_suc.data),
                    'train/lbl_suc/acc': acc_lbl_suc[0],
                    'train/lbl_suc/acc_cls': acc_lbl_suc[1],
                    'train/lbl_suc/mean_iu': acc_lbl_suc[2],
                    'train/lbl_suc/fwavacc': acc_lbl_suc[3],
                }
                log['epoch'] = self.iter_train.epoch
                log['iteration'] = iteration
                log['elapsed_time'] = time.time() - stamp_start
                log.update(log_train)
                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    f.write(','.join(str(log[h]) for h in self.log_headers) +
                            '\n')

            if self.interval_save and \
                    self.iteration != 0 and \
                    self.iteration % self.interval_save == 0:
                self.save_model()

            if iteration >= self.max_iter:
                self.save_model()
                break
