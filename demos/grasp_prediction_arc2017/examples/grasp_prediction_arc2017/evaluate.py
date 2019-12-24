#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import fcn
import numpy as np
import skimage.io
import tqdm

import mvtk
from mvtk.contrib.grasp_prediction_arc2017 import datasets
from mvtk.contrib.grasp_prediction_arc2017 import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()

    gpu = args.gpu

    dataset = datasets.ItemDataDataset(
        'valid', osp.expanduser('~/data/arc2017/item_data/pick_re-experiment'))
    n_class = len(dataset.class_names)

    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=1, repeat=False, shuffle=False,
        n_prefetch=20, shared_mem=10 ** 7)

    model = models.FCN32s(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    chainer.config.enable_backprop = False
    chainer.config.train = False

    imgs, lbl_preds, lbl_trues = [], [], []
    lbl_suc_preds, lbl_suc_trues = [], []
    for batch in tqdm.tqdm(iter_valid, total=len(dataset)):
        imgs.append(batch[0][0])
        batch = map(fcn.datasets.transform_lsvrc2012_vgg16, batch)
        x_data, lbl_true, lbl_suc_true = zip(*batch)
        x_data, lbl_true, lbl_suc_true = \
            map(np.asarray, [x_data, lbl_true, lbl_suc_true])
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)

        x = chainer.Variable(x_data)
        model(x)
        lbl_pred = chainer.functions.argmax(model.score_cls, axis=1)
        lbl_pred = lbl_pred.data
        lbl_suc_pred = chainer.functions.argmax(model.score_suc, axis=1)
        lbl_suc_pred = lbl_suc_pred.data

        lbl_preds.append(chainer.cuda.to_cpu(lbl_pred[0]))
        lbl_trues.append(lbl_true[0])
        lbl_suc_preds.append(chainer.cuda.to_cpu(lbl_suc_pred[0]))
        lbl_suc_trues.append(lbl_suc_true[0])

    acc, acc_cls, mean_iu, fwavacc = \
        fcn.utils.label_accuracy_score(lbl_trues, lbl_preds, n_class)
    print('lbl_cls:')
    print('  Accuracy: %.4f' % (100 * acc))
    print('  AccClass: %.4f' % (100 * acc_cls))
    print('  Mean IoU: %.4f' % (100 * mean_iu))
    print('  Fwav Acc: %.4f' % (100 * fwavacc))

    acc, acc_cls, mean_iu, fwavacc = \
        fcn.utils.label_accuracy_score(lbl_suc_trues, lbl_suc_preds, 2)
    print('lbl_suc:')
    print('  Accuracy: %.4f' % (100 * acc))
    print('  AccClass: %.4f' % (100 * acc_cls))
    print('  Mean IoU: %.4f' % (100 * mean_iu))
    print('  Fwav Acc: %.4f' % (100 * fwavacc))

    out_dir = osp.splitext(args.model_file)[0]
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(imgs)):
        viz_cls = mvtk.datasets.visualize_segmentation(
            imgs[i], lbl_trues[i], lbl_preds[i],
            class_names=dataset.class_names)
        viz_suc = mvtk.datasets.visualize_segmentation(
            imgs[i], lbl_suc_trues[i], lbl_suc_preds[i],
            class_names=['no_suction', 'suction'])
        viz = np.hstack([viz_cls, viz_suc])
        skimage.io.imsave(osp.join(out_dir, '%06d.jpg' % i), viz)
    print('Wrote results to: %s' % out_dir)


if __name__ == '__main__':
    main()
