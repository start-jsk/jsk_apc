#!/usr/bin/env python

import argparse
import chainer
import chainer.serializers as S
import datetime
import fcn
import numpy as np
import os
import os.path as osp
import scipy.misc
import yaml

import grasp_data_generator


filepath = osp.dirname(osp.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-dir')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    gpu = args.gpu
    log_dir = osp.join(filepath, args.log_dir)
    model_dir = osp.join(log_dir, 'models')
    cfgpath = osp.join(log_dir, 'config.yaml')

    with open(cfgpath, 'r') as f:
        config = yaml.load(f)

    validdir = osp.join(filepath, '../../data/valid')
    imagedirs = os.listdir(validdir)
    imagedirs = [d for d in imagedirs if osp.isfile(osp.join(validdir, d))]

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = osp.join(filepath, 'out_viz', timestamp)
    if not osp.exists(outdir):
        os.makedirs(outdir)

    dataset_class = getattr(
        grasp_data_generator.datasets, config['dataset_class'])
    label_names = dataset_class.label_names()
    n_class = len(label_names)
    npznames = sorted(os.listdir(model_dir))

    for npzname in npznames:
        model_path = osp.join(model_dir, npzname)
        if not osp.isfile(model_path):
            continue
        if not npzname.endswith('.npz'):
            continue
        outvizname = '{}_valid.png'.format(npzname.split('.')[0])
        outvizpath = osp.join(outdir, outvizname)

        model = grasp_data_generator.models.DualarmGraspFCN32s(n_class=n_class)
        S.load_npz(model_path, model)

        if 'alpha' in config:
            alpha = config['alpha']
            if isinstance(alpha, dict):
                model.alpha_single = alpha['single']
                model.alpha_dual = alpha['dual']
            else:
                model.alpha_single = alpha
                model.alpha_dual = 1.0
        else:
            model.alpha_single = 1.0
            model.alpha_dual = 1.0

        if 'frequency_balancing' in config:
            frq_balancing = config['frequency_balancing']
            model.frq_balancing = frq_balancing
        else:
            model.frq_balancing = False

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

        vizs = []
        for dirname in imagedirs:
            img = scipy.misc.imread(osp.join(validdir, dirname))
            img = scipy.misc.imresize(img, (480, 640))

            # inference
            img_data = img.copy()
            x_data = fcn.datasets.transform_lsvrc2012_vgg16([img_data])
            x_data = np.array(x_data, dtype=np.float32)
            x_data = chainer.cuda.to_gpu(x_data, device=gpu)
            x = chainer.Variable(x_data)
            model(x)

            score = model.score
            prob = chainer.functions.softmax(score)
            prob = prob.data
            mask = prob < 0.5
            prob[mask] = 0.0
            lbl_pred = chainer.functions.argmax(prob, axis=1)
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
            single_grasp_score = model.single_grasp_score
            single_grasp_pred = chainer.functions.argmax(
                single_grasp_score, axis=1)
            single_grasp_pred = chainer.cuda.to_cpu(single_grasp_pred.data)
            dual_grasp_score = model.dual_grasp_score
            dual_grasp_pred = chainer.functions.argmax(
                dual_grasp_score, axis=1)
            dual_grasp_pred = chainer.cuda.to_cpu(dual_grasp_pred.data)

            viz = grasp_data_generator.utils.visualize(
                label_names=label_names,
                lbl_pred=lbl_pred[0],
                single_grasp_pred=single_grasp_pred[0],
                dual_grasp_pred=dual_grasp_pred[0],
                img=img, n_class=model.n_class,
                alpha=0.3)
            vizs.append(viz)
        outimg = fcn.utils.get_tile_image(vizs)
        scipy.misc.imsave(outvizpath, outimg)


if __name__ == '__main__':
    main()
