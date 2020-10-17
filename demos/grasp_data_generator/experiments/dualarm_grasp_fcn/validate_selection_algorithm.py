#!/usr/bin/env python

import argparse
import chainer
import chainer.serializers as S
import cupy
import fcn
import numpy as np
import os
import os.path as osp
from sklearn.cluster import KMeans
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

    npznames = sorted(os.listdir(model_dir))
    for npzname in npznames:
        model_path = osp.join(model_dir, npzname)
        if not osp.isfile(model_path):
            continue
        if not npzname.endswith('.npz'):
            continue

        # datasets
        random_state = int(config['random_state'])
        if 'dataset_class' not in config:
            config['dataset_class'] = 'DualarmGraspDataset'
        dataset_class = getattr(grasp_data_generator.datasets,
                                config['dataset_class'])
        dataset_valid = dataset_class('valid', imgaug=False,
                                      random_state=random_state)

        n_class = len(dataset_valid.label_names)
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

        chainer.config.train = False
        chainer.config.train = False
        chainer.config.enable_backprop = False
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

        accs = []

        for i in range(0, len(dataset_valid)):
            data = dataset_valid.get_example(i)
            img_data, label_data, sg_data, dg_data = data

            gt = ground_truth(label_data, sg_data, dg_data)

            # inference
            x_data = fcn.datasets.transform_lsvrc2012_vgg16([img_data])
            x_data = np.array(x_data, dtype=np.float32)
            x_data = chainer.cuda.to_gpu(x_data, device=gpu)
            x = chainer.Variable(x_data)
            model(x)

            score = chainer.functions.softmax(model.score)
            sg_score = chainer.functions.softmax(model.single_grasp_score)
            dg_score = chainer.functions.softmax(model.dual_grasp_score)
            score = score.data[0]
            sg_score = sg_score.data[0]
            dg_score = dg_score.data[0]

            # [H, W]
            sg_grasp = sg_score[1, :, :]
            dg_grasp = dg_score[1, :, :]

            # [C, H, W]
            sg_grasp = score * sg_grasp[cupy.newaxis, :, :]
            dg_grasp = score * dg_grasp[cupy.newaxis, :, :]

            # {grasp_way: [label]}
            pred = validate(sg_grasp, dg_grasp)
            acc = cal_acc(gt, pred)
            accs.append(acc)

        accs = np.array(accs)
        accs = accs.astype(np.int32)
        total_acc = accs.sum() / float(len(accs))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('model file: {}'.format(npzname))
        print('acc: {}'.format(total_acc))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


def ground_truth(label, sg, dg):
    sg_grasp = label * sg
    sg_count = np.bincount(sg_grasp.flatten(), minlength=41)
    dg_grasp = label * dg
    dg_count = np.bincount(dg_grasp.flatten(), minlength=41)
    gt = {
        'single': [],
        'dual': []
    }
    for i, c in enumerate(sg_count):
        if i == 0:
            continue
        if c > 0:
            gt['single'].append(i)
    for i, c in enumerate(dg_count):
        if i == 0:
            continue
        if c > 0:
            gt['dual'].append(i)
    return gt


def validate(sg_grasp, dg_grasp):
    pred = {
        'single': [],
        'dual': []
    }

    # [C, H, W]
    sg_grasp = chainer.cuda.to_cpu(sg_grasp)
    dg_grasp = chainer.cuda.to_cpu(dg_grasp)
    sg_grasp = sg_grasp > 0.5
    dg_grasp = dg_grasp > 0.5

    for i, img in enumerate(sg_grasp):
        # __background__
        if i == 0:
            continue
        indices = np.column_stack(np.where(img))
        if len(indices) < 80:
            continue
        pred['single'].append(i)
    for i, img in enumerate(dg_grasp):
        # __background__
        if i == 0:
            continue
        if not img.any():
            continue
        indices = np.column_stack(np.where(img))
        if len(indices) < 80:
            continue
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(indices)
        centers = kmeans.cluster_centers_
        distance = np.linalg.norm(centers[0] - centers[1])
        if distance > 50:
            pred['dual'].append(i)
    return pred


def cal_acc(gt, pred):
    for style in ['single', 'dual']:
        if len(pred[style]) == 0:
            if len(gt[style]) != 0:
                return False
            continue
        for p in pred[style]:
            if p not in gt[style]:
                return False
    return True


if __name__ == '__main__':
    main()
