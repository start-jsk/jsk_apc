#!/usr/bin/env python

import argparse
import chainer
from chainer import cuda
import chainer.serializers as S
from chainer import Variable
import cv2
import numpy as np
import os
import os.path as osp
import scipy.misc
import scipy.ndimage
import yaml

from selective_dualarm_stowing.datasets import DualarmDatasetV5
from selective_dualarm_stowing.models import DualarmAlex
from selective_dualarm_stowing.utils import get_APC_pt


label_names = np.array(['drop', 'protrusion'])


def visualize_val_result(gpu, model_path, out, config):

    # config load
    cross_validation = config['cross_validation']
    threshold = config['threshold']
    random_state = config['random_state']
    resize_rate = config['resize_rate']
    test_size = config['test_size']
    with_damage = config['with_damage']

    dataset = DualarmDatasetV5(
        'val', random_state, resize_rate, test_size,
        cross_validation, 0, False, with_damage, True)
    n_failure = len(dataset.failure_label)
    n_class = len(dataset.class_label)

    model = DualarmAlex(n_failure, n_class, threshold, get_APC_pt)
    model.train_conv = False

    S.load_hdf5(model_path, model)
    chainer.config.train = False
    chainer.config.enable_backprop = False
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()

    for i in range(0, len(dataset)):
        x_data, t_data, _ = dataset.get_example(i)
        # masked
        masked_img = dataset.datum_to_img(x_data)
        masked_img = masked_img.copy()
        masked_img = scipy.ndimage.rotate(masked_img, -90)
        # original
        img, _, _ = dataset.get_example(i, False)
        img = dataset.datum_to_img(img)
        img = scipy.ndimage.rotate(img, -90)
        rgb = np.hstack((img, masked_img))

        if t_data[0] != -1:
            is_singlearm = True
            answer_label = t_data[:2]
        else:
            is_singlearm = False
            answer_label = t_data[2:]

        x_data = np.array([x_data], dtype=np.float32)
        x_data = cuda.to_gpu(x_data, device=gpu)
        x = Variable(x_data)
        model(x)
        fail_prob = cuda.to_cpu(model.fail_prob.data)
        fail_prob = fail_prob[0]

        if is_singlearm:
            predicted_proba = fail_prob[:2]
        else:
            predicted_proba = fail_prob[2:]
        predicted_label = (predicted_proba > 0.5).astype(np.int32)

        answer_result = label_names[answer_label == 1]
        predicted_result = label_names[predicted_label == 1]
        answer_text = None
        predicted_text = None
        for result in answer_result:
            if answer_text is None:
                answer_text = 'Answer : {}'.format(result)
            else:
                answer_text += ', {}'.format(result)
        if answer_text is None:
            answer_text = 'Answer : success'
        for result in predicted_result:
            if predicted_text is None:
                predicted_text = 'Prediction : {}'.format(result)
            else:
                predicted_text += ', {}'.format(result)
        if predicted_text is None:
            predicted_text = 'Prediction : success'

        if all(x == y for x, y in zip(answer_label, predicted_label)):
            rgb[-70:, :] = np.array([0, 255, 0])
        else:
            rgb[-70:, :] = np.array([255, 0, 0])
        cv2.putText(
            rgb, answer_text, org=(60, 620),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.3, color=(0, 0, 0), thickness=2)
        cv2.putText(
            rgb, predicted_text, org=(520, 620),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.3, color=(0, 0, 0), thickness=2)
        filename = 'result_{0:02d}.png'.format(i)
        if is_singlearm:
            filepath = osp.join(this_dir, out, 'singlearm', filename)
        else:
            filepath = osp.join(this_dir, out, 'dualarm', filename)
        # scipy.misc.imshow(rgb)
        scipy.misc.imsave(filepath, rgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-m', '--model-path')
    parser.add_argument('-o', '--out', default='output_viz')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out

    # config
    if args.config is None:
        cfgpath = osp.join(this_dir, 'cfg/dualarm_alexnet/config.yaml')
    else:
        cfgpath = args.config
    with open(cfgpath, 'r') as f:
        config = yaml.load(f)

    if not osp.exists(out):
        os.mkdir(out)
        os.mkdir(osp.join(out, 'singlearm'))
        os.mkdir(osp.join(out, 'dualarm'))

    visualize_val_result(gpu, args.model_path, out, config)
