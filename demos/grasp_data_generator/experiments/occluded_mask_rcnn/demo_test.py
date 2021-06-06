#!/usr/bin/env python

from __future__ import division

import argparse
import easydict
import os.path as osp
import yaml

import chainer
import matplotlib.pyplot as plt

from grasp_data_generator.datasets import OIDualarmGraspDatasetV1
from grasp_data_generator.models import OccludedMaskRCNNResNet101
from grasp_data_generator.visualizations \
    import vis_occluded_instance_segmentation


thisdir = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--logs', type=str)
    parser.add_argument('--pretrained-model', type=str)
    args = parser.parse_args()

    with open(osp.join(thisdir, args.logs, 'params.yaml'), 'r') as f:
        params = easydict.EasyDict(yaml.load(f))

    if params.dataset == 'v1':
        test_data = OIDualarmGraspDatasetV1(split='val', imgaug=False)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(params.dataset))
    label_names = test_data.label_names

    model = OccludedMaskRCNNResNet101(
        n_fg_class=len(label_names),
        anchor_scales=params.anchor_scales,
        min_size=params.min_size,
        max_size=params.max_size,
        rpn_dim=params.rpn_dim)
    model.nms_thresh = 0.3
    model.score_thresh = 0.5
    chainer.serializers.load_npz(
        osp.join(thisdir, args.pretrained_model), model)

    if args.gpu >= 0:
        model.to_gpu()

    for i in range(len(test_data)):
        img = test_data[i][0]
        ins_labels, labels, bboxes, scores = model.predict([img])
        ins_label, label, bbox, score \
            = ins_labels[0], labels[0], bboxes[0], scores[0]
        vis_occluded_instance_segmentation(
            img, ins_label, label, bbox, score, label_names)
        plt.show()


if __name__ == '__main__':
    main()
