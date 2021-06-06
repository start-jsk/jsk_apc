#!/usr/bin/env python

from __future__ import division

import argparse
import easydict
import os.path as osp
import yaml

import chainer
from chainer import iterators
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV1
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV2
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV3
from grasp_data_generator.datasets import OIDualarmGraspDatasetV1
from grasp_data_generator.datasets import OIDualarmGraspDatasetV2
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV1
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV2
from grasp_data_generator.evaluations import eval_instance_segmentation_voc
from grasp_data_generator.models import OccludedGraspMaskRCNNResNet101


thisdir = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--yaml', type=str)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    with open(osp.join(thisdir, args.yaml), 'r') as f:
        params = easydict.EasyDict(yaml.load(f))

    if args.dataset is not None:
        params.dataset = args.dataset

    if params.dataset == 'v1':
        test_data = OIDualarmGraspDatasetV1(
            split='val', imgaug=False, return_rotation=False)
    elif params.dataset == 'v2':
        test_data = OIDualarmGraspDatasetV2(
            split='val', imgaug=False, return_rotation=True)
    elif params.dataset == 'fv1':
        test_data = FinetuningOIDualarmGraspDatasetV1(
            split='val', imgaug=False, return_rotation=True)
    elif params.dataset == 'fv2':
        test_data = FinetuningOIDualarmGraspDatasetV2(
            split='val', imgaug=False, return_rotation=True)
    elif params.dataset == 'fv3':
        test_data = FinetuningOIDualarmGraspDatasetV3(
            split='val', imgaug=False, return_rotation=True)
    elif params.dataset == 'ev1':
        test_data = OIRealAnnotatedDatasetV1(
            split='all', imgaug=False, clip=args.clip)
    elif params.dataset == 'ev2':
        test_data = OIRealAnnotatedDatasetV2(
            split='all', imgaug=False, clip=args.clip)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(params.dataset))

    if 'rotate_angle' not in params:
        rotate_angle = None
    else:
        rotate_angle = params.rotate_angle

    model = OccludedGraspMaskRCNNResNet101(
        n_fg_class=len(test_data.label_names),
        anchor_scales=params.anchor_scales,
        min_size=params.min_size,
        max_size=params.max_size,
        rpn_dim=params.rpn_dim,
        rotate_angle=rotate_angle)
    model.nms_thresh = 0.3
    model.score_thresh = 0.05
    chainer.serializers.load_npz(
        osp.join(thisdir, args.pretrained_model), model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    iterator = iterators.SerialIterator(
        test_data, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(test_data)))
    # delete unused iterators explicitly
    del in_values

    pred_masks, pred_labels, _, pred_scores, _, _ = out_values
    gt_masks, gt_labels = rest_values[:2]

    result = eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, iou_thresh=args.iou_thresh,
        use_07_metric=False)

    print('')
    print('mAP: {:f}'.format(result['map']))
    print('mSQ: {:f}'.format(result['msq']))
    print('mSQ/Vis: {:f}'.format(result['msq/vis']))
    print('mSQ/Occ: {:f}'.format(result['msq/occ']))
    print('mDQ: {:f}'.format(result['mdq']))
    print('mPQ: {:f}'.format(result['mpq']))


if __name__ == '__main__':
    main()
