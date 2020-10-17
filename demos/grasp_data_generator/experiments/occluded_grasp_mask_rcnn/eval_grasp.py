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
from grasp_data_generator.evaluations import eval_grasp_segmentation
from grasp_data_generator.models import OccludedGraspMaskRCNNResNet101


thisdir = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--yaml', type=str)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    args = parser.parse_args()

    with open(osp.join(thisdir, args.yaml), 'r') as f:
        params = easydict.EasyDict(yaml.load(f))

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

    _, _, _, _, pred_sg_masks, pred_dg_masks = out_values
    _, _, gt_sg_masks, gt_dg_masks = rest_values

    results = eval_grasp_segmentation(
        pred_sg_masks, pred_dg_masks, gt_sg_masks, gt_dg_masks)

    for grasp_name in ['sg', 'dg']:
        print('')
        print('grasp name: {}'.format(grasp_name))
        result = results[grasp_name]
        for class_name, iu in zip(['background', 'graspable'], result['iou']):
            print('{:>23} : {:.4f}'.format(class_name, iu))
        print('{:>23} : {:.4f}'.format('mean IoU', result['miou']))
        print('{:>23} : {:.4f}'.format(
            'Class average accuracy', result['mean_class_accuracy']))
        print('{:>23} : {:.4f}'.format(
            'Global average accuracy', result['pixel_accuracy']))

    # result = eval_instance_segmentation_voc(
    #     pred_sg_masks, pred_labels, pred_scores,
    #     gt_sg_masks, gt_labels, iou_thresh=args.iou_thresh,
    #     use_07_metric=False)

    # print('')
    # print('mAP: {:f}'.format(result['map']))
    # for l, name in enumerate(dataset.label_names):
    #     if result['ap'][l]:
    #         print('{:s}: {:f}'.format(name, result['ap'][l]))
    #     else:
    #         print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
