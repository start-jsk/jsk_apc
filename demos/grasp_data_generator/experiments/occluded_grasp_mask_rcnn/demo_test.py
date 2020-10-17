#!/usr/bin/env python

from __future__ import division

import argparse
import easydict
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import yaml

import chainer
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox

from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV1
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV2
from grasp_data_generator.datasets import FinetuningOIDualarmGraspDatasetV3
from grasp_data_generator.datasets import OIDualarmGraspDatasetV1
from grasp_data_generator.datasets import OIDualarmGraspDatasetV2
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV1
from grasp_data_generator.datasets import OIRealAnnotatedDatasetV2
from grasp_data_generator.models.occluded_grasp_mask_rcnn.utils \
    import rot_to_rot_lbl
from grasp_data_generator.models import OccludedGraspMaskRCNNResNet101
from grasp_data_generator.visualizations \
    import vis_occluded_grasp_instance_segmentation


thisdir = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.', default=0)
    parser.add_argument('--yaml', type=str)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--grasp-thresh', type=float, default=None)
    parser.add_argument('--no-rot', action='store_true')
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
            split='all', imgaug=False)
    elif params.dataset == 'ev2':
        test_data = OIRealAnnotatedDatasetV2(
            split='all', imgaug=False)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(params.dataset))

    if 'rotate_angle' not in params:
        rotate_angle = None
    else:
        rotate_angle = params.rotate_angle

    label_names = test_data.label_names
    model = OccludedGraspMaskRCNNResNet101(
        n_fg_class=len(label_names),
        anchor_scales=params.anchor_scales,
        min_size=params.min_size,
        max_size=params.max_size,
        rpn_dim=params.rpn_dim,
        rotate_angle=rotate_angle)
    model.nms_thresh = 0.3
    model.score_thresh = 0.5
    chainer.serializers.load_npz(
        osp.join(thisdir, args.pretrained_model), model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    grasp_thresh = args.grasp_thresh

    for i in range(len(test_data)):
        in_data = test_data[i]
        if rotate_angle is None:
            img, gt_ins_label, gt_label, gt_sg_mask, gt_dg_mask = in_data
        else:
            img, gt_ins_label, gt_label = in_data[:3]
            if len(in_data) == 5:
                gt_sg_mask, gt_dg_mask = in_data[3:]
                gt_rotation = None
            elif len(in_data) == 6:
                gt_sg_mask, gt_dg_mask, gt_rotation = in_data[3:]
            else:
                gt_sg_mask, gt_dg_mask, gt_rotation = None, None, None

        f, (axes1, axes2) = plt.subplots(2, 5, sharey=True)
        f.canvas.set_window_title('visualization: prediction vs gt')

        # prediction
        if args.no_rot:
            try:
                ins_labels, ins_probs, labels, bboxes, scores, \
                    _, sg_probs, _, dg_probs = model.predict(
                            [img], return_probs=True)
                ins_label, label, bbox, score \
                    = ins_labels[0], labels[0], bboxes[0], scores[0]
                ins_prob, sg_prob, dg_prob = \
                    ins_probs[0], sg_probs[0], dg_probs[0]
                if grasp_thresh is None:
                    sg_label = np.sum(sg_prob[:, 1:], axis=1) > 0.5
                    sg_label = sg_label.astype(np.int32)
                    dg_label = np.sum(dg_prob[:, 1:], axis=1) > 0.5
                    dg_label = dg_label.astype(np.int32)
                else:
                    sg_ins_prob = ins_prob[:, 1, :, :] * \
                        np.sum(sg_prob[:, 1:, :, :], axis=1)
                    sg_label = (sg_ins_prob > grasp_thresh).astype(np.int32)
                    dg_ins_prob = ins_prob[:, 1, :, :] * \
                        np.sum(dg_prob[:, 1:, :, :], axis=1)
                    dg_label = (dg_ins_prob > grasp_thresh).astype(np.int32)
                vis_occluded_grasp_instance_segmentation(
                    img, ins_label, label, bbox, score,
                    sg_label, dg_label, label_names,
                    rotate_angle=None, prefix='pred', axes=axes1)
            except IndexError:
                print('no predict returned')
        else:
            ins_labels, labels, bboxes, scores, sg_labels, dg_labels = \
                model.predict([img], return_probs=False)
            try:
                ins_label, label, bbox, score \
                    = ins_labels[0], labels[0], bboxes[0], scores[0]
                sg_label, dg_label = sg_labels[0], dg_labels[0]
                vis_occluded_grasp_instance_segmentation(
                    img, ins_label, label, bbox, score,
                    sg_label, dg_label, label_names,
                    rotate_angle=rotate_angle,
                    prefix='pred', axes=axes1)
            except IndexError:
                print('no predict returned')

        # gt
        gt_bbox = mask_to_bbox(gt_ins_label != 0)
        if gt_sg_mask is None or gt_dg_mask is None:
            gt_sg_label = np.zeros(gt_ins_label.shape, dtype=np.int32)
            gt_dg_label = np.zeros(gt_ins_label.shape, dtype=np.int32)
        else:
            if rotate_angle is None:
                gt_sg_label = gt_sg_mask.astype(np.int32)
                gt_dg_label = gt_dg_mask.astype(np.int32)
            else:
                gt_sg_label = []
                gt_dg_label = []
                for gt_rot, gt_sg_msk, gt_dg_msk in zip(
                        gt_rotation, gt_sg_mask, gt_dg_mask):
                    gt_rot_lbl = rot_to_rot_lbl(gt_rot, rotate_angle)
                    gt_sg_lbl = gt_sg_msk.astype(np.int32) * gt_rot_lbl
                    gt_dg_lbl = gt_dg_msk.astype(np.int32) * gt_rot_lbl
                    gt_sg_label.append(gt_sg_lbl[None])
                    gt_dg_label.append(gt_dg_lbl[None])
                gt_sg_label = np.concatenate(gt_sg_label, axis=0)
                gt_dg_label = np.concatenate(gt_dg_label, axis=0)
        vis_occluded_grasp_instance_segmentation(
            img, gt_ins_label, gt_label, gt_bbox, None,
            gt_sg_label, gt_dg_label, label_names,
            rotate_angle=rotate_angle,
            prefix='gt', axes=axes2)

        plt.show()


if __name__ == '__main__':
    main()
