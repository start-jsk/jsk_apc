#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint

import chainer
import numpy as np
import yaml

import chainer_mask_rcnn as cmr

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm
from instance_occlsegm_lib.contrib import synthetic2d


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    # XXX: see also evaluate.py
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # param
    params = yaml.load(open(osp.join(args.log_dir, 'params.yaml')))
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    # dataset
    if 'class_names' in params:
        class_names = params['class_names']
    elif params['dataset'] == 'voc':
        test_data = cmr.datasets.SBDInstanceSegmentationDataset('val')
        class_names = test_data.class_names
    elif params['dataset'] == 'coco':
        test_data = cmr.datasets.COCOInstanceSegmentationDataset('minival')
        class_names = test_data.class_names
    else:
        raise ValueError

    # model

    if params['dataset'] == 'voc':
        if 'min_size' not in params:
            params['min_size'] = 600
        if 'max_size' not in params:
            params['max_size'] = 1000
        if 'anchor_scales' not in params:
            params['anchor_scales'] = (4, 8, 16, 32)
    elif params['dataset'] == 'coco':
        if 'min_size' not in params:
            params['min_size'] = 800
        if 'max_size' not in params:
            params['max_size'] = 1333
        if 'anchor_scales' not in params:
            params['anchor_scales'] = (2, 4, 8, 16, 32)
    else:
        assert 'min_size' in params
        assert 'max_size' in params
        assert 'anchor_scales' in params

    if params['pooling_func'] == 'align':
        pooling_func = cmr.functions.roi_align_2d
    elif params['pooling_func'] == 'pooling':
        pooling_func = cmr.functions.roi_pooling_2d
    elif params['pooling_func'] == 'resize':
        pooling_func = cmr.functions.crop_and_resize
    else:
        raise ValueError(
            'Unsupported pooling_func: {}'.format(params['pooling_func'])
        )

    pretrained_model = osp.join(args.log_dir, 'snapshot_model.npz')
    print('Using pretrained_model: %s' % pretrained_model)

    model = params['model']
    mask_rcnn = instance_occlsegm.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        anchor_scales=params['anchor_scales'],
        mean=params.get('mean', (123.152, 115.903, 103.063)),
        min_size=params['min_size'],
        max_size=params['max_size'],
        roi_size=params.get('roi_size', 7),
        rpn_dim=params.get('rpn_dim', 1024),
    )
    mask_rcnn.nms_thresh = 0
    mask_rcnn.score_thresh = 0
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    dataset = synthetic2d.datasets.ARC2017OcclusionDataset(split='test')
    # dataset = instance_occlsegm_lib.datasets.apc.\
    #    ARC2017InstanceSegmentationDataset('test')
    # img = dataset[2][0]
    imgs = [dataset[2][0], dataset[3][0]]
    imgs_chw = [img.transpose(2, 0, 1) for img in imgs]
    del dataset

    bboxes, masks, labels, scores = mask_rcnn.predict(imgs_chw)

    # thresh_proba = 0.7
    thresh_proba = 0.5

    for i in range(len(imgs)):
        show(
            imgs[i],
            bboxes[i],
            masks[i],
            labels[i],
            scores[i],
            thresh_proba,
            class_names,
        )


def show(img, bbox, mask, label, score, thresh_proba, class_names):
    keep = score >= thresh_proba
    bbox_ = bbox[keep]
    mask_ = mask[keep]
    label_ = label[keep]
    score_ = score[keep]

    indices = np.argsort(score_)
    bbox_ = bbox_[indices]
    mask_ = mask_[indices]
    label_ = label_[indices]
    score_ = score_[indices]

    captions = [
        '{:s}: {:.1%}'.format(class_names[l], s)
        for l, s in zip(label_, score_)
    ]
    print('# ' + '-' * 77)
    for caption in captions:
        print(caption)
    print('# ' + '-' * 77)
    viz_ins = cmr.utils.draw_instance_bboxes(
        img,
        bbox_,
        label_ + 1,
        n_class=len(class_names) + 1,
        masks=mask_ == 1,
        captions=captions,
    )
    instance_occlsegm_lib.io.imshow(viz_ins)
    instance_occlsegm_lib.io.waitkey()


if __name__ == '__main__':
    main()
