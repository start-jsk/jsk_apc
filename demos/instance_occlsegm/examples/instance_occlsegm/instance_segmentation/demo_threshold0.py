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
from instance_occlsegm_lib.contrib import synthetic2d


def instance_to_semantic(img_shape, n_fg_class, bboxes, labels, masks, scores):
    assert masks.dtype == bool

    indices = np.argsort(scores)
    bboxes = bboxes[indices]
    labels = labels[indices]
    masks = masks[indices]
    scores = scores[indices]

    H, W = img_shape[:2]
    lbl_logits = np.full((H, W, n_fg_class), np.nan, dtype=np.float32)

    logits = np.log(scores / (1 - scores))

    n_instance = len(bboxes)
    for i in range(n_instance):
        y1, x1, y2, x2 = bboxes[i]
        label = labels[i]
        mask = masks[i]
        logit = logits[i]
        isnan = np.isnan(lbl_logits[:, :, label])
        mask_subst = np.logical_and(isnan, mask)
        lbl_logits[:, :, label][mask_subst] = logit
        mask_add = np.logical_and(~isnan, mask)
        lbl_logits[:, :, label][mask_add] += logit

    lbl_proba = np.full(lbl_logits.shape, np.nan, dtype=lbl_logits.dtype)
    isnan = np.isnan(lbl_logits)
    lbl_proba[~isnan] = 1 / (1 + np.exp(-lbl_logits[~isnan]))
    lbl_proba[isnan] = 0

    return lbl_proba


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
    mask_rcnn = cmr.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        anchor_scales=params['anchor_scales'],
        mean=params.get('mean', (123.152, 115.903, 103.063)),
        min_size=params['min_size'],
        max_size=params['max_size'],
        roi_size=params.get('roi_size', 7),
    )
    mask_rcnn.nms_thresh = 0
    mask_rcnn.score_thresh = 0
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    dataset = synthetic2d.datasets.ARC2017OcclusionDataset(split='test')
    # dataset = instance_occlsegm_lib.datasets.apc.\
    #     ARC2017InstanceSegmentationDataset('test')
    img = dataset[2][0]
    del dataset

    bboxes, masks, labels, scores = mask_rcnn.predict([img.transpose(2, 0, 1)])
    bbox, mask, label, score = bboxes[0], masks[0], labels[0], scores[0]
    del bboxes, masks, labels, scores

    thresh_proba = 0.7
    # thresh_proba = 0.5

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
    for caption in captions:
        print(caption)
    viz_ins = cmr.utils.draw_instance_bboxes(
        img,
        bbox_,
        label_ + 1,
        n_class=len(class_names) + 1,
        masks=mask_,
        captions=captions,
    )

    proba = instance_to_semantic(
        img.shape,
        n_fg_class=len(class_names),
        bboxes=bbox,
        labels=label,
        masks=mask,
        scores=score,
    )

    viz_proba = []
    for c in range(len(class_names)):
        proba_c = proba[:, :, c]
        viz_proba_c = instance_occlsegm_lib.image.overlay_color_on_mono(
            img_color=instance_occlsegm_lib.image.colorize_heatmap(proba_c),
            img_mono=img,
            alpha=0.5,
        )
        viz_proba.append(viz_proba_c)
    viz_proba = instance_occlsegm_lib.image.tile(viz_proba, boundary=True)

    max_proba = proba.max(axis=2)
    viz_maxp = instance_occlsegm_lib.image.colorize_heatmap(max_proba)

    proba_bg = np.full(
        (proba.shape[0], proba.shape[1], 1), 0, dtype=proba.dtype
    )
    proba_withbg = np.concatenate((proba_bg, proba), axis=2)
    lbl = np.argmax(proba_withbg, axis=2)
    viz_lbl = instance_occlsegm_lib.image.label2rgb(
        lbl,
        label_names=['__background__'] + class_names,
        thresh_suppress=0.01,
    )
    viz_lbl2 = instance_occlsegm_lib.image.label2rgb(
        lbl,
        img=img,
        label_names=['__background__'] + class_names,
        thresh_suppress=0.01,
    )

    viz = instance_occlsegm_lib.image.tile([
        img,
        viz_ins,
        viz_proba,
        viz_maxp,
        viz_lbl,
        viz_lbl2,
    ], boundary=True)
    instance_occlsegm_lib.io.imsave('logs/demo_threshold0.jpg', viz)


if __name__ == '__main__':
    main()
