#!/usr/bin/env python

import argparse
import glob
import os
import os.path as osp
import pprint

import chainer
import numpy as np
import skimage.io
import yaml

import chainer_mask_rcnn as mrcnn

import contrib


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('frames_dir', help='contain dir of sequential frames')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    log_dir = 'logs/train_mrcnn/20180204_140810'
    params = yaml.load(open(osp.join(log_dir, 'params.yaml')))
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    chainer.cuda.get_device_from_id(args.gpu).use()

    class_names = contrib.datasets.ARC2017RealDataset('train').class_names
    fg_class_names = class_names[1:]

    # Default Config
    min_size = 600
    max_size = 1000
    anchor_scales = [4, 8, 16, 32]
    proposal_creator_params = dict(
        n_train_pre_nms=12000,
        n_train_post_nms=2000,
        n_test_pre_nms=6000,
        n_test_post_nms=1000,
        min_size=0,
    )
    assert params['pooling_func'] == 'align'
    pooling_func = mrcnn.functions.roi_align_2d

    mask_rcnn = mrcnn.models.MaskRCNNResNet(
        n_layers=101,
        n_fg_class=len(fg_class_names),
        pretrained_model=osp.join(log_dir, 'snapshot_model.npz'),
        pooling_func=pooling_func,
        anchor_scales=anchor_scales,
        proposal_creator_params=proposal_creator_params,
        min_size=min_size,
        max_size=max_size,
    )
    mask_rcnn.use_preset('visualize')
    mask_rcnn.nms_thresh = 0.3
    mask_rcnn.score_thresh = 0.6
    mask_rcnn.to_gpu()

    bboxes_, masks_, labels_ = None, None, None
    for img_file in sorted(glob.glob(osp.join(args.frames_dir, '*.jpg'))):
        print('Infering for: %s' % img_file)
        img = skimage.io.imread(img_file)
        img_chw = img.transpose(2, 0, 1)

        bboxes, masks, labels, scores = mask_rcnn.predict_masks([img_chw])

        bboxes = bboxes[0]
        masks = masks[0]
        labels = labels[0]
        scores = scores[0]

        captions = ['{}: {:.2%}'.format(fg_class_names[l], s)
                    for l, s in zip(labels, scores)]
        viz1 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks, captions=captions)

        if bboxes_ is None:
            masks_inv = None
        else:
            masks_inv = []
            for i in range(len(bboxes)):
                mask = masks[i]
                label = labels[i]

                mask_inv = np.zeros_like(mask)
                for j in range(len(bboxes_)):
                    label_ = labels_[j]
                    mask_ = masks_[j]
                    if label_ == label:
                        continue
                    mask_inv = np.bitwise_or(
                        mask_inv,
                        np.bitwise_and(mask, mask_),  # invisible region
                    )
                masks_inv.append(mask_inv)

        viz2 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks_inv, captions=captions)

        viz = np.hstack([img, viz1, viz2])
        base, ext = osp.splitext(img_file)
        out_dir = osp.join(osp.dirname(img_file), 'out')
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
        out_file = osp.join(out_dir, osp.basename(img_file))
        skimage.io.imsave(out_file, viz)
        print('Saved result to: %s' % out_file)

        bboxes_, masks_, labels_ = bboxes, masks, labels


if __name__ == '__main__':
    main()
