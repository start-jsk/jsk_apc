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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument('--images-dir', help='images dir')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    with open(osp.join(args.log_dir, 'params.yaml')) as f:
        params = yaml.load(f)

    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    chainer.cuda.get_device_from_id(args.gpu).use()

    class_names = contrib.datasets.ARC2017RealDataset('train').class_names
    fg_class_names = class_names[1:]

    # Default Config
    assert params['pooling_func'] == 'align'
    pooling_func = mrcnn.functions.roi_align_2d

    mask_rcnn = contrib.models.MaskRCNNResNet(
        n_layers=int(params['model'].lstrip('resnet')),
        n_fg_class=len(fg_class_names),
        pretrained_model=osp.join(args.log_dir, 'snapshot_model.npz'),
        pooling_func=pooling_func,
        anchor_scales=params['anchor_scales'],
        min_size=params['min_size'],
        max_size=params['max_size'],
        mask_loss=params['mask_loss'],
    )
    mask_rcnn.nms_thresh = 0.3
    mask_rcnn.to_gpu()

    score_thresh = 0.6
    for img_file in sorted(glob.glob(osp.join(args.images_dir, '*.jpg'))):
        print('Infering for: %s' % img_file)
        img = skimage.io.imread(img_file)
        img_chw = img.transpose(2, 0, 1)

        bboxes, masks, labels, scores = mask_rcnn.predict_masks([img_chw])

        bboxes = bboxes[0]
        masks = masks[0]
        labels = labels[0]
        scores = scores[0]

        keep = scores >= score_thresh
        bboxes = bboxes[keep]
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]

        captions = ['{}: {:.2%}'.format(fg_class_names[l], s)
                    for l, s in zip(labels, scores)]

        for i in range(len(bboxes) + 1):
            if i == len(bboxes):
                draw = [True] * len(bboxes)
            else:
                draw = [False] * len(bboxes)
                draw[i] = True
            viz0 = mrcnn.utils.draw_instance_bboxes(
                img,
                bboxes,
                labels,
                n_class=40,
                bg_class=-1,
                masks=masks == 0,
                captions=captions,
                draw=draw,
            )
            viz1 = mrcnn.utils.draw_instance_bboxes(
                img,
                bboxes,
                labels,
                n_class=40,
                bg_class=-1,
                masks=masks == 1,
                captions=captions,
                draw=draw,
            )
            viz2 = mrcnn.utils.draw_instance_bboxes(
                img,
                bboxes,
                labels,
                n_class=40,
                bg_class=-1,
                masks=masks == 2,
                captions=captions,
                draw=draw,
            )

            viz = np.hstack([img, viz0, viz1, viz2])
            base, ext = osp.splitext(img_file)
            out_dir = osp.join(osp.dirname(img_file), 'out_infer_mrcnn_lbl')
            try:
                os.makedirs(out_dir)
            except OSError:
                pass
            out_file = osp.join(
                out_dir,
                '%s_%08d.jpg' % (osp.splitext(osp.basename(img_file))[0], i),
            )
            skimage.io.imsave(out_file, viz)
            print('Saved result to: %s' % out_file)


if __name__ == '__main__':
    main()
