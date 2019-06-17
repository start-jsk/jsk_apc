#!/usr/bin/env python

import os
import os.path as osp

import numpy as np

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


def get_data():
    dataset = instance_occlsegm_lib.datasets.apc.\
        ARC2017InstanceSegmentationDataset(split='train')

    img, bboxes, labels, masks = dataset[0]
    fg_class_names = dataset.class_names
    class_names = tuple(['__background__'] + list(fg_class_names))
    n_fg_class = len(fg_class_names)

    n_instance = len(bboxes)
    mask_n_classes = []
    for i in range(n_instance):
        bbox = bboxes[i]
        label = labels[i]
        mask = masks[i]

        y1, x1, y2, x2 = bbox.astype(int)

        mask = mask[y1:y2, x1:x2]
        fg = mask.astype(bool)
        mask = mask.astype(np.float32)
        mask[fg] = np.random.uniform(0.75, 0.95, size=fg.sum())
        mask[~fg] = np.random.uniform(0.05, 0.25, size=(~fg).sum())
        mask = instance_occlsegm_lib.image.resize(mask, height=14, width=14)

        mask_n_class = np.zeros((n_fg_class, 14, 14))
        mask_n_class = mask_n_class.astype(np.float32)
        mask_n_class[label] = mask
        mask_n_classes.append(mask_n_class)
    mask_n_classes = np.asarray(mask_n_classes)

    return img, bboxes, labels, mask_n_classes, class_names


def main():
    out_dir = 'logs/sample_roi_unpooling_2d'
    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    img, bboxes, labels, masks, class_names = get_data()

    x = masks
    outh, outw = img.shape[:2]
    rois = bboxes.astype(np.float32)
    roi_indices = np.zeros((len(rois), 1), dtype=np.float32)
    indices_and_rois = np.hstack((roi_indices, rois))

    y = instance_occlsegm.functions.roi_unpooling_2d(
        x,
        indices_and_rois,
        outb=1,
        outh=outh,
        outw=outw,
        spatial_scale=1,
        axes='yx',
    )
    y = y[0].array

    imgs = []
    for yi in y:
        # print(yi.min(), yi.max())
        imgs.append(instance_occlsegm_lib.image.colorize_heatmap(yi))
    viz = instance_occlsegm_lib.image.tile(imgs, boundary=True)
    instance_occlsegm_lib.io.imsave(osp.join(out_dir, '001.jpg'), viz)

    proba = y
    max_proba = proba.max(axis=0)
    viz = instance_occlsegm_lib.image.colorize_depth(max_proba)
    instance_occlsegm_lib.io.imsave(osp.join(out_dir, '002.jpg'), viz)
    bg = max_proba < 0.5
    lbl = np.argmax(proba, axis=0) + 1
    lbl[bg] = 0

    viz = instance_occlsegm_lib.image.label2rgb(
        lbl, img=img, label_names=class_names)
    instance_occlsegm_lib.io.imsave(osp.join(out_dir, '003.jpg'), viz)

    print('Write to:', out_dir)


if __name__ == '__main__':
    main()
