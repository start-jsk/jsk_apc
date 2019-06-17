#!/usr/bin/env python

import os
import os.path as osp

import chainer_mask_rcnn as cmr
import numpy as np
import skimage.color

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


out_dir = 'logs/input_output'
try:
    os.makedirs(out_dir)
except OSError:
    pass

dataset = instance_occlsegm.datasets.PanopticOcclusionSegmentationDataset(
    'train')
class_names = dataset.class_names
img, bboxes, labels, masks, lbl_vis, lbl_occ = dataset[174]

offset_y = 130
offset_x = 100
img = img[offset_y:, offset_x:]
masks = masks[:, offset_y:, offset_x:]
bboxes[:, 0] -= offset_y
bboxes[:, 2] -= offset_y
bboxes[:, 1] -= offset_x
bboxes[:, 3] -= offset_x
lbl_vis = lbl_vis[offset_y:, offset_x:]
lbl_occ = lbl_occ[offset_y:, offset_x:]

instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'image.jpg'), img)

ratios = 1. * (masks == 1).sum(axis=(1, 2)) / np.isin(masks, (1, 2)).sum(axis=(1, 2))  # NOQA
keep = ratios > 0.05
bboxes = bboxes[keep]
labels = labels[keep]
masks = masks[keep]

captions = [class_names[l] for l in labels]
viz = cmr.utils.draw_instance_bboxes(
    img,
    bboxes,
    labels + 1,
    n_class=len(class_names) + 1,
    masks=masks == 1,
    captions=captions,
)
instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'instance_visible.jpg'), viz)
# instance_occlsegm_lib.io.imgplot(viz)
# instance_occlsegm_lib.io.show()

img_gray = skimage.color.rgb2gray(img)
img_gray = skimage.color.gray2rgb(img_gray)
img_gray = (img_gray * 255).astype(np.uint8)

viz = []
for i in range(len(bboxes)):
    draw = np.zeros((len(bboxes),), dtype=bool)
    draw[i] = True
    bbox = bboxes[i]
    mask = masks[i]
    label = labels[i]
    caption = class_names[label]
    # img2 = img.copy()
    # img2[masks[i] == 0] = img_gray[masks[i] == 0]
    viz.append(cmr.utils.draw_instance_bboxes(
        # img2,
        img_gray,
        [bbox, bbox],
        [label + 1, label + 1],
        n_class=len(class_names) + 1,
        masks=[mask == 1, mask == 2],
        captions=[caption, caption],
    ))
for i, v in enumerate(viz):
    instance_occlsegm_lib.io.imsave(
        osp.join(out_dir, 'instance_occluded/%04d.jpg' % i), v)
# viz = instance_occlsegm_lib.image.tile(viz, boundary=True)
# instance_occlsegm_lib.io.imsave(
#     osp.join(out_dir, 'instance_occluded.jpg'), viz)
# instance_occlsegm_lib.io.imgplot(viz)
# instance_occlsegm_lib.io.show()

viz = instance_occlsegm_lib.image.label2rgb(
    lbl_vis,
    img,
    label_names=[''] + class_names.tolist()
)
instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'semantic_visible.jpg'), viz)
# instance_occlsegm_lib.io.imgplot(viz)
# instance_occlsegm_lib.io.show()

viz = []
for c in labels:
    lbl_occ_c = lbl_occ[:, :, c].copy()
    if 1. * (lbl_occ_c == 1).sum() / lbl_occ_c.size < 0.01:
        continue
    lbl_occ_c[lbl_occ_c == 1] = c + 1
    viz.append(instance_occlsegm_lib.image.label2rgb(
        lbl_occ_c,
        img,
        # n_labels=len(class_names) + 1,
        label_names=[''] + class_names.tolist()
    ))
for i, v in enumerate(viz):
    instance_occlsegm_lib.io.imsave(
        osp.join(out_dir, 'semantic_occluded/%04d.jpg' % i), v)
# viz = instance_occlsegm_lib.image.tile(viz, boundary=True)
# instance_occlsegm_lib.io.imsave(
#    osp.join(out_dir, 'semantic_occluded.jpg'), viz)
# instance_occlsegm_lib.io.imgplot(viz)
# instance_occlsegm_lib.io.show()
