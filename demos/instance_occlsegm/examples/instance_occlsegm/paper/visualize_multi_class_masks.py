#!/usr/bin/env python

import os
import os.path as osp

import chainer_mask_rcnn as cmr
import numpy as np
import skimage.color

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


data = instance_occlsegm.datasets.PanopticOcclusionSegmentationDataset('train')
class_names = data.class_names
img, bboxes, labels, masks, _, _ = data[1]

keep = labels == 16
bboxes = bboxes[keep]
labels = labels[keep]
masks = masks[keep]
assert keep.sum() == 1

img_org = img.copy()

mask = masks[0]
img_gray = skimage.color.rgb2gray(img)
img_gray = skimage.color.gray2rgb(img_gray)
img_gray = (img_gray * 255).astype(np.uint8)
img[mask != 1] = img_gray[mask != 1]

# captions = ['{}: {}'.format(l, class_names[l]) for l in labels]
captions = [class_names[l] for l in labels]
for caption in captions:
    print(caption)

viz_org = cmr.utils.draw_instance_bboxes(
    img_org,
    bboxes,
    labels + 1,
    n_class=len(class_names) + 1,
    captions=captions,
)
viz_bg = cmr.utils.draw_instance_bboxes(
    img,
    bboxes,
    labels + 1,
    masks=masks == 0,
    n_class=len(class_names) + 1,
    captions=captions,
)
viz_vis = cmr.utils.draw_instance_bboxes(
    img,
    bboxes,
    labels + 1,
    masks=masks == 1,
    n_class=len(class_names) + 1,
    captions=captions,
)
viz_occ = cmr.utils.draw_instance_bboxes(
    img,
    bboxes,
    labels + 1,
    masks=masks == 2,
    n_class=len(class_names) + 1,
    captions=captions,
)

bbox = bboxes[0]
y1, x1, y2, x2 = [int(round(x)) for x in bbox]
viz_bg = viz_bg[y1:y2, x1:x2]
viz_vis = viz_vis[y1:y2, x1:x2]
viz_occ = viz_occ[y1:y2, x1:x2]

offset_y = 60
viz_org = viz_org[offset_y:-30, 30:-40]
# viz_bg = viz_bg[offset_y:, :]
# viz_vis = viz_vis[offset_y:, :]
# viz_occ = viz_occ[offset_y:, :]

viz_bg = instance_occlsegm_lib.image.resize(viz_bg, height=viz_org.shape[0])
viz_vis = instance_occlsegm_lib.image.resize(viz_vis, height=viz_org.shape[0])
viz_occ = instance_occlsegm_lib.image.resize(viz_occ, height=viz_org.shape[0])

out_dir = 'logs/multi_class_masks'
try:
    os.makedirs(out_dir)
except OSError:
    pass

instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'image.jpg'), viz_org)
instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'background.jpg'), viz_bg)
instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'visible.jpg'), viz_vis)
instance_occlsegm_lib.io.imsave(osp.join(out_dir, 'occluded.jpg'), viz_occ)

# viz_masks = np.hstack([viz_bg, viz_vis, viz_occ])
# viz_org = instance_occlsegm_lib.image.resize(
#     viz_org, width=viz_masks.shape[1])
# viz_masks = instance_occlsegm_lib.image.resize(
#     viz_masks, width=viz_org.shape[1])
# viz = np.vstack([viz_org, viz_masks])
# viz = instance_occlsegm_lib.image.tile(
#     [img_org, viz_bg, viz_vis, viz_occ], (1, 4), boundary=True
# )
# instance_occlsegm_lib.io.imsave('result.jpg', viz)
# instance_occlsegm_lib.io.imshow(viz)
# instance_occlsegm_lib.io.waitkey()
