#!/usr/bin/env python

import glob
import grasp_fusion_lib
import os.path as osp
import skimage.io


home = osp.expanduser('~')
masks_dir = osp.join(
    home, 'data/grasp_fusion_lib/grasp_fusion/ItemDataAll_MaskPred')

if not osp.exists(masks_dir):
    print('please run ./view_dataset.py')
    quit()

vizs = []
for dir in glob.glob(osp.join(masks_dir, '*')):
    # Try to display front image
    img_list = sorted(glob.glob(dir + '/*.png'), reverse=True)
    img = skimage.io.imread(img_list[0])
    vizs.append(img)
# len(vizs) = 112
viz = grasp_fusion_lib.image.tile(vizs, (len(vizs) // 14, 14), boundary=True)
viz = grasp_fusion_lib.image.resize(viz, width=2500)
grasp_fusion_lib.io.imshow(viz)
grasp_fusion_lib.io.waitkey(0)
