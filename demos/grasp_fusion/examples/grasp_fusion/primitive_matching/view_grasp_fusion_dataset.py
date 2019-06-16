#!/usr/bin/env python

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion


dataset_pinch = grasp_fusion.datasets.PinchDataset('train')
dataset_suction = grasp_fusion.datasets.SuctionDataset('train')

# All of color, depth, label are heightmap from viewpoint above the tote
# with grid mapping by 0.002m voxels, so 1pixel = 0.002m = 2mm.
# Pinch's label (label1) has shape (height, width, 6)
# and has value represents -1: unlabeled, 0: bad, 1: good.
# The z-axis rotation is represented by its channel (6 channels):
# (0,30,60,90,120,150).
# Suction's label (label2) has shape (height, width)
# and has value represents -1: unlabeled, 0: bad, 1: good.
#
#     color1, depth1, label1 = dataset_pinch[0]
#     color2, depth2, label2 = dataset_suction[0]

viz1 = dataset_pinch.visualize(0)
viz2 = dataset_suction.visualize(0)
viz = grasp_fusion_lib.image.tile([viz1, viz2], (2, 1), boundary=True)

# grasp_fusion_lib.io.imsave('view_collaborative_dataset.jpg', viz)
grasp_fusion_lib.io.imshow(viz)
grasp_fusion_lib.io.waitkey()
