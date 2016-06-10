#!/usr/bin/env python
import sys
import os
import rospkg
from jsk_apc2016_common.rbo_segmentation.apc_data import APCDataSet, APCSample
from jsk_apc2016_common.rbo_segmentation.probabilistic_segmentation\
        import ProbabilisticSegmentationBP
import jsk_apc2016_common.rbo_segmentation.apc_data as apc_data
import pickle

import cv2
import numpy as np

import matplotlib.pyplot as plt

rospack = rospkg.RosPack()
common_path = rospack.get_path('jsk_apc2016_common')
module_path = common_path + '/python/jsk_apc2016_common'
params = {
        'use_features': ['color', 'dist2shelf', 'height3D'],
        'segmentation_method': "max_smooth", 'selection_method': "max_smooth",
        'make_convex': True, 'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True}


def create_apc_sample(path):
    """Create APCSample from path

    Args:
        path (str): example... 'save_pick_layout_1_2016061006_bin_l'

    """
    data = {}
    with open(path + '.pkl', 'rb') as f:
        data = pickle.load(f)

    data['color'] = cv2.cvtColor(cv2.imread(path + '_color.png'), cv2.COLOR_BGR2HSV)

    data['mask_image'] = np.sum(cv2.imread(path + '_mask.pbm'), axis=2).astype(np.bool)

    # scale x5 is due to saving procedure
    data['depth_image'] = cv2.imread(path + '_depth.png').astype(np.float32) * 5.
    data['dist2shelf_image'] = cv2.imread(path + '_dist.png').astype(np.float32)
    data['height3D_image'] = cv2.imread(path + '_height.png').astype(np.float32)

    # complete neccesary data
    data['height2D_image'] = np.zeros_like(data['height3D_image'])
    data['has3D_image'] = (data['depth_image'] > 0).astype(np.uint8)

    # read label and set label
    apc_sample = APCSample(labeled=False, data_dict=data)
    return apc_sample




def train(pkl_path):
    sys.modules['apc_data'] = apc_data

    data_path = common_path + '/data/'
    dataset_name = 'tokyo_run/1'
    dataset_path = os.path.join(data_path, dataset_name)

    create_apc_sample(os.path.join(
        dataset_path, 'save_pick_layout_1_2016061006_bin_l'))




    """
    dataset = APCDataSet(
        name=dataset_name,
        dataset_path=dataset_path,
        compute_from_images=True)
    """







if __name__ == '__main__':
    train(common_path + '/data/trained_segmenter_2016.pkl')
