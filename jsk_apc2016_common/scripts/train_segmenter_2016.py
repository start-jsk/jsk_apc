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


def train(dataset_path):
    """
    Args:
        dataset_path (str): ex.. /home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data/tokyo_run
    """

    # initialize empty dataset
    dataset = APCDataSet(from_pkl=True)

    data_file_prefixes = []
    key = '_color.png'
    for dir_name, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if key == f[-len(key):]:
                data_file_prefixes.append(
                    os.path.join(dir_name, f[:-len(key)]))

    print data_file_prefixes


    for file_prefix in data_file_prefixes:
        dataset.samples.append(
            APCSample(data_2016_prefix=os.path.join(dataset_path, file_prefix),
                      labeled=True, is_2016=True))

    """
    dataset = APCDataSet(
        name=dataset_name,
        dataset_path=dataset_path,
        compute_from_images=True)
    """


if __name__ == '__main__':
    sys.modules['apc_data'] = apc_data
    data_path = common_path + '/data/'
    dataset_name = 'tokyo_run'
    dataset_path = os.path.join(data_path, dataset_name)
    train(dataset_path)
