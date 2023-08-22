#!/usr/bin/env python

import sys
import os
import rospkg
from jsk_apc2016_common.rbo_segmentation.apc_data import APCDataSet, APCSample
from jsk_apc2016_common.rbo_segmentation.probabilistic_segmentation\
    import ProbabilisticSegmentationBP
import jsk_apc2016_common.rbo_segmentation.apc_data as apc_data
import pickle


rospack = rospkg.RosPack()
common_path = rospack.get_path('jsk_apc2016_common')
module_path = common_path + '/python/jsk_apc2016_common'


def train(dataset_path, params):
    """
    Args:
        dataset_path (str): example
            /home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data/tokyo_run
    """
    pkl_path = common_path + '/data/trained_segmenter_2016.pkl'

    bp = ProbabilisticSegmentationBP(**params)

    # initialize empty dataset
    dataset = APCDataSet(from_pkl=True)

    data_file_prefixes = []
    key = '.jpg'
    for dir_name, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if key == f[-len(key):]:
                data_file_prefixes.append(
                    os.path.join(dir_name, f[:-len(key)]))

    print(data_file_prefixes)
    for file_prefix in data_file_prefixes:
        dataset.samples.append(
            APCSample(data_2016_prefix=file_prefix,
                      labeled=True, is_2016=True, infer_shelf_mask=True))

    bp.fit(dataset)
    print("done fitting")

    with open(pkl_path, 'wb') as f:
        pickle.dump(bp, f)
    print("done dumping model")

    # with open(common_path + '/data/dataset.pkl', 'wb') as f:
    #    pickle.dump(dataset, f)
    # print "done saving dataset as pkl"


if __name__ == '__main__':
    params = {
        'use_features': ['color', 'dist2shelf', 'height3D'],
        'segmentation_method': "max_smooth", 'selection_method': "max_smooth",
        'make_convex': True, 'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True}

    sys.modules['apc_data'] = apc_data
    data_path = common_path + '/data/'
    dataset_name = 'tokyo_run/single_item_labeled'
    dataset_path = os.path.join(data_path, dataset_name)
    train(dataset_path, params)
