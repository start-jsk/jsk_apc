#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np

import rospy

from sift_matcher import imgsift_client, load_siftdata
from common import save_siftdata, get_train_imgs

import jsk_2015_apc_common


def extract_sift(obj_name):
    """Extract sift data from object images"""
    positions = []
    descriptors = []
    data_dir = rospy.get_param('~train_data', None)
    only_appropriate = rospy.get_param('~only_appropriate', True)
    with_mask = rospy.get_param('~with_mask', True)
    train_imgs = get_train_imgs(obj_name=obj_name,
                                data_dir=data_dir,
                                only_appropriate=only_appropriate,
                                with_mask=with_mask)
    for train_img in train_imgs:
        train_features = imgsift_client(train_img)
        train_pos = np.array(train_features.positions)
        train_des = np.array(train_features.descriptors)
        positions.append(train_pos)
        descriptors.append(train_des)
    if len(positions) == 0 or len(descriptors) == 0:
        rospy.logerr('no images found: {0}'.format(obj_name))
        return
    positions, descriptors = map(np.array, [positions, descriptors])
    siftdata = dict(positions=positions, descriptors=descriptors)
    # save sift data
    save_siftdata(obj_name, siftdata)


def main():
    rospy.init_node('extract_sift')
    obj_names = jsk_2015_apc_common.data.object_list()
    for obj_name in obj_names:
        if load_siftdata(obj_name, dry_run=True):
            continue  # already extracted
        extract_sift(obj_name)


if __name__ == '__main__':
    main()
