#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import cv2
import numpy as np
import progressbar

import rospy

from sift_matcher import imgsift_client, load_siftdata
from matcher_common import save_siftdata, get_object_list, get_train_imgpaths


def extract_sift(obj_name):
    """Extract sift data from object images"""
    positions = []
    descriptors = []
    imgpaths = get_train_imgpaths(obj_name)
    if imgpaths is None or len(imgpaths) == 0:
        return   # skip if img does not exists
    widget = ['{0}: '.format(obj_name), progressbar.Bar(),
              progressbar.Percentage(), ' ', progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widget)
    for raw_path, mask_path in progress(imgpaths):
        raw_img = cv2.imread(raw_path)
        mask_img = cv2.imread(mask_path)
        train_img = cv2.add(mask_img, raw_img)
        train_features = imgsift_client(train_img)
        train_pos = np.array(train_features.positions)
        train_des = np.array(train_features.descriptors)
        positions.append(train_pos)
        descriptors.append(train_des)
    positions, descriptors = map(np.array, [positions, descriptors])
    siftdata = dict(positions=positions, descriptors=descriptors)
    # save sift data
    save_siftdata(obj_name, siftdata)


def main():
    rospy.init_node('extract_sift')
    obj_names = get_object_list()
    for obj_name in obj_names:
        if load_siftdata(obj_name, dry_run=True):
            continue  # already extracted
        extract_sift(obj_name)


if __name__ == '__main__':
    main()
