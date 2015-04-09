#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
This script is to extract sift features from object data
distributed by Robot Learning Lab, UC Berkeley.
Object data is available from here::

    * http://rll.berkeley.edu/amazon_picking_challenge/


Usage
-----
1. Download dataset(Raw High Resolution RGB) to data dir, and extract it.
2. Execute following::

    $ roslaunch jsk_2014_picking_challenge sift_matcher.launch
    $ rosrun extract_sift_from_objdata.py _object:=oreo_mega_stuf


Attention
---------
You should change dirname for following items manually::

    * kygen_squeakin_eggs_plush_puppies  -> kyjen_squeakin_eggs_plush_puppies
    * rollodex_mesh_collection_jumbo_pencil_cup -> rolodex_jumbo_pencil_cup

"""
import cv2
import numpy as np
import progressbar

import rospy
from sensor_msgs.msg import CameraInfo

from sift_matcher import imgsift_client, load_siftdata
from matcher_common import save_siftdata, get_object_list, get_train_imgpaths


def extract_sift_from_objdata(obj_name):
    """Extract sift data from object images"""
    positions = []
    descriptors = []
    imgpaths = get_train_imgpaths(obj_name)
    if imgpaths is None:
        return   # skip if img does not exists
    progress = progressbar.ProgressBar(widgets=['{o}: '.format(o=obj_name),
        progressbar.Bar(), progressbar.Percentage(), ' ', progressbar.ETA()])
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
    rospy.init_node('extract_sift_from_objdata')
    pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=1)
    pub.publish()  # to enable imagesift service

    all_objects = get_object_list()

    obj_names = rospy.get_param('~object',
                                'oreo_mega_stuf,safety_works_safety_glasses')
    obj_names = obj_names.split(',')
    if len(obj_names) == 1 and obj_names[0] == 'all':
        obj_names = all_objects
    rospy.loginfo('objects: {obj}'.format(obj=obj_names))

    for obj_name in obj_names:
        if obj_name not in all_objects:
            rospy.logwarn('Unknown object, skipping: {o}'.format(o=obj_name))
            continue
        elif load_siftdata(obj_name, dry_run=True):
            continue  # already extracted
        extract_sift_from_objdata(obj_name)


if __name__ == '__main__':
    main()
