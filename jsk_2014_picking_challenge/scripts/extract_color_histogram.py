#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
This script is to extract color histogram from object data
distributed by Robot Learning Lab, UC Berkeley.
Object data is available from here::

    * http://rll.berkeley.edu/amazon_picking_challenge/

Usage
-----
1. Download dataset(Raw High Resolution RGB) to data dir, and extract it.
2. Execute following::

    $ roscore
    $ rosrun imagesift imagesift
    $ rosrun extract_sift_from_objdata.py _object:=oreo_mega_stuf


Attention
---------
You should change dirname for following items manually::

    * kygen_squeakin_eggs_plush_puppies  -> kyjen_squeakin_eggs_plush_puppies
    * rollodex_mesh_collection_jumbo_pencil_cup -> rolodex_jumbo_pencil_cup

"""
import os
import sys
import collections
import gzip
import cPickle as pickle

import cv2
import numpy as np
import yaml
import progressbar

import rospy
from sensor_msgs.msg import CameraInfo
from posedetection_msgs.srv import Feature0DDetect
from posedetection_msgs.msg import ImageFeature0D

from sift_matcher_oneimg import SiftMatcherOneImg

from extract_sift_from_objdata import get_train_imgpaths

def save_color_hist_data(histdata, obj_name):
    """Save color histogram data to data/histdata/{obj_name}.pkl.gz"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    histdata_dir = os.path.join(dirname, '../data/histdata')
    if not os.path.exists(histdata_dir):
        os.mkdir(histdata_dir)
    filename = os.path.join(histdata_dir, obj_name+'.pkl.gz')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(histdata, f)

def extract_color_histogram_from_objdata(obj_name):
    """Extract color histogram data from object images"""
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
        # train_features = HistMatcherOneImg.imghist_client(train_img)
        # change here.
        #
        train_pos = np.array(train_features.positions)
        train_des = np.array(train_features.descriptors)
        positions.append(train_pos)
        descriptors.append(train_des)
    positions, descriptors = map(np.array, [positions, descriptors])
    histdata = dict(positions=positions, descriptors=descriptors)
    # save hist data
    save_histdata(histdata, obj_name)


def main():
    rospy.init_node('extract_color_histogram_from_objdata')
    pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=1)
    pub.publish()  # to enable imagehist service

    dirname = os.path.dirname(os.path.abspath(__file__))
    ymlfile = os.path.join(dirname, '../data/object_list.yml')
    all_objects = yaml.load(open(ymlfile))

    obj_names = rospy.get_param('~object',
                                'oreo_mega_stuf,safety_works_safety_glasses')
    obj_names = obj_names.split(',')
    if len(obj_names) == 1 and obj_names[0] == 'all':
        obj_names = all_objects
    rospy.loginfo('objects: {obj}'.format(obj=obj_names))

    for obj_name in obj_names:
        if obj_name not in all_objects:
            rospy.logwarn('Unknown object, skipping: {}'.format(obj_name))
        else:
            extract_color_histogram_from_objdata(obj_name)


if __name__ == '__main__':
    main()
