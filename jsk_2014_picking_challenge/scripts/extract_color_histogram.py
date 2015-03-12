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
    $ rosrun jsk_2014_picking_challenge color_histogram.launch
    $ rosrun extract_color_histogram.py _object:=oreo_mega_stuf


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
import cv_bridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from posedetection_msgs.srv import Feature0DDetect
from posedetection_msgs.msg import ImageFeature0D

from sift_matcher_oneimg import SiftMatcherOneImg

from extract_sift_from_objdata import get_train_imgpaths

from jsk_recognition_msgs.msg import ColorHistogram


class ExtractColorHistogram(object):
    def __init__(self, obj_name):
        self.obj_name = obj_name
        self.color_hist = None

    def save_color_hist_data(self, histdata, obj_name):
        """Save color histogram data to data/histdata/{obj_name}.pkl.gz"""
        dirname = os.path.dirname(os.path.abspath(__file__))
        histdata_dir = os.path.join(dirname, '../data/histdata')
        if not os.path.exists(histdata_dir):
            os.mkdir(histdata_dir)
        filename = os.path.join(histdata_dir, self.obj_name+'_red.pkl.gz')
        with gzip.open(filename, 'wb') as f:
            pickle.dump(histdata, f)

    def color_hist_cb(self, msg):
        self.color_hist = msg.histogram

    def extract_color_histogram_from_objdata(self):
        """Extract color histogram data from object images"""
        color_histograms = []
        imgpaths = get_train_imgpaths(self.obj_name)
        if imgpaths is None:
            return   # skip if img does not exists
        progress = progressbar.ProgressBar(widgets=['{o}: '.format(o=self.obj_name),
                                        progressbar.Bar(), progressbar.Percentage(), ' ', progressbar.ETA()])
        image_pub = rospy.Publisher('image_publisher/output', Image, queue_size=1)
        for raw_path, mask_path in progress(imgpaths):
            raw_img = cv2.imread(raw_path)
            mask_img = cv2.imread(mask_path)
            train_img = cv2.add(mask_img, raw_img)

            color_hist_sub = rospy.Subscriber('single_channel_histogram/output', ColorHistogram, self.color_hist_cb)
            bridge = cv_bridge.CvBridge()
            train_img_msg = bridge.cv2_to_imgmsg(train_img, encoding="bgr8")
            train_img_msg.header.stamp = rospy.Time.now()

            self.color_hist = None
            while self.color_hist == None:
                image_pub.publish(train_img_msg)
                rospy.sleep(1)
            color_histograms.append(self.color_hist)
        color_histograms = np.array(color_histograms)
        self.save_histogram_data(color_histograms, self.obj_name)

    def save_histogram_data(self, histogram_data, obj_name):
        """Save histogram data to data/histogram_data/{obj_name}.pkl.gz"""
        dirname = os.path.dirname(os.path.abspath(__file__))
        histogram_data_dir = os.path.join(dirname, '../data/histogram_data')
        if not os.path.exists(histogram_data_dir):
            os.mkdir(histogram_data_dir)
        filename = os.path.join(histogram_data_dir, obj_name+'.pkl.gz')
        with gzip.open(filename, 'wb') as f:
            pickle.dump(histogram_data, f)

def main():
    rospy.init_node('extract_color_histogram_from_objdata')
    pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=1)
    pub.publish()  # to enable imagehist service

    dirname = os.path.dirname(os.path.abspath(__file__))
    ymlfile = os.path.join(dirname, '../data/object_list.yml')
    all_objects = yaml.load(open(ymlfile))

    obj_names = rospy.get_param('~object',
                                ["champion_copper_plus_spark_plug",
                                "cheezit_big_original",
                                "crayola_64_ct",
                                "elmers_washable_no_run_school_glue",
                                "expo_dry_erase_board_eraser",
                                "feline_greenies_dental_treats",
                                "first_years_take_and_toss_straw_cups",
                                "genuine_joe_plastic_stir_sticks",
                                "highland_6539_self_stick_notes",
                                "kong_air_dog_squeakair_tennis_ball",
                                "kong_duck_dog_toy",
                                "kong_sitting_frog_dog_toy",
                                "kyjen_squeakin_eggs_plush_puppies",
                                "mark_twain_huckleberry_finn",
                                "mead_index_cards",
                                "mommys_helper_outlet_plugs",
                                "munchkin_white_hot_duck_bath_toy",
                                "oreo_mega_stuf",
                                "paper_mate_12_count_mirado_black_warrior",
                                "rolodex_jumbo_pencil_cup",
                                "safety_works_safety_glasses",
                                "sharpie_accent_tank_style_highlighters",
                                "stanley_66_052"]
    )
    # obj_names = obj_names.split(',')
    if len(obj_names) == 1 and obj_names[0] == 'all':
        obj_names = all_objects
    rospy.loginfo('objects: {obj}'.format(obj=obj_names))

    for obj_name in obj_names:
        if obj_name not in all_objects:
            rospy.logwarn('Unknown object, skipping: {}'.format(obj_name))
        else:
            e = ExtractColorHistogram(obj_name)
            e.extract_color_histogram_from_objdata()

if __name__ == '__main__':
    main()
