#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
This script is to extract color histogram from object data
distributed by Robot Learning Lab, UC Berkeley.
See extract_sift_from_objdata.py for more details.

Usage
-----
1. Download dataset(Raw High Resolution RGB) to data dir, and extract it.
2. Execute following::

    $ roslaunch jsk_2015_05_baxter_apc extract_color_histogram.launch
    $ rosrun jsk_2015_05_baxter_apc extract_color_histogram.py \
            _object:=all _color_space:=rgb

"""
import os
import gzip
import cPickle as pickle

import cv2
import numpy as np
import progressbar

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import ColorHistogram


class ExtractColorHistogram(object):
    def __init__(self, object_nm, color, raw_paths, mask_paths=None):
        if mask_paths is None:
            self.imgpaths = raw_paths
        else:
            if len(raw_paths) != len(mask_paths):
                ValueError('Number of raw and mask images should be same')
            self.imgpaths = zip(raw_paths, mask_paths)
        self.object_nm = object_nm
        self.color = color
        self.color_hist = None
        self.stamp = None
        self.image_pub = rospy.Publisher('~train_image', Image,
                                         queue_size=1)

    def color_hist_cb(self, msg):
        self.color_hist = msg.histogram
        self.stamp = msg.header.stamp

    def extract(self):
        """Extract color histogram data from object images"""
        imgpaths = self.imgpaths
        object_nm = self.object_nm
        color_histograms = []
        progress = progressbar.ProgressBar(
            widgets=['{o}: '.format(o=object_nm), progressbar.Bar(),
            progressbar.Percentage(), ' ', progressbar.ETA()])
        for imgpath in progress(list(imgpaths)):
            if type(imgpath) is tuple:
                raw_path, mask_path = imgpath
                raw_img = cv2.imread(raw_path)
                mask_img = cv2.imread(mask_path)
                train_img = cv2.add(mask_img, raw_img)
            else:
                raw_path = imgpath
                train_img = cv2.imread(raw_path)

            color_hist_sub = rospy.Subscriber('single_channel_histogram_'
                + self.color + '/output', ColorHistogram, self.color_hist_cb)
            bridge = cv_bridge.CvBridge()
            train_imgmsg = bridge.cv2_to_imgmsg(train_img, encoding='bgr8')
            train_imgmsg.header.stamp = rospy.Time.now()
            # wait for histogram extracted from new image
            while not self.stamp or self.stamp < train_imgmsg.header.stamp:
                self.image_pub.publish(train_imgmsg)
                rospy.sleep(0.3)
            color_histograms.append(self.color_hist)
        return np.array(color_histograms)

    def save(self, hist_data):
        """Save histogram data to data/histogram_data/{object_nm}.pkl.gz"""
        object_nm = self.object_nm
        color = self.color
        data_dir = os.path.join(os.path.dirname(__file__),
                                '../data/histogram_data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        filename = os.path.join(data_dir, object_nm+'_'+color+'.pkl.gz')
        with gzip.open(filename, 'wb') as f:
            pickle.dump(hist_data, f)

    def extract_and_save(self):
        hist_data = self.extract()
        self.save(hist_data=hist_data)


def main():
    from common import get_train_imgpaths
    import jsk_apc2015_common

    rospy.init_node('extract_color_histogram')

    all_objects = jsk_apc2015_common.get_object_list()

    color_space_param = rospy.get_param('~color_space', 'lab')
    if color_space_param == 'rgb':
        colors = ['red', 'green', 'blue']
    elif color_space_param == 'lab':
        colors = ['l']
    else:
        raise ValueError('Unknown color space')
    rospy.loginfo('color space: {c}'.format(c=color_space_param))

    object_param = rospy.get_param('~object', all_objects)
    object_nms = object_param.split(',')
    if len(object_nms) == 1 and object_nms[0] == 'all':
        object_nms = all_objects
    rospy.loginfo('objects: {obj}'.format(obj=object_nms))

    for object_nm in object_nms:
        if object_nm not in all_objects:
            rospy.logwarn('Unknown object, skipping: {}'.format(object_nm))
        else:
            imgpaths = get_train_imgpaths(object_nm)
            raw_paths, mask_paths = zip(*imgpaths)
            for color in colors:
                e = ExtractColorHistogram(object_nm=object_nm, color=color,
                        raw_paths=raw_paths, mask_paths=mask_paths)
                e.extract_and_save()


if __name__ == '__main__':
    main()

