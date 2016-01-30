#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
This script is to match objects with sift features which is extracted by
jsk_2015_05_baxter_apc/node_scripts/extract_sift_from_objdata.py.

Usage
-----

    $ roslaunch jsk_2015_05_baxter_apc sift_matcher_usbcamera.launch
    $ rosrun jsk_2015_05_baxter_apc sift_matcher.py
    $ rosservice call /semi/sift_matcher \
        "{objects: ['oreo_mega_stuf', 'safety_works_safety_glasses']}"

"""
from __future__ import print_function, division
from future_builtins import zip
import os

import cv2
import numpy as np

import rospy
import cv_bridge
import dynamic_reconfigure.server
from posedetection_msgs.msg import ImageFeature0D
from posedetection_msgs.srv import Feature0DDetect
from jsk_2015_05_baxter_apc.cfg import SIFTMatcherConfig

from common import ObjectMatcher, load_siftdata
import jsk_apc2015_common


class SiftMatcher(object):
    def __init__(self):
        self.knn_threshold = 0.75
        rospy.Subscriber('/ImageFeature0D', ImageFeature0D,
                         self._cb_imgfeature)
        dynamic_reconfigure.server.Server(SIFTMatcherConfig,
                                          self._cb_dynamic_reconfigure)
        rospy.loginfo('wait for message [{p}]'.format(p=os.getpid()))
        rospy.wait_for_message('/ImageFeature0D', ImageFeature0D)
        rospy.loginfo('found the message [{p}]'.format(p=os.getpid()))

    def _cb_imgfeature(self, msg):
        """Callback function of Subscribers to listen ImageFeature0D"""
        self.query_features = msg.features

    def _cb_dynamic_reconfigure(self, config, level):
        """Callback function of dynamic reconfigure server"""
        self.knn_threshold = config['knn_threshold']
        return config

    def find_match(self, query_des, train_des):
        """Find match points of query and train images"""
        # parepare to match keypoints
        query_des = np.array(query_des).reshape((-1, 128))
        query_des = (query_des * 255).astype('uint8')
        train_des = np.array(train_des).reshape((-1, 128))
        train_des = (train_des * 255).astype('uint8')
        # find good match points
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_des, train_des, k=2)
        good_matches = [n1 for n1, n2 in matches
                        if n1.distance < self.knn_threshold*n2.distance]
        return good_matches


class SiftObjectMatcher(SiftMatcher, ObjectMatcher):
    def __init__(self):
        SiftMatcher.__init__(self)
        ObjectMatcher.__init__(self, '/semi/sift_matcher')
        self.object_list = jsk_apc2015_common.get_object_list()
        self.siftdata_cache = {}

    def match(self, obj_names):
        """Get object match probabilities"""
        query_features = self.query_features
        n_matches = []
        siftdata_set = self._handle_siftdata_cache(obj_names)
        for obj_name, siftdata in zip(obj_names, siftdata_set):
            if obj_name not in self.object_list:
                n_matches.append(0)
                continue
            if siftdata is None:  # does not exists data file
                n_matches.append(0)
                continue
            # find best match in train features
            rospy.loginfo('Searching matches: {}'.format(obj_name))
            train_matches = []
            for train_des in siftdata['descriptors']:
                matches = self.find_match(query_features.descriptors,
                                          train_des)
                train_matches.append(len(matches))
            n_match = max(train_matches)  # best match about the object
            rospy.loginfo("{}'s best match: {}".format(obj_name, n_match))
            n_matches.append(n_match)
        # make match point counts to probabilities
        n_matches = np.array(n_matches)
        rospy.loginfo('Number of matches: {}'.format(n_matches))
        try:
            return n_matches / n_matches.sum()
        except ZeroDivisionError:
            return n_matches

    def _handle_siftdata_cache(self, obj_names):
        """Sift data cache handler
        if same obj_names set: don't update, else: update
        """
        for obj_name in obj_names:
            if obj_name in self.siftdata_cache:
                siftdata = self.siftdata_cache[obj_name]
            else:
                if len(self.siftdata_cache) > 3:
                    # free cache data to avoid becoming too big
                    del self.siftdata_cache[np.random.choice(
                        self.siftdata_cache.keys())]
                siftdata = load_siftdata(obj_name)
                # set cache
                self.siftdata_cache[obj_name] = siftdata
            if siftdata is None:
                continue
            yield siftdata


def imgsift_client(img):
    """Request to imagesift with Image as service client"""
    client = rospy.ServiceProxy('/Feature0DDetect', Feature0DDetect)
    client.wait_for_service()
    bridge = cv_bridge.CvBridge()
    img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    img_msg.header.stamp = rospy.Time.now()
    resp = client(img_msg)
    return resp.features


def main():
    rospy.init_node('sift_matcher')
    SiftObjectMatcher()
    rospy.spin()


if __name__ == '__main__':
    main()

