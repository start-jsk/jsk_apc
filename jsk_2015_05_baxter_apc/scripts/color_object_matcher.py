#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
this script will test classification by the color_histogram
"""

import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import ColorHistogram
from jsk_2015_05_baxter_apc.msg import ObjectRecognition
import jsk_2015_apc_common
from common import ObjectMatcher
from color_histogram_features import ColorHistogramFeatures
import cv_bridge

import numpy as np


class ColorObjectMatcher(ObjectMatcher):
    def __init__(self):
        super(ColorObjectMatcher, self).__init__('/color_object_matcher')
        rospy.Subscriber('~input', Image, self._cb_image)
        self._pub_recog = rospy.Publisher('~output', ObjectRecognition,
                                          queue_size=1)
        self._pub_debug = rospy.Publisher(
            '~debug', ColorHistogram, queue_size=1)

        self.query_image = Image()
        self.estimator = ColorHistogramFeatures()
        self.estimator.load_data()
    def _cb_image(self, msg):
        """ Callback function fo Subscribers to listen sensor_msgs/Image """
        self.query_image = msg
    def predict_now(self):
        query_image = self.query_image

        object_list = jsk_2015_apc_common.data.object_list()
        probs = self.match(object_list)
        matched_idx = np.argmax(probs)
        # prepare message
        res = ObjectRecognition()
        res.header.stamp = query_image.header.stamp
        res.matched = object_list[matched_idx]
        res.probability = probs[matched_idx]
        res.candidates = object_list
        res.probabilities = probs
        return res

    def match(self, obj_names):
        stamp = rospy.Time.now()
        while (self.query_image.header.stamp < stamp) or (self.query_image.height == 0):
            rospy.sleep(0.3)
        query_image = self.query_image
        # convert image
        bridge = cv_bridge.CvBridge()
        input_image = bridge.imgmsg_to_cv2(query_image, 'rgb8')

        hist = self.estimator.color_hist(input_image)
        self._pub_debug.publish(
            ColorHistogram(header=query_image.header, histogram=hist))

        object_list = jsk_2015_apc_common.data.object_list()
        obj_indices = [object_list.index(o) for o in obj_names]
        obj_probs = self.estimator.predict(input_image)[0][obj_indices]
        return obj_probs / obj_probs.sum()
    def spin_once(self):
        res = self.predict_now()
        if res is None:
            return
        self._pub_recog.publish(res)
    def spin(self):
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()

def main():
    rospy.init_node('color_object_matcher')
    matcher = ColorObjectMatcher()
    matcher.spin()

if __name__ == "__main__":
    main()
