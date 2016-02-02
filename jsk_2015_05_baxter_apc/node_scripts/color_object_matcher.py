#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
this script will test classification by the color_histogram
"""

import numpy as np

import cv_bridge
import jsk_apc2015_common
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport

import jsk_recognition_utils
import message_filters
import rospy
from sensor_msgs.msg import Image

from color_histogram_features import ColorHistogramFeatures


class ColorObjectMatcher(ConnectionBasedTransport):

    def __init__(self):
        super(ColorObjectMatcher, self).__init__()
        self._pub = self.advertise('~output', ClassificationResult,
                                   queue_size=1)
        self.estimator = ColorHistogramFeatures()
        self.estimator.load_data()

    def subscribe(self):
        self.sub_img = message_filters.Subscriber('~input', Image)
        self.sub_label = message_filters.Subscriber('~input/label', Image)
        queue_size = rospy.get_param('~queue_size', 100)
        if rospy.get_param('~approximate_sync', False):
            sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_img, self.sub_label], queue_size=queue_size,
                slop=0.1)
        else:
            sync = message_filters.TimeSynchronizer(
                [self.sub_img, self.sub_label], queue_size=queue_size)
        sync.registerCallback(self._predict)

    def unsubscribe(self):
        self.sub_img.sub.unregister()
        self.sub_label.sub.unregister()

    def _predict(self, img_msg, label_msg):
        # convert image
        bridge = cv_bridge.CvBridge()
        input_image = bridge.imgmsg_to_cv2(img_msg, 'rgb8')
        input_label = bridge.imgmsg_to_cv2(label_msg)
        # predict
        region_imgs = []
        for l in np.unique(input_label):
            if l == 0:  # bg_label
                continue
            mask = (input_label == l)
            region = jsk_recognition_utils.bounding_rect_of_mask(
                input_image, mask)
            region_imgs.append(region)
        y_proba = self.estimator.predict(region_imgs)
        target_names = np.array(jsk_apc2015_common.get_object_list())
        y_pred = np.argmax(y_proba, axis=-1)
        label_proba = [p[i] for p, i in zip(y_proba, y_pred)]
        # prepare message
        res = ClassificationResult()
        res.header = img_msg.header
        res.labels = y_pred
        res.label_names = target_names[y_pred]
        res.label_proba = label_proba
        res.probabilities = y_proba.reshape(-1)
        res.classifier = '<jsk_2015_05_baxter_apc.ColorHistogramFeatures>'
        res.target_names = target_names
        self._pub.publish(res)


if __name__ == "__main__":
    rospy.init_node('color_object_matcher')
    ColorObjectMatcher()
    rospy.spin()
