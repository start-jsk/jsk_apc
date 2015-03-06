#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from jsk_2014_picking_challenge.srv import ObjectMatch, ObjectMatchResponse
from jsk_recognition_msgs.msg import ColorHistogram

query_features = None
target_features = None

class ColorHistogramMatcher(object):
    def __init__(self):
        self.query_histogram = {}
        self.target_histograms = None

        rospy.Service('/semi/color_histogram_matcher', ObjectMatch,
            self.handle_colorhist_matcher)
        # input is color_histograms extracted by camera_image
        rospy.Subscriber('~input/histogram/red', ColorHistogram,
            self.cb_histogram_red)
        rospy.Subscriber('~input/histogram/green', ColorHistogram,
            self.cb_histogram_green)
        rospy.Subscriber('~input/histogram/blue', ColorHistogram,
            self.cb_histogram_blue)

    def handle_colorhist_matcher(self, req):
        """Handler of service request"""
        self.load_target_histograms(req.objects)
        return ObjectMatchResponse(probabilities=self.get_probabilities())

    def load_target_histograms(self):
        """Load extracted color histogram features of objects"""
        rospy.loginfo('Loading object color histogram features')
        # self.target_histograms = ...
        raise NotImplementedError

    def coefficient(query_hist, target_hist, method=0):
        """Compute coefficient of 2 histograms with several methods"""
        if method == 0:
            return (1. + cv2.compareHist(query_hist, target_hist,
                cv2.cv.CV_COMP_CORREL)) / 2.;

    def get_probabilities(self):
        """Get probabilities of color matching"""
        query_histogram = self.query_histogram
        target_histograms = self.target_histograms
        obj_coefs = []
        for obj_name, target_histgram in target_histograms.iteritems():
            # loop for RGB color &
            # compute max coefficient about each histograms
            coefs = []
            for q_hist, t_hist in zip(
                    query_histogram.values(), target_histogram.values()):
                coefs.append(coefficient(q_hist, t_hist))
            obj_coefs.append(max(coefs))
        obj_coefs = np.array(obj_coefs)
        # change coefficient array to probability array
        if obj_coefs.sum() == 0:
            return obj_coefs
        else:
            return obj_coefs / obj_coefs.sum()

    def cb_histogram_red(self, msg):
        """Get input red histogram"""
        self.query_histogram['red'] = msg.histogram

    def cb_histogram_green(self, msg):
        """Get input green histogram"""
        self.query_histogram['green'] = msg.histogram

    def cb_histogram_blue(self, msg):
        """Get input blue histogram"""
        self.query_histogram['blue'] = msg.histogram


def main():
    m = ColorHistogramMatcher()
    rospy.spin()


if __name__ == '__main__':
    main()

