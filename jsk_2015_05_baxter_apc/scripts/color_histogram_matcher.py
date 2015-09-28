#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Color histogram matcher using extracted histogram
by extract_color_histogram.py.

Usage
-----

    $ roslaunch kinect2_bridge kinect2_bridge.launch
    $ roslaunch jsk_2015_05_baxter_apc rgb_color_histogram_matcher.launch
    $ rosservice call /semi/color_histogram_matcher \
        "{objects: oreo_mega_stuf, crayola_64_ct}"

"""
from __future__ import division
import rospy
import cv2
import cPickle
import gzip
import numpy as np
import os

from sensor_msgs.msg import Image
from jsk_2015_05_baxter_apc.srv import ObjectMatch, ObjectMatchResponse
from jsk_recognition_msgs.msg import ColorHistogram


class ColorHistogramMatcher(object):
    def __init__(self, color_space='lab'):
        self.query_histogram = {}
        self.target_histograms = {}
        rospy.Service('/semi/color_histogram_matcher', ObjectMatch,
            self.handle_colorhist_matcher)
        # input is color_histograms extracted by camera_image
        if color_space == 'rgb':
            self.colors = ['red', 'green', 'blue']
            rospy.Subscriber('~input/histogram/red', ColorHistogram,
                             lambda msg: self.cb_histogram(msg, 'red'))
            rospy.Subscriber('~input/histogram/green', ColorHistogram,
                             lambda msg: self.cb_histogram(msg, 'green'))
            rospy.Subscriber('~input/histogram/blue', ColorHistogram,
                             lambda msg: self.cb_histogram(msg, 'blue'))
        elif color_space == 'lab':
            self.colors = ['l']
            rospy.Subscriber('~input/histogram/l', ColorHistogram,
                             lambda msg: self.cb_histogram(msg, 'l'))

    def handle_colorhist_matcher(self, req):
        """Handler of service request"""
        rospy.loginfo("handl_colohhist_matcher")
        self.load_target_histograms(req.objects)
        probs = self.get_probabilities(req.objects)
        return ObjectMatchResponse(probabilities=probs)

    def load_target_histograms(self, object_names):
        """Load extracted color histogram features of objects"""
        rospy.loginfo('objects: {}'.format(object_names))

        # initialize
        self.target_histograms = {}

        dirname = os.path.dirname(os.path.abspath(__file__))
        for object_name in object_names:
            obj_dir = os.path.join(dirname, '../data/histogram_data/', object_name)
            self.target_histograms[object_name] = {}
            for color in self.colors:
                with gzip.open(obj_dir + '_' + color + '.pkl.gz', 'rb') as gf:
                    histogram = np.array(cPickle.load(gf), dtype='float32')
                    self.target_histograms[object_name][color] = histogram

    def coefficient(self, query_hist, target_hist, method=0):
        """Compute coefficient of 2 histograms with several methods"""
        if method == 0:
            return (1. + cv2.compareHist(query_hist, target_hist,
                cv2.cv.CV_COMP_CORREL)) / 2.;

    def get_probabilities(self, object_names):
        """Get probabilities of color matching
        (the order of object_names and probabilities is same)
        """
        rospy.loginfo("get probabilities")

        query_histogram = self.query_histogram
        targets_histograms = self.target_histograms
        obj_coefs = []
        for obj_name in object_names:
            target_histograms = targets_histograms[obj_name]
            # loop for RGB color &
            # compute max coefficient about each histograms
            coefs = []
            for color in query_histogram.keys():
                q_hist = query_histogram[color]
                t_hists = target_histograms[color]
                coefs.append(max(self.coefficient(q_hist, t_hist)
                                 for t_hist in t_hists))
            obj_coefs.append(sum(coefs))
        obj_coefs = np.array(obj_coefs)
        # change coefficient array to probability array
        try:
            return obj_coefs / obj_coefs.sum()
        except ZeroDivisionError:
            return obj_coefs

    def cb_histogram(self, msg, color):
        """Get input histogram"""
        self.query_histogram[color] = np.array(msg.histogram, dtype='float32')


def main():
    rospy.init_node('color_histogram_matcher')
    color_space = rospy.get_param('~color_space', 'lab')
    m = ColorHistogramMatcher(color_space=color_space)
    rospy.spin()


if __name__ == '__main__':
    main()
