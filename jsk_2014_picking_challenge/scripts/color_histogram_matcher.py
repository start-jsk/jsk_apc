#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Usage
-----
$ roscore
$ rosrun jsk_2014_picking_challenge color_histogram_matcher.launch

"""
from __future__ import division
import rospy
import cv2
import cPickle
import gzip
import numpy as np
import os

from sensor_msgs.msg import Image
from jsk_2014_picking_challenge.srv import ObjectMatch, ObjectMatchResponse
from jsk_recognition_msgs.msg import ColorHistogram

query_features = None
target_features = None

class ColorHistogramMatcher(object):
    def __init__(self):
        self.query_histogram = {}
        self.target_histograms = {}

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
        rospy.loginfo("handl_colohhist_matcher")
        self.load_target_histograms(req.objects)
        return ObjectMatchResponse(probabilities=self.get_probabilities())

    def load_target_histograms(self, object_names):
        """Load extracted color histogram features of objects"""
        rospy.loginfo(object_names)

        # initialize
        self.target_histograms = {}

        dirname = os.path.dirname(os.path.abspath(__file__))
        for object_name in object_names:
            obj_dir = os.path.join(dirname, '../data/histogram_data/', object_name)
            rospy.loginfo(obj_dir)
            self.target_histograms[object_name] = {}
            for color in ['red', 'green', 'blue']:
                with gzip.open(obj_dir + '_' + color + '.pkl.gz', 'rb') as gf:
                    histogram = np.array(cPickle.load(gf), dtype='float32')
                    self.target_histograms[object_name][color] = histogram

    def coefficient(self, query_hist, target_hist, method=0):
        """Compute coefficient of 2 histograms with several methods"""
        if method == 0:
            return (1. + cv2.compareHist(query_hist, target_hist,
                cv2.cv.CV_COMP_CORREL)) / 2.;

    def get_probabilities(self):
        """Get probabilities of color matching"""
        rospy.loginfo("get probabilities")

        query_histogram = self.query_histogram
        targetes_histograms = self.target_histograms
        obj_coefs = []
        for obj_name, target_histograms in targetes_histograms.iteritems():
            # loop for RGB color &
            # compute max coefficient about each histograms
            coefs = []
            for q_hist, t_hists in zip(query_histogram.values(), target_histograms.values()):
                sum_of_coefs = 0
                for t_hist in t_hists:
                    # coefs.append(self.coefficient(q_hist, t_hist))
                    sum_of_coefs += self.coefficient(q_hist, t_hist)
                coefs.append(sum_of_coefs)
            obj_coefs.append(max(coefs))
        obj_coefs = np.array(obj_coefs)
        # change coefficient array to probability array
        if obj_coefs.sum() == 0:
            return obj_coefs
        else:
            return obj_coefs / obj_coefs.sum()

    def cb_histogram_red(self, msg):
        """Get input red histogram"""
        self.query_histogram['red'] = np.array(msg.histogram, dtype='float32')

    def cb_histogram_green(self, msg):
        """Get input green histogram"""
        self.query_histogram['green'] = np.array(msg.histogram, dtype='float32')

    def cb_histogram_blue(self, msg):
        """Get input blue histogram"""
        self.query_histogram['blue'] = np.array(msg.histogram, dtype='float32')


def main():
    rospy.init_node('color_histogram_matcher')
    m = ColorHistogramMatcher()
    rospy.spin()


if __name__ == '__main__':
    main()
