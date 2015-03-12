#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
        # self.query_histogram = {}
        self.query_histogram = {'red':np.array([63750, 4289, 3947, 22, 1,2,2,3,1,2], dtype='float32'),
                                'green':np.array([63750, 4289, 3947, 22, 1,2,2,3,1,2], dtype='float32'),
                                'blue':np.array([63750, 4289, 3947, 22, 1,2,2,3,1,2], dtype='float32')}
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
        for object_name in object_names:
            obj_dir = os.path.join('../data/histogram_data/', object_name)
            rospy.loginfo(obj_dir)
            with gzip.open(obj_dir + '.pkl.gz', 'rb') as gf:
                x = np.array(cPickle.load(gf), dtype='float32')
                self.target_histograms[object_name] = {'red':x,
                                                        'blue':x,
                                                        'green':x}

    def coefficient(self, query_hist, target_hist, method=0):
        """Compute coefficient of 2 histograms with several methods"""
        if method == 0:
            return (1. + cv2.compareHist(query_hist, target_hist,
                cv2.cv.CV_COMP_CORREL)) / 2.;

    def get_probabilities(self):
        """Get probabilities of color matching"""
        rospy.loginfo("probs")
        query_histogram = self.query_histogram
        targetes_histograms = self.target_histograms
        obj_coefs = []
        for obj_name, target_histograms in targetes_histograms.iteritems():
            # loop for RGB color &
            # compute max coefficient about each histograms
            coefs = []
            for q_hist, t_hists in zip(query_histogram.values(), target_histograms.values()):
                for t_hist in t_hists:
                    coefs.append(self.coefficient(q_hist, t_hist))
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
    rospy.init_node('color_histogram_matcher')
    m = ColorHistogramMatcher()
    rospy.spin()


if __name__ == '__main__':
    main()
