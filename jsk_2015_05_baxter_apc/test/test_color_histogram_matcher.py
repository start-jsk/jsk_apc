#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""This script is to test color histogram & its matcher

Usage
-----

    $ # to extract color histogram
    $ roslaunch jsk_2015_05_baxter_apc extract_color_histogram.launch
        input_image:=/test_color_histogram/train_image
    $ rosrun jsk_2015_05_baxter_apc test_color_histogram.py --extract

    $ # to test color histogram matcher
    $ roslaunch jsk_2015_05_baxter_apc \
        test_color_histogram_matching.launch
    $ rosrun jsk_2015_05_baxter_apc test_color_histogram.py --test

"""
from __future__ import division
import os
import sys
import argparse
import unittest

import numpy as np

import rospy
from jsk_2015_05_baxter_apc.srv import ObjectMatch, StringEmpty

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from extract_color_histogram import ExtractColorHistogram
from common import listdir_for_img
from test_object_matching import TestObjectMatching


def get_nations():
    data_dir = os.path.join(os.path.dirname(__file__),
                            '../data/national_flags')
    return os.listdir(data_dir)


def get_data_dirs():
    data_dir = os.path.join(os.path.dirname(__file__),
                            '../data/national_flags')
    for nation in get_nations():
        yield os.path.join(data_dir, nation)


def prepare_train_data(colors):
    for data_dir in get_data_dirs():
        nation_nm = os.path.basename(data_dir)
        raw_paths = map(lambda x: os.path.join(data_dir, x),
                        listdir_for_img(data_dir))
        for color in colors:
            extractor = ExtractColorHistogram(object_nm=nation_nm,
                color=color, raw_paths=raw_paths)
            extractor.extract_and_save()


class TestColorHistogramMatcher(unittest.TestCase):
    def test_matching(self):
        client_of_matcher = rospy.ServiceProxy(
            '/semi/color_histogram_matcher', ObjectMatch)
        client_of_img = rospy.ServiceProxy('/image_publish_server',
                                           StringEmpty)
        nations = np.array(get_nations())
        for i, target_obj in enumerate(nations):
            # request to publish image
            rospy.loginfo('target: {}'.format(target_obj))
            imgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../data/national_flags/{0}/{0}.png'.format(target_obj))
            client_of_img(string=imgpath)
            rospy.sleep(3)
            # request to object matcher
            probs = client_of_matcher(objects=nations).probabilities
            probs = np.array(probs)
            # about max probability
            max_index = probs.argmax()
            rospy.loginfo('correct?: {}'.format(max_index == i))
            self.assertEqual(max_index, i)
            # about similar objects
            similars = nations[probs.argsort()][::-1][:3]
            rospy.loginfo('similar: {}'.format(similars))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('color_space', help='color space')
    parser.add_argument('-e', '--extract', action='store_true',
                        help='flag to extract color histogram')
    parser.add_argument('-t', '--test', action='store_true',
                        help='flag to test color histogram matcher')
    args = parser.parse_args(rospy.myargv()[1:])
    flags = dict(args._get_kwargs()).values()
    if not any(flags) or all(flags):
        print('either -e or -t should be set (both is not allowed)')
        parser.print_help()
        parser.exit()
    return args


def main():
    args = parse_args()
    if args.color_space == 'rgb':
        colors = ['red', 'green', 'blue']
    elif args.color_space == 'lab':
        colors = ['l']
    else:
        raise ValueError('Unknown color space')
    if args.extract:
        prepare_train_data(colors)
    elif args.test:
        suite = unittest.TestLoader().loadTestsFromTestCase(
            TestColorHistogramMatcher)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        rospy.logerr('Unknown args')


if __name__ == '__main__':
    rospy.init_node('test_color_histogram_matcher')
    main()

