#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Matching for all objects with camera image

Usage
-----

    $ roslaunch jsk_2014_picking_challenge test_object_matching.launch \
        sift:=true
    or
    $ rolaunch jsk_2014_picking_challenge test_object_matching.launch \
        color_histogram:=true

"""
import os
import csv

import numpy as np

import rospy
from jsk_2014_picking_challenge.srv import ObjectMatch, StringEmpty

from matcher_common import get_object_list


def load_csv(filename):
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            yield row


class TestObjectMatching(object):
    def __init__(self, matcher):
        self.matcher = matcher
        self.client_of_img = rospy.ServiceProxy('/image_publish_server',
                                                StringEmpty)
        if matcher == 'sift':
            self.client_of_matcher = rospy.ServiceProxy(
                '/semi/sift_matcher', ObjectMatch)
        elif matcher == 'color_histogram':
            self.client_of_matcher = rospy.ServiceProxy(
                '/semi/color_histogram_matcher', ObjectMatch)
        else:
            raise ValueError('Unknown matcher: {0}'.format(matcher))

    def save_result(self, target_obj, probabilities):
        """Save test result to csv"""
        object_list = get_object_list()
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../data/test_{m}_matching_result.csv'.format(m=self.matcher))
        if not os.path.exists(filename):
            # initialize csv file
            with open(filename, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['target_obj', 'is_correct'] + object_list)
        # save result
        best_match_obj = object_list[np.argmax(probabilities)]
        row = [target_obj, best_match_obj==target_obj] + probabilities
        with open(filename, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)

    def get_already_tested(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../data/test_{m}_matching_result.csv'.format(m=self.matcher))
        test_data = load_csv(filename)
        return np.array(list(test_data))[:, 0]

    def wait_for_service(self, service_client):
        rospy.loginfo('wait for service')
        service_client.wait_for_service()
        rospy.loginfo('found the service')

    def run(self):
        object_list = get_object_list()
        already_tested = self.get_already_tested()
        for target_obj in object_list:
            if target_obj in already_tested:
                continue
            # request to publish image
            imgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../data/raw_img/{t}.jpg'.format(t=target_obj))
            if not os.path.exists(imgpath):
                continue
            self.wait_for_service(self.client_of_img)
            self.client_of_img(string=imgpath)
            # request to object matcher
            rospy.loginfo('target object: {target}'.format(target=target_obj))
            self.wait_for_service(self.client_of_matcher)
            res = self.client_of_matcher(objects=object_list)
            rospy.loginfo('results: {res}'.format(res=res.probabilities))
            best_match_obj = object_list[np.argmax(res.probabilities)]
            rospy.loginfo('best match: {best}'.format(best=best_match_obj))
            self.save_result(target_obj=target_obj,
                             probabilities=list(res.probabilities))


if __name__ == '__main__':
    rospy.init_node('test_object_matching')
    matcher = rospy.get_param('~matcher', 'sift')
    test = TestObjectMatching(matcher=matcher)
    test.run()

