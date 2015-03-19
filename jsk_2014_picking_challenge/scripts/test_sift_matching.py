#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Matching for all objects with camera image

Usage
-----

    $ roslaunch jsk_2014_picking_challenge test_sift_matching.launch

"""
import os
import csv
import argparse

import numpy as np
import yaml

import rospy
from sensor_msgs.msg import CameraInfo
from jsk_2014_picking_challenge.srv import ObjectMatch, StringEmpty

from matcher_common import get_object_list


def load_csv(filename):
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            yield row


class TestSiftMatching(object):
    def __init__(self):
        self.client_of_img = rospy.ServiceProxy('/image_publish_server',
                                                StringEmpty)
        self.client_of_siftmatcher = rospy.ServiceProxy('/semi/sift_matcher',
                                                        ObjectMatch)

    def save_result(self, target_obj, probabilities):
        """Save test result to csv"""
        object_list = get_object_list()
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../data/test_sift_matching_result.csv')
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
            '../data/test_sift_matching_result.csv')
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
            # request to sift matcher
            rospy.loginfo('target object: {target}'.format(target=target_obj))
            self.wait_for_service(self.client_of_siftmatcher)
            res = self.client_of_siftmatcher(objects=object_list)
            rospy.loginfo('results: {res}'.format(res=res.probabilities))
            best_match_obj = object_list[np.argmax(res.probabilities)]
            rospy.loginfo('best match: {best}'.format(best=best_match_obj))
            self.save_result(target_obj=target_obj,
                             probabilities=list(res.probabilities))


if __name__ == '__main__':
    rospy.init_node('test_sift_matching')
    test = TestSiftMatching()
    test.run()

