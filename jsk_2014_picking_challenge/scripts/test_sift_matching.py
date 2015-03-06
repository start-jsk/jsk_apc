#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Matching for all objects with camera image

Usage
-----

    $ roslaunch kinect2_bridge kinect2_bridge.launch
    $ roslaunch jsk_2014_picking_challenge sift_matcher.launch
    $ rosrun jsk_2014_picking_challenge sift_matcher.py
    $ rosrun jsk_2014_picking_challenge test_sift_matching.py

"""
import os

import yaml

import rospy
from jsk_2014_picking_challenge.srv import (
    ObjectMatch,
    ObjectMatchRequest,
    )


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    ymlfile = os.path.join(dirname, '../data/object_list.yml')
    object_list = yaml.load(open(ymlfile))

    client_siftmatcher = rospy.ServiceProxy('/semi/sift_matcher', ObjectMatch)
    client_siftmatcher.wait_for_service()

    rospy.loginfo('Objects: {obj}'.format(obj=object_list))
    resp = client_siftmatcher(ObjectMatchRequest(objects=object_list))
    rospy.loginfo('Results: {res}'.format(res=resp.probabilities))
    rospy.loginfo('Best match: {best}'.format(
        best=object_list[np.argmax(resp.probabilities)])


if __name__ == '__main__':
   main()

