#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import sys

from jsk_2014_picking_challenge.srv import ObjectMatch, ObjectMatchResponse

def color_histogram_matcher_client(object_names):
    rospy.wait_for_service('/semi/color_histogram_matcher')
    try:
        probs = rospy.ServiceProxy('/semi/color_histogram_matcher', ObjectMatch)
        resp = probs(object_names)
        print resp
        return resp.probabilities
    except rospy.ServiceException, e:
        rospy.loginfo("Service call faield. {}".format(e))

def main():
    print(color_histogram_matcher_client(['champion_copper_plus_spark_plug', 'cheezit_big_original', 'crayola_64_ct']))

if __name__ == "__main__":
    main()
