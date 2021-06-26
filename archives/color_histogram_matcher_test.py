#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import sys

from jsk_2015_05_baxter_apc.srv import ObjectMatch, ObjectMatchResponse

def color_histogram_matcher_client(object_names):
    rospy.wait_for_service('/semi/color_histogram_matcher')
    try:
        probs = rospy.ServiceProxy('/semi/color_histogram_matcher', ObjectMatch)
        resp = probs(object_names)
        print(object_names)
        return resp
    except rospy.ServiceException as e:
        rospy.loginfo("Service call faield. {}".format(e))

def main():
    print(color_histogram_matcher_client(['champion_copper_plus_spark_plug', 'crayola_64_ct']))
    print(color_histogram_matcher_client(['elmers_washable_no_run_school_glue', 'cheezit_big_original', 'expo_dry_erase_board_eraser']))

if __name__ == "__main__":
    main()
