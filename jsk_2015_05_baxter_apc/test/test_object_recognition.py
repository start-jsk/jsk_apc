#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import rospy
from jsk_2015_05_baxter_apc.msg import ObjectRecognition

left_result = None
right_result = None

def _cb_left(msg):
    global left_result
    left_result = dict(zip(msg.candidates, msg.probabilities))

def _cb_right(msg):
    global right_result
    right_result = dict(zip(msg.candidates, msg.probabilities))

rospy.init_node('test_object_recognition')
rospy.Subscriber('/left_process/bof_object_matcher/output', ObjectRecognition, _cb_left)
rospy.Subscriber('/right_process/bof_object_matcher/output', ObjectRecognition, _cb_right)

left_or_right = sys.argv[1]
import pprint
while not rospy.is_shutdown():
    result = left_result if left_or_right == 'left' else right_result
    if result is None:
        continue
    print('---------------------------------')
    pprint.pprint(sorted(right_result.items(),
                         key=lambda x:x[1],
                         reverse=True))

    rospy.sleep(1)
