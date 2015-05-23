#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import rospy
from jsk_2014_picking_challenge.msg import ObjectRecognition

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
