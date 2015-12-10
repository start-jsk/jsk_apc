#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import rospy
import numpy as np
from jsk_rviz_plugins.msg import OverlayText
from jsk_2015_05_baxter_apc.msg import ObjectRecognition


def get_bin_contents(json_file):
    with open(json_file, 'r') as f:
        bin_contents = json.load(f)['bin_contents']
    for bin_, objects in bin_contents.items():
        bin_ = bin_.split('_')[1].lower()  # bin_A -> a
        yield (bin_, objects)


def get_work_order(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)['work_order']
    for order in data:
        bin_ = order['bin'].split('_')[1].lower()  # bin_A -> a
        target_object = order['item']
        yield (bin_, target_object)

recognition_msg = None
def cb_recognition(msg):
    global recognition_msg
    recognition_msg = msg
sub_recognition = rospy.Subscriber('/right_hand/object_verification/output', ObjectRecognition, cb_recognition)


rospy.init_node('visualize_on_rviz')

json_file = rospy.get_param('~json')

bin_contents = dict(list(get_bin_contents(json_file)))
work_order = dict(list(get_work_order(json_file)))

pub_r_state = rospy.Publisher('~right_state', OverlayText, queue_size=1)
pub_r_target = rospy.Publisher('~right_target', OverlayText, queue_size=1)
pub_r_contents = rospy.Publisher('~right_contents', OverlayText, queue_size=1)
pub_r_order = rospy.Publisher('~right_order', OverlayText, queue_size=1)
pub_r_result = rospy.Publisher('~right_result', OverlayText, queue_size=1)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    # robot state
    r_msg = OverlayText()
    r_msg.text = 'right_arm: state: ' + rospy.get_param('right_hand/state', '-')
    pub_r_state.publish(r_msg)
    target_bin = rospy.get_param('right_hand/target_bin', '-')
    r_msg.text = 'right_arm: target_bin: ' + target_bin
    pub_r_target.publish(r_msg)
    if not target_bin:
        rate.sleep()
        continue
    # bin contents
    contents = bin_contents[target_bin]
    r_msg.text = 'right_arm: objects in bin: {}: {}'.format(target_bin, ', '.join(contents))
    pub_r_contents.publish(r_msg)
    # work order
    order = work_order[target_bin]
    r_msg.text = 'right_arm target object: {}'.format(order)
    pub_r_order.publish(r_msg)
    # recognition result
    if recognition_msg is None:
        rate.sleep()
        continue
    result = recognition_msg.candidates[np.argmax(recognition_msg.probabilities)]
    r_msg.text = 'right_arm: recognized as: ' + result
    pub_r_result.publish(r_msg)
    rate.sleep()
