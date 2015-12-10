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

recognition_msg = {'left': None, 'right': None}
def cb_recognition_l(msg):
    global recognition_msg
    recognition_msg['left'] = msg
def cb_recognition_r(msg):
    global recognition_msg
    recognition_msg['right'] = msg
sub_recognition_l = rospy.Subscriber('/left_hand/object_verification/output', ObjectRecognition, cb_recognition_l)
sub_recognition_r = rospy.Subscriber('/right_hand/object_verification/output', ObjectRecognition, cb_recognition_r)


rospy.init_node('visualize_on_rviz')

json_file = rospy.get_param('~json')

bin_contents = dict(list(get_bin_contents(json_file)))
work_order = dict(list(get_work_order(json_file)))

pub_l = rospy.Publisher('~left', OverlayText, queue_size=1)
pub_r = rospy.Publisher('~right', OverlayText, queue_size=1)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    msg = {}
    for arm in ['left', 'right']:
        msg[arm] = OverlayText()
        # robot state
        state_text = rospy.get_param(arm+'_hand/state', '')
        target_bin = rospy.get_param(arm+'_hand/target_bin', '')
        target_text = target_bin
        # contents and order
        contents_text = ''
        order_text = ''
        if target_bin:
            contents = bin_contents[target_bin]
            contents_text = "objects in bin '{}': {}".format(target_bin.upper(), ', '.join(contents))
            order = work_order[target_bin]
            order_text = order
        # recognition result
        result_text = ''
        if recognition_msg[arm] is not None:
            result = recognition_msg[arm].candidates[np.argmax(recognition_msg[arm].probabilities)]
            result_text = result
        msg[arm].text = '''\
{}_arm:
  - state: {}
  - target bin: {}
  - objects in bin: {}
  - target object: {}
  - recognized as: {}'''.format(arm, state_text, target_text, order_text, result_text, result_text)
    pub_l.publish(msg['left'])
    pub_r.publish(msg['right'])
    rate.sleep()
