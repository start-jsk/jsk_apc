#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import json
import rospy
from jsk_2016_01_baxter_apc.msg import StowWorkOrder, StowWorkOrderArray
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn


class StowWorkOrderServer():
    def __init__(self):
        self.json_file = rospy.get_param('~json', None)
        self.is_apc2016 = rospy.get_param('~is_apc2016', True)
        self.gripper = rospy.get_param('~gripper', 'gripper2016')
        self.msg = dict(left=StowWorkOrderArray(), right=StowWorkOrderArray())
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        self.pub_left = rospy.Publisher('~left_hand',
                                   StowWorkOrderArray,
                                   queue_size=1)
        self.pub_right = rospy.Publisher('~right_hand',
                                    StowWorkOrderArray,
                                    queue_size=1)
        self.object_data = None
        if self.is_apc2016:
           self.object_data = jsk_apc2016_common.get_object_data()

    def run(self):
        self.sort_work_order()
        rate = rospy.Rate(rospy.get_param('~rate', 1))
        while not rospy.is_shutdown():
            self.pub_left.publish(self.msg['left'])
            self.pub_right.publish(self.msg['right'])
            rate.sleep()

    def sort_work_order(self):
        bin_contents = jsk_apc2016_common.get_bin_contents(self.json_file)
        sorted_bin_list = bin_contents.keys()
        sorted_bin_list = sorted(sorted_bin_list,
                                 key=lambda bin_: len(bin_contents[bin_]))
        for bin_ in sorted_bin_list:
            order = StowWorkOrder(bin=bin_)
            if bin_ in 'abdegj':
                self.msg['left'].array.append(order)
            elif bin_ in 'cfhikl':
                self.msg['right'].array.append(order)


if __name__ == '__main__':
    rospy.init_node('stow_work_order')
    stow_server = StowWorkOrderServer()
    stow_server.run()
