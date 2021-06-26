#!/usr/bin/env python
# -*- coding: utf-8 -*-
# demo_real.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import rospy

from std_msgs.msg import (
    Empty,
    String,
    )
from geometry_msgs.msg import PoseStamped, Pose
from jsk_rviz_plugins.msg import OverlayText

from jsk_2015_05_baxter_apc.msg import *
from jsk_2015_05_baxter_apc.srv import *

class DemoReal(object):
    """Demo for video sending to Amazon Official to get Kiva & Items"""
    def __init__(self):
        rospy.init_node('demo_real')
        # publishers
        self.pb_rviz_msg = rospy.Publisher('/semi/rviz_msg', OverlayText)
        # self.pb_get_item = rospy.Publisher('/semi/get_item', Empty)
        # properties
        self.qrcode_info = {}
        self.target_bin = ''

    def cl_qrcode_reader(self):
        """QR code reader to get the position of bins"""
        rospy.logwarn("======================== cl_qrcode_reader ========================")
        rospy.wait_for_service('/semi/qrcode_pos')
        try:
            qrcode_reader = rospy.ServiceProxy('/semi/qrcode_pos', QrStampsrv)
            resp = qrcode_reader(Empty)
            for stamp in resp.qrstamps.qrcode_stampes:
                rospy.logwarn(stamp.label.data)
                self.qrcode_info[stamp.label.data] = stamp.qrcode_pose_stamp
            return resp
        except rospy.ServiceException as e:
            rospy.logwarn('/semi/qrcode_pos Service call failed: {0}'.format(e))

    def cl_get_item(self):
        rospy.logwarn('move to ' + self.target_bin + " =================================")
        rospy.wait_for_service('/move_right_arm_service')
        try:
            get_item = rospy.ServiceProxy('/move_right_arm_service', MoveArm)
            # get_item = rospy.ServiceProxy('/semi/move_right_arm', MoveArm)

            get_item(self.qrcode_info[self.target_bin])

            # rospy.logwarn(self.qrcode_info)
            # rospy.logwarn(get_item)
            # rospy.logwarn(self.qrcode_info[(self.target_bin)])
            # resp = get_item(self.qrcode_info[(self.target_bin)])
            # rospy.logwarn('get_item ========================================================')
            # rospy.logwarn(resp)

            # rospy.wait_for_service('/semi/get_item')
            # get_item = rospy.ServiceProxy('/semi/get_item', Cue)
            # resp = get_item()
            # resp = get_item(PoseStamped(self.qrcode_info[self.target_bin]))
            # if resp.succeeded is False:
            #     rospy.logwarn('move arm to {0} is failed'.format(self.target_bin))
        except rospy.ServiceException as e:
            rospy.logwarn('/semi/get_item Service call failed: {0}'.format(e))

    def cl_release_item(self):
        rospy.wait_for_service('/semi/release_item')
        rospy.logwarn("===============================release item =====================================")
        try:
            release_item = rospy.ServiceProxy('/semi/release_item', ReleaseItem)
            resp = release_item(Empty)
            if resp.succeeded is False:
                rospy.logwarn('release item is failed'.format(self.target_bin))
        except rospy.ServiceException as e:
            rospy.logwarn('/semi/release_item Service call failed: {0}'.format(e))

    def main(self):
        # read QR code
        self.pb_rviz_msg.publish(OverlayText(text='Started reading QR code and get position of each bins.'))
        succeeded = self.cl_qrcode_reader()
        # Get item
        self.target_bin = 'bin_E'
        self.pb_rviz_msg.publish(OverlayText(text='Getting item in bin name: {0}.'.format(self.target_bin)))
        succeeded = self.cl_get_item()
        # Release item
        self.pb_rviz_msg.publish(OverlayText(text='Releasing item.'))
        self.cl_release_item()
        self.pb_rviz_msg.publish(OverlayText(text="baxter waiting"))

if __name__ == '__main__':
    demo_real = DemoReal()
    demo_real.main()
