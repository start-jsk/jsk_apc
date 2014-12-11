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
# from jsk_2014_picking_challenge.srv import QrStampsrv, MoveArm
from jsk_2014_picking_challenge.srv import *


class DemoReal(object):
    """Demo for video sending to Amazon Official to get Kiva & Items"""
    def __init__(self):
        rospy.init_node('demo_real')
        # publishers
        self.pb_rviz_msg = rospy.Publisher('/semi/rviz_msg', OverlayText)
        # properties
        self.qrcode_info = {}
        self.target_bin = ''

    def cl_qrcode_reader(self):
        """QR code reader to get the position of bins"""
        rospy.wait_for_service('/semi/qrcode._pos')
        try:
            qrcode_reader = rospy.ServiceProxy('/semi/qrcode_pos', QrStampsrv)
            resp = qrcode_reader(Empty)
            # rospy.logwarn(resp.qrstamps.qrcode_poses)
            for stamp in resp.qrstamps.qrcode_poses:
                # rospy.logwarn(stamp)
                self.target_bin = stamp.header.frame_id
                stamp.header.frame_id = ""
                self.qrcode_info[self.target_bin] = stamp
                rospy.logwarn(self.qrcode_info[self.target_bin])
                rospy.logwarn("test =========== " + self.target_bin)
            return resp
        except rospy.ServiceException, e:
            rospy.logwarn('/semi/qrcode_pos Service call failed: {0}'.format(e))

    def cl_get_item(self):
        rospy.wait_for_service('/move_right_arm_service')
        try:
            get_item = rospy.ServiceProxy('/move_right_arm_service', MoveArm)
            rospy.logwarn(get_item)
            rospy.logwarn(self.qrcode_info[(self.target_bin)])
            get_item(self.qrcode_info[(self.target_bin)])
            # resp = get_item(PoseStamped(self.qrcode_info[self.target_bin]))
            # if resp.succeeded is False:
            #     rospy.logwarn('move arm to {0} is failed'.format(self.target_bin))
        except rospy.ServiceException, e:
            rospy.logwarn('/semi/get_item Service call failed: {0}'.format(e))

    def cl_release_item(self):
        rospy.wait_for_service('/semi/release_item')
        try:
            release_item = rospy.ServiceProxy('/semi/release_item', ReleaseItem)
            resp = release_item(Empty)
            if resp.succeeded is False:
                rospy.logwarn('release item is failed'.format(self.target_bin))
        except rospy.ServiceException, e:
            rospy.logwarn('/semi/release_item Service call failed: {0}'.format(e))

    def main(self):
        # read QR code
        self.cl_qrcode_reader()
        self.pb_rviz_msg.publish(OverlayText(text='Started reading QR code and get position of each bins.'))
        # Get item
        self.target_bin = 'bin_F'
        self.pb_rviz_msg.publish(OverlayText(text='Getting item in bin name: {0}.'.format(self.target_bin)))
        succeeded = self.cl_get_item()
        # Release item
        self.pb_rviz_msg.publish(OverlayText(text='Releasing item.'))
        self.cl_release_item()
        self.pb_rviz_msg.publish(OverlayText(text="baxter waiting"))


if __name__ == '__main__':
    demo_real = DemoReal()
    demo_real.main()
