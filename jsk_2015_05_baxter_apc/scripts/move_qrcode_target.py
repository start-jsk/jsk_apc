#!/usr/bin/env python
#-*- coding:utf-8 -*-

import rospy
from zbar_ros.msg import ()

def callback(data):




def listener():
    rospy.init_node('move_arm', anonymous=True)

    right = baxter_interface.Limb('right')

    rospy.Subscriber('semi/joint_angle_chatter', String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
