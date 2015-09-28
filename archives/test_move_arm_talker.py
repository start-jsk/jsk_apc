#!/usr/bin/env python
# -*- coding:utf-8 -*-
# test_arm_talker.py
"""
Test script for move_arm_listener.py

How to use
---

    $ rosrun jsk_2015_05_baxter_apc move_arm_listener.py
    # on another terminal
    $ rosrun jsk_2015_05_baxter_apc test_move_arm_talker.py

"""

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header


def talker():
    # initialization
    rospy.init_node('test_move_arm_talker')
    move_larm_pub = rospy.Publisher('semi/move_left_arm', PoseStamped)
    move_rarm_pub = rospy.Publisher('semi/move_right_arm', PoseStamped)
    # left arm pose
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    position=Point(
            x=0.9743684236857688,
            y=0.400063802027193,
            z=0.45121855877314805,
            )
    orientation=Quaternion(
            x=0.05532824006190543,
            y=0.8002393257489635,
            z=0.09079879412889569,
            w=0.5901791137961719,
            )
    pose = Pose(position=position, orientation=orientation)
    left_pose = PoseStamped(header=hdr, pose=pose)
    # right arm pose
    position = Point(x=position.x, y=-position.y, z=position.z)
    pose = Pose(position=position, orientation=orientation)
    right_pose = PoseStamped(header=hdr, pose=pose)
    # publish arm status while running
    while not rospy.is_shutdown():
        # move left arm
        log = "Publishing left_arm_pose:\n{}".format(left_pose)
        rospy.loginfo(log)
        move_larm_pub.publish(left_pose)
        rospy.sleep(1.)
        # move right arm
        log = "Publishing right_arm_pose:\n{}".format(right_pose)
        rospy.loginfo(log)
        move_rarm_pub.publish(right_pose)
        rospy.sleep(1.)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
