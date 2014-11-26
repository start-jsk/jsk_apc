#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
右腕のエンドエフェクタの位置を出力する
"""
import struct
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

import baxter_interface
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


def callback(pose):
    log = rospy.get_name() + ": I heard {}".format(pose)
    rospy.loginfo(log)

    right = baxter_interface.Limb('right')
    rj = right.joint_names()

    print right.endpoint_pose()
    rospy.sleep(3)


def get_arm_position():
    rospy.init_node('get_right_arm_position', anonymous=True)

    right = baxter_interface.Limb('right')
    print right.endpoint_pose()
    rospy.sleep(3)


if __name__ == '__main__':
    get_arm_position()
