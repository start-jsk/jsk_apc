#!/usr/bin/env python
#-*- coding:utf-8 -*-
import roslib; roslib.load_manifest('jsk_2015_05_baxter_apc')

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from jsk_2015_05_baxter_apc.srv import MoveArm


def move_arm_client(pose):
    rospy.wait_for_service('semi/move_left_arm')
    try:
        move_arm = rospy.ServiceProxy('semi/move_left_arm', MoveArm)
        resp = move_arm(pose)
        return resp.succeeded
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == '__main__':
    rospy.init_node('test_move_left_arm_client')
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    position = Point(x=0.6243138656339947, y=0.819321998063199, z=0.32688056167151336)
    orientation = Quaternion(x=-0.3823049553045683, y=0.9224791764636031, z=0.019960757611487724, w=0.04976603556824752)
    pose = Pose(position=position, orientation=orientation)
    left_pose = PoseStamped(header=hdr, pose=pose)

    print("Requesting {0}, {1}".format(position, orientation))
    if move_arm_client(left_pose) is True:
        print("The move succeeded")
    else:
        print("The move failed")
