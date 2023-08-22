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
    rospy.wait_for_service('semi/move_right_arm')
    try:
        move_arm = rospy.ServiceProxy('semi/move_right_arm', MoveArm)
        resp = move_arm(pose)
        return resp.succeeded
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == '__main__':
    rospy.init_node('test_move_right_arm_client')
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    position = Point(x=0.6246819211728672, y=-0.819074585555165, z=0.3265580285317628)
    orientation = Quaternion(x=0.3817200917099027, y=0.92271996211121, z=-0.020952656721378728, w=0.04938247951234029)
    pose = Pose(position=position, orientation=orientation)
    right_pose = PoseStamped(header=hdr, pose=pose)

    print("Requesting {0}, {1}".format(position, orientation))
    if move_arm_client(right_pose) is True:
        print("The move succeeded")
    else:
        print("The move failed")
