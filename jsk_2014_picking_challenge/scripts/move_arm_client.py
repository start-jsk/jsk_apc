#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
to test move_arm_server.py
"""
import roslib; roslib.load_manifest('jsk_2014_picking_challenge')

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from jsk_2014_picking_challenge.srv import MoveArm


def move_arm_client(pose):
    rospy.wait_for_service('move_arm')
    try:
        move_arm = rospy.ServiceProxy('move_arm', MoveArm)
        resp = move_arm(pose)
        return resp.succeeded
    # try:
    #     add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
    #     resp1 = add_two_ints(x, y)
    #     return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

# def usage():
#     return "%s [x y]" % sys.argv[0]

if __name__ == '__main__':
    # if len(sys.argv) == 3:
    #     x = int(sys.argv[1])
    #     y = int(sys.argv[2])
    # else:
    #     print usage()
    #     sys.exit(1)
    rospy.init_node('move_arm_client')
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    position = Point(x=0.7241942937871891, y=-0.39845057139098894, z=0.5456768943508293)
    orientation = Quaternion(x=0.0017747791414926728, y=0.9587746912729604, z=0.006296746310126851, w=0.2840920493772123)
    pose = Pose(position=position, orientation=orientation)
    right_pose = PoseStamped(header=hdr, pose=pose)

    print "Requesting {0}, {1}".format(position, orientation)
    if move_arm_client(right_pose) is True:
        print "The move succeeded"
    else:
        print "The move failed"
