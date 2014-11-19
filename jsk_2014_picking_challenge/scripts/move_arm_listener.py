#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
"semi/move_arm_chatter"でPublishされているPoseStamped型のメッセージ
を受け取り、PoseStampedで指定されている位置へ右腕を移動させる。

テストの仕方
```
$ rosrun jsk_2014_picking_challenge joint_angle_talker.py
# in another terminal
$ rosrun jsk_2014_picking_challenge move_arm.py
```
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
    ns = "ExternalTools/" + 'right' + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                               resp.result_type)
    if (resp_seeds[0] != resp.RESULT_INVALID):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp_seeds[0], 'None')
        print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print "\nIK Joint Solution:\n", limb_joints
        print "------------------"
        print "Response Message:\n", resp
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    log = rospy.get_name() + ": I heard {}".format(pose)
    rospy.loginfo(log)

    right = baxter_interface.Limb('right')
    rj = right.joint_names()
    if len(resp.joints[0].position) !=0:
        joint_command = {jt_nm: resp.joints[0].position[i]
                for i, jt_nm in enumerate(rj)}
        right.move_to_joint_positions(joint_command)


def listener():
    rospy.init_node('move_arm_listener', anonymous=True)

    # initialize arm position
    right = baxter_interface.Limb('right')

    rospy.Subscriber('semi/move_arm_chatter', PoseStamped, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
